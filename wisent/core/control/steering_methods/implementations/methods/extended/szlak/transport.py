"""
Geodesic Optimal Transport — math primitives.

Provides:
- sinkhorn(): Sinkhorn-Knopp algorithm for entropic OT (pure torch)
- sinkhorn_one_sided(): One-sided EOT with free target marginal (closed-form row-softmax)
- build_knn_graph(): k-NN adjacency via sklearn NearestNeighbors
- compute_geodesic_cost(): full pipeline from activations to cost matrix
- compute_attention_affinity_cost(): attention-affinity cost C_ij = -q_i·k_j/√d_k
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors
from wisent.core.constants import (
    NORM_EPS,
    SZLAK_SINKHORN_REG,
    SZLAK_MAX_ITER,
    SZLAK_HIGH_REG,
    SZLAK_SPARSE_K,
    SZLAK_GEODESIC_INF_MULTIPLIER,
)

__all__ = [
    "sinkhorn",
    "sinkhorn_one_sided",
    "build_knn_graph",
    "compute_geodesic_cost",
    "compute_attention_affinity_cost",
]


def sinkhorn(
    cost: torch.Tensor,
    reg: float = SZLAK_SINKHORN_REG,
    max_iter: int = SZLAK_MAX_ITER,
    tol: float = NORM_EPS,
) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm for entropic optimal transport.

    Args:
        cost: [N, M] cost matrix (non-negative).
        reg: Entropic regularization parameter (epsilon).
        max_iter: Maximum Sinkhorn iterations.
        tol: Convergence tolerance on marginal error.

    Returns:
        T: [N, M] transport plan (rows sum to 1/N, cols sum to 1/M).
    """
    N, M = cost.shape
    K = torch.exp(-cost / reg)

    # Uniform marginals
    a = torch.ones(N, device=cost.device, dtype=cost.dtype) / N
    b = torch.ones(M, device=cost.device, dtype=cost.dtype) / M

    u = torch.ones(N, device=cost.device, dtype=cost.dtype)
    v = torch.ones(M, device=cost.device, dtype=cost.dtype)

    for _ in range(max_iter):
        u_prev = u.clone()
        u = a / (K @ v)
        v = b / (K.T @ u)
        err = torch.abs(u - u_prev).max().item()
        if err < tol:
            break

    T = torch.diag(u) @ K @ torch.diag(v)
    return T


def sinkhorn_one_sided(
    cost: torch.Tensor,
    reg: float = SZLAK_HIGH_REG,
    max_iter: int = SZLAK_MAX_ITER,
    tol: float = NORM_EPS,
) -> torch.Tensor:
    """
    One-sided entropic OT: free target marginal (per the attention = EOT paper).

    The closed-form solution is T_ij = softmax(-C_i / reg) / N_source.
    This is the row-softmax transport plan where only the source marginal
    is constrained to be uniform.

    Args:
        cost: [N, M] cost matrix.
        reg: Entropic regularization (temperature).
        max_iter: Unused (closed-form), kept for API consistency.
        tol: Unused (closed-form), kept for API consistency.

    Returns:
        T: [N, M] transport plan (rows sum to 1/N).
    """
    N = cost.shape[0]
    # Row-softmax: each row gets softmax(-C_i / reg), then divide by N for marginal
    logits = -cost / reg
    T = torch.softmax(logits, dim=1) / N
    return T


def compute_attention_affinity_cost(
    q_neg: torch.Tensor,
    k_pos: torch.Tensor,
    num_heads: int | None = None,
    num_kv_heads: int | None = None,
) -> torch.Tensor:
    """
    Attention-affinity cost matrix: C_ij = -q_i · k_j / sqrt(d_k).

    From the paper proving attention = one-sided entropic optimal transport.
    Handles GQA models where Q and K have different dimensions by computing
    per-head costs and averaging across heads.

    Args:
        q_neg: [N_neg, q_dim] query projections from negative activations.
        k_pos: [N_pos, k_dim] key projections from positive activations.
        num_heads: Number of attention heads (needed for GQA).
        num_kv_heads: Number of key-value heads (needed for GQA).

    Returns:
        cost: [N_neg, N_pos] attention-affinity cost matrix.
    """
    q_dim = q_neg.shape[-1]
    k_dim = k_pos.shape[-1]
    if q_dim == k_dim:
        return -(q_neg @ k_pos.T) / math.sqrt(q_dim)
    # GQA: Q and K have different dims, compute per-head cost and average
    if num_heads is None or num_kv_heads is None:
        raise ValueError(
            f"Q dim ({q_dim}) != K dim ({k_dim}): num_heads and num_kv_heads required for GQA"
        )
    head_dim = q_dim // num_heads
    groups = num_heads // num_kv_heads
    q = q_neg.reshape(-1, num_heads, head_dim)
    k = k_pos.reshape(-1, num_kv_heads, head_dim)
    k = k.repeat_interleave(groups, dim=1)  # [N_pos, num_heads, head_dim]
    cost = -torch.einsum('ihd,jhd->ij', q, k) / (num_heads * math.sqrt(head_dim))
    return cost


def build_knn_graph(
    X: np.ndarray,
    k: int = SZLAK_SPARSE_K,
) -> csr_matrix:
    """
    Build a symmetric k-NN graph with Euclidean edge weights.

    Args:
        X: [N, D] array of points.
        k: Number of nearest neighbors.

    Returns:
        Sparse symmetric adjacency matrix with Euclidean distances.
    """
    k_actual = min(k, X.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean", algorithm="auto")
    nn.fit(X)
    dist_matrix = nn.kneighbors_graph(mode="distance")
    sym = dist_matrix.maximum(dist_matrix.T)
    return sym


def compute_geodesic_cost(
    neg: torch.Tensor,
    pos: torch.Tensor,
    k: int = SZLAK_SPARSE_K,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Compute geodesic cost matrix between neg and pos activations.

    1. Concatenate X = [neg; pos]
    2. Build k-NN graph on X
    3. Compute all-pairs shortest paths (geodesic distances)
    4. Extract cost block C[i, j] = geodesic(neg_i, pos_j)

    Args:
        neg: [N_neg, D] negative activations (float32 tensor).
        pos: [N_pos, D] positive activations (float32 tensor).
        k: Number of nearest neighbors for graph construction.

    Returns:
        cost: [N_neg, N_pos] geodesic cost matrix (torch tensor).
        geodesic_all: Full [N, N] geodesic distance matrix (numpy).
    """
    N_neg = neg.shape[0]

    X = torch.cat([neg, pos], dim=0).cpu().numpy()
    graph = build_knn_graph(X, k=k)
    geodesic_all = shortest_path(graph, directed=False)

    # Replace inf with large finite value (disconnected components)
    max_finite = geodesic_all[np.isfinite(geodesic_all)].max()
    geodesic_all[~np.isfinite(geodesic_all)] = max_finite * SZLAK_GEODESIC_INF_MULTIPLIER

    # Extract neg-to-pos block
    cost_np = geodesic_all[:N_neg, N_neg:]
    cost = torch.from_numpy(cost_np.astype(np.float32)).to(neg.device)

    return cost, geodesic_all
