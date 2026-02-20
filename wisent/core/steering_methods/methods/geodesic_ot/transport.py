"""
Geodesic Optimal Transport — math primitives.

Provides:
- sinkhorn(): Sinkhorn-Knopp algorithm for entropic OT (pure torch)
- build_knn_graph(): k-NN adjacency via sklearn NearestNeighbors
- compute_geodesic_cost(): full pipeline from activations to cost matrix
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "sinkhorn",
    "build_knn_graph",
    "compute_geodesic_cost",
]


def sinkhorn(
    cost: torch.Tensor,
    reg: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-8,
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


def build_knn_graph(
    X: np.ndarray,
    k: int = 10,
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
    k: int = 10,
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
    geodesic_all[~np.isfinite(geodesic_all)] = max_finite * 2.0

    # Extract neg-to-pos block
    cost_np = geodesic_all[:N_neg, N_neg:]
    cost = torch.from_numpy(cost_np.astype(np.float32)).to(neg.device)

    return cost, geodesic_all
