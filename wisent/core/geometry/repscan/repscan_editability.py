"""RepScan Step 5: Editability Analysis via SVD-based null-space diagnostics.

Computes how safely a concept can be edited via null-space constrained
weight modification without disrupting other behaviors.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import torch


@dataclass
class EditabilityResult:
    """Result of editability analysis."""
    editing_capacity: float           # fraction of hidden dim free [0,1]
    effective_preserved_rank: float   # sum of Tikhonov scale factors
    singular_values: List[float]      # top singular values for spectrum
    spectral_decay_rate: float        # exponential decay rate
    steering_survival_ratio: float    # ||P_null @ steering|| / ||steering||
    verdict: str                      # HIGH_CAPACITY / MODERATE_CAPACITY / LOW_CAPACITY
    concept_interference: Optional[Dict[str, float]] = field(default=None)


def _compute_svd_metrics(
    pos: torch.Tensor, epsilon: float,
) -> tuple:
    """Compute SVD and Tikhonov-regularized scale factors.

    Returns:
        (U, S, V, scale_factors, editing_capacity, effective_preserved_rank)
        where scale_factors = S^2 / (S^2 + epsilon).
    """
    pos_np = pos.detach().cpu().float().numpy()
    U, S, Vt = np.linalg.svd(pos_np, full_matrices=False)
    V = Vt.T  # (d, k)

    S_sq = S ** 2
    scale = S_sq / (S_sq + epsilon)

    effective_preserved_rank = float(np.sum(scale))
    hidden_dim = pos_np.shape[1]
    editing_capacity = 1.0 - (effective_preserved_rank / hidden_dim)
    editing_capacity = float(np.clip(editing_capacity, 0.0, 1.0))

    return U, S, V, scale, editing_capacity, effective_preserved_rank


def _compute_spectral_decay_rate(S: np.ndarray) -> float:
    """Least-squares fit of log(S) vs index to get exponential decay rate.

    A steeper negative slope means singular values decay faster,
    indicating the representation is more compressible (more null space).
    """
    S_pos = S[S > 0]
    if len(S_pos) < 2:
        return 0.0
    log_S = np.log(S_pos)
    indices = np.arange(len(S_pos), dtype=np.float64)
    A = np.vstack([indices, np.ones(len(indices))]).T
    result = np.linalg.lstsq(A, log_S, rcond=None)
    slope = result[0][0]
    return float(slope)


def _compute_steering_survival(
    pos: torch.Tensor, neg: torch.Tensor,
    scale: np.ndarray, V: np.ndarray,
) -> float:
    """Project steering direction through null space without materializing P_null.

    Uses efficient formulation: P_null @ v = v - V @ (scale * (V^T @ v))
    where scale are the Tikhonov factors. This avoids building the full
    (d x d) projection matrix.

    Returns:
        steering_survival_ratio = ||P_null @ steering|| / ||steering||
    """
    diff = (pos - neg).detach().cpu().float().numpy()
    steering = diff.mean(axis=0)
    steering_norm = np.linalg.norm(steering)
    if steering_norm < 1e-12:
        return 0.0

    VtS = V.T @ steering
    projected_out = V @ (scale * VtS)
    null_component = steering - projected_out

    survival = float(np.linalg.norm(null_component) / steering_norm)
    return np.clip(survival, 0.0, 1.0)


def _compute_concept_interference(
    pos: torch.Tensor, neg: torch.Tensor,
    cluster_labels: List[int], n_concepts: int, epsilon: float,
) -> Dict[str, float]:
    """Pairwise concept interference via cross-null-space projection.

    For each pair (i, j): compute SVD of concept j's positives,
    project concept i's steering direction through j's null space.
    interference(i,j) = 1 - survival(i through j's null space).

    High interference means editing concept j would disrupt concept i.
    """
    labels = np.array(cluster_labels)
    pos_np = pos.detach().cpu().float().numpy()
    neg_np = neg.detach().cpu().float().numpy()

    concept_steering = {}
    concept_svd = {}
    for c in range(n_concepts):
        mask = labels == c
        if mask.sum() < 2:
            continue
        c_pos = pos_np[mask]
        c_neg = neg_np[mask]
        concept_steering[c] = (c_pos - c_neg).mean(axis=0)

        _, S_c, Vt_c = np.linalg.svd(c_pos, full_matrices=False)
        V_c = Vt_c.T
        S_sq_c = S_c ** 2
        scale_c = S_sq_c / (S_sq_c + epsilon)
        concept_svd[c] = (V_c, scale_c)

    interference = {}
    for i in concept_steering:
        for j in concept_svd:
            if i == j:
                continue
            v_i = concept_steering[i]
            V_j, scale_j = concept_svd[j]
            v_norm = np.linalg.norm(v_i)
            if v_norm < 1e-12:
                interference[f"{i}_{j}"] = 0.0
                continue
            VtV = V_j.T @ v_i
            projected_out = V_j @ (scale_j * VtV)
            null_component = v_i - projected_out
            survival = np.linalg.norm(null_component) / v_norm
            interference[f"{i}_{j}"] = float(np.clip(1.0 - survival, 0.0, 1.0))

    return interference


def _classify_capacity(editing_capacity: float) -> str:
    """Classify editing capacity into verdict categories."""
    if editing_capacity >= 0.7:
        return "HIGH_CAPACITY"
    elif editing_capacity >= 0.4:
        return "MODERATE_CAPACITY"
    return "LOW_CAPACITY"


def test_editability(
    pos: torch.Tensor, neg: torch.Tensor,
    epsilon: float = 1e-6,
    cluster_labels: Optional[List[int]] = None,
    n_concepts: int = 1,
) -> EditabilityResult:
    """Step 5: Editability analysis via SVD-based null-space diagnostics.

    Computes how much of the hidden dimension is free for editing (editing
    capacity), how fast singular values decay (spectral decay), and whether
    the mean steering direction survives null-space projection.

    Args:
        pos: Positive activations (n_samples, hidden_dim)
        neg: Negative activations (n_samples, hidden_dim)
        epsilon: Tikhonov regularization parameter
        cluster_labels: Per-sample concept labels (from decomposition)
        n_concepts: Number of detected concepts

    Returns:
        EditabilityResult with all editability metrics.
    """
    U, S, V, scale, editing_capacity, eff_rank = _compute_svd_metrics(pos, epsilon)
    decay_rate = _compute_spectral_decay_rate(S)
    survival = _compute_steering_survival(pos, neg, scale, V)
    verdict = _classify_capacity(editing_capacity)

    n_top = min(20, len(S))
    top_sv = [float(s) for s in S[:n_top]]

    concept_interference = None
    if cluster_labels is not None and n_concepts > 1:
        concept_interference = _compute_concept_interference(
            pos, neg, cluster_labels, n_concepts, epsilon,
        )

    return EditabilityResult(
        editing_capacity=editing_capacity,
        effective_preserved_rank=eff_rank,
        singular_values=top_sv,
        spectral_decay_rate=decay_rate,
        steering_survival_ratio=survival,
        verdict=verdict,
        concept_interference=concept_interference,
    )
