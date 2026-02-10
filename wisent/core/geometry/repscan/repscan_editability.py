"""RepScan Step 5: Editability Analysis via SVD-based null-space diagnostics.

Computes how safely a concept can be edited via null-space constrained
weight modification without disrupting other behaviors.

Uses a composite editability_score for the verdict to avoid the n << d
tautology where raw editing capacity (1 - n/d) is always trivially high.
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
    editability_score: float = 0.0    # composite score driving verdict
    participation_ratio: float = 0.0  # effective rank via PR of singular values
    warnings: List[str] = field(default_factory=list)


def _compute_svd_metrics(
    pos: torch.Tensor, epsilon: Optional[float] = None,
) -> tuple:
    """Compute SVD with adaptive epsilon based on singular value distribution.

    When epsilon is None, uses median(S^2) as the Tikhonov threshold to
    separate signal directions from noise based on the actual spectrum
    rather than an arbitrary constant.

    Returns:
        (U, S, V, scale, editing_capacity, effective_preserved_rank,
         participation_ratio, adaptive_epsilon, warnings)
    """
    pos_np = pos.detach().cpu().float().numpy()
    n, d = pos_np.shape
    U, S, Vt = np.linalg.svd(pos_np, full_matrices=False)
    V = Vt.T

    S_sq = S ** 2
    sum_s = float(np.sum(S))
    sum_s_sq = float(np.sum(S_sq))
    participation_ratio = (sum_s ** 2) / sum_s_sq if sum_s_sq > 0 else 0.0

    if epsilon is None:
        epsilon = float(np.median(S_sq))

    scale = S_sq / (S_sq + epsilon)
    effective_preserved_rank = float(np.sum(scale))
    editing_capacity = 1.0 - (effective_preserved_rank / d)
    editing_capacity = float(np.clip(editing_capacity, 0.0, 1.0))

    warnings = []
    ratio = n / d
    if ratio < 0.5:
        warnings.append(
            f"Underdetermined regime: n_samples ({n}) << hidden_dim ({d}), "
            f"ratio={ratio:.3f}. Raw SVD rank bounded by n_samples. "
            f"Editing capacity reflects dimensional headroom, not concept quality. "
            f"Rely on editability_score and steering_survival_ratio instead."
        )

    return U, S, V, scale, editing_capacity, effective_preserved_rank, participation_ratio, epsilon, warnings


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
    cluster_labels: List[int], n_concepts: int, epsilon: Optional[float] = None,
) -> Dict[str, float]:
    """Pairwise concept interference via cross-null-space projection.

    For each pair (i, j): compute SVD of concept j's positives,
    project concept i's steering direction through j's null space.
    interference(i,j) = 1 - survival(i through j's null space).

    High interference means editing concept j would disrupt concept i.
    Uses per-concept adaptive epsilon when epsilon is None.
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
        eps_c = float(np.median(S_sq_c)) if epsilon is None else epsilon
        scale_c = S_sq_c / (S_sq_c + eps_c)
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


def _compute_editability_score(
    steering_survival: float, spectral_decay_rate: float,
    participation_ratio: float, n_samples: int,
) -> float:
    """Composite editability score incorporating multiple signals.

    Weights:
    - 50% steering survival: most direct test of editability (does the
      steering direction survive null-space projection?)
    - 25% spectral concentration: PR/n closer to 0 means the concept
      occupies few directions (structured, easier to isolate for editing)
    - 25% spectral sharpness: steeper decay means representation is
      more compressible (larger effective null space for safe editing)

    Returns score in [0, 1] where higher = more editable.
    """
    ss = float(np.clip(steering_survival, 0.0, 1.0))
    concentration = 1.0 - min(participation_ratio / max(n_samples, 1), 1.0)
    sharpness = min(-spectral_decay_rate / 0.05, 1.0) if spectral_decay_rate < 0 else 0.0
    return 0.5 * ss + 0.25 * concentration + 0.25 * sharpness


def _classify_verdict(editability_score: float) -> str:
    """Classify composite editability score into verdict categories."""
    if editability_score >= 0.6:
        return "HIGH_CAPACITY"
    elif editability_score >= 0.35:
        return "MODERATE_CAPACITY"
    return "LOW_CAPACITY"


def test_editability(
    pos: torch.Tensor, neg: torch.Tensor,
    epsilon: Optional[float] = None,
    cluster_labels: Optional[List[int]] = None,
    n_concepts: int = 1,
) -> EditabilityResult:
    """Step 5: Editability analysis via SVD-based null-space diagnostics.

    Computes how much of the hidden dimension is free for editing (editing
    capacity), how fast singular values decay (spectral decay), and whether
    the mean steering direction survives null-space projection.

    The verdict uses a composite editability_score rather than raw editing
    capacity, avoiding the n << d tautology where EC = 1 - n/d is always
    trivially high. When n_samples << hidden_dim, a warning is emitted.

    Args:
        pos: Positive activations (n_samples, hidden_dim)
        neg: Negative activations (n_samples, hidden_dim)
        epsilon: Tikhonov regularization parameter. If None, uses adaptive
                 threshold based on median(S^2).
        cluster_labels: Per-sample concept labels (from decomposition)
        n_concepts: Number of detected concepts

    Returns:
        EditabilityResult with all editability metrics.
    """
    U, S, V, scale, editing_capacity, eff_rank, pr, adaptive_eps, warnings = (
        _compute_svd_metrics(pos, epsilon)
    )
    decay_rate = _compute_spectral_decay_rate(S)
    survival = _compute_steering_survival(pos, neg, scale, V)

    n_samples = pos.shape[0]
    score = _compute_editability_score(survival, decay_rate, pr, n_samples)
    verdict = _classify_verdict(score)

    n_top = min(20, len(S))
    top_sv = [float(s) for s in S[:n_top]]

    concept_interference = None
    if cluster_labels is not None and n_concepts > 1:
        concept_interference = _compute_concept_interference(
            pos, neg, cluster_labels, n_concepts, adaptive_eps,
        )

    return EditabilityResult(
        editing_capacity=editing_capacity,
        effective_preserved_rank=eff_rank,
        singular_values=top_sv,
        spectral_decay_rate=decay_rate,
        steering_survival_ratio=survival,
        verdict=verdict,
        concept_interference=concept_interference,
        editability_score=score,
        participation_ratio=pr,
        warnings=warnings,
    )
