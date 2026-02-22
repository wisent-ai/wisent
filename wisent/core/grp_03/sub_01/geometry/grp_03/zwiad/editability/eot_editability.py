"""
EOT-based editability metrics for Zwiad Step 5.

Measures how sensitive each layer's attention distribution is to activation
perturbations, using the connection between attention and one-sided entropic
optimal transport (EOT).

Composite editability: 50% steering_survival + 25% spectral_concentration + 25% spectral_sharpness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EOTEditabilityResult:
    """Result from EOT editability analysis."""
    attention_entropy: float
    jacobian_sensitivity: float
    cost_perturbation_effect: float
    steering_survival: float
    spectral_concentration: float
    spectral_sharpness: float
    composite_editability: float


def compute_attention_entropy(cost: torch.Tensor, temperature: float = 1.0) -> float:
    """
    Shannon entropy of the softmax attention distribution.

    H = -sum_ij P_ij * log(P_ij), where P = softmax(-C / T).
    High entropy => diffuse attention => easier to steer.

    Args:
        cost: [N, M] cost matrix.
        temperature: Softmax temperature.

    Returns:
        Scalar entropy value.
    """
    logits = -cost / temperature
    log_probs = torch.log_softmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)
    row_entropy = -(probs * log_probs).sum(dim=1)
    return row_entropy.mean().item()


def compute_jacobian_sensitivity(cost: torch.Tensor, temperature: float = 1.0) -> float:
    """
    Frobenius norm of the Jacobian d(softmax)/d(cost).

    For softmax P_i = softmax(-C_i/T), the Jacobian of row i is:
    J_i = (1/T) * (diag(P_i) - P_i P_i^T)
    Sensitivity = avg_i ||J_i||_F.

    Args:
        cost: [N, M] cost matrix.
        temperature: Softmax temperature.

    Returns:
        Average Frobenius norm of per-row Jacobian.
    """
    logits = -cost / temperature
    probs = torch.softmax(logits, dim=1)  # [N, M]
    N = probs.shape[0]
    total_norm = 0.0
    for i in range(N):
        p = probs[i]  # [M]
        J = (torch.diag(p) - p.unsqueeze(1) * p.unsqueeze(0)) / temperature
        total_norm += torch.linalg.norm(J, ord="fro").item()
    return total_norm / N


def compute_steering_survival(cost: torch.Tensor, temperature: float = 1.0) -> float:
    """
    Fraction of perturbation that survives the softmax nonlinearity.

    Applies a small random perturbation delta to cost, measures
    ||softmax(-(C+delta)/T) - softmax(-C/T)||_F / ||delta||_F.

    Values close to 1 mean the layer faithfully transmits perturbations.

    Args:
        cost: [N, M] cost matrix.
        temperature: Softmax temperature.

    Returns:
        Survival ratio in [0, 1].
    """
    logits = -cost / temperature
    base_probs = torch.softmax(logits, dim=1)

    delta = torch.randn_like(cost) * 0.01
    perturbed_logits = -(cost + delta) / temperature
    perturbed_probs = torch.softmax(perturbed_logits, dim=1)

    output_change = torch.linalg.norm(perturbed_probs - base_probs, ord="fro").item()
    input_change = torch.linalg.norm(delta, ord="fro").item()

    if input_change < 1e-12:
        return 0.0

    raw_ratio = output_change / input_change
    survival = min(raw_ratio * temperature, 1.0)
    return survival


def compute_spectral_metrics(cost: torch.Tensor) -> tuple[float, float]:
    """
    SVD-based metrics of the cost matrix structure.

    spectral_concentration: fraction of variance in top singular value.
    spectral_sharpness: ratio of top to second singular value (normalized).

    Args:
        cost: [N, M] cost matrix.

    Returns:
        (spectral_concentration, spectral_sharpness) tuple.
    """
    S = torch.linalg.svdvals(cost.float())
    total_variance = (S ** 2).sum().item()
    if total_variance < 1e-12:
        return 0.0, 0.0
    concentration = (S[0] ** 2).item() / total_variance
    sharpness = (S[0] / (S[1] + 1e-12)).item() if len(S) > 1 else 1.0
    sharpness = min(sharpness / (sharpness + 1.0), 1.0)
    return concentration, sharpness


def compute_eot_editability(
    pos: torch.Tensor,
    neg: torch.Tensor,
    q_neg: Optional[torch.Tensor] = None,
    k_pos: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> EOTEditabilityResult:
    """
    Compute full EOT editability metrics for a layer.

    If Q/K projections are provided, uses attention-affinity cost.
    Otherwise, uses Euclidean pairwise distance as a surrogate cost.

    Composite: 50% steering_survival + 25% spectral_concentration + 25% spectral_sharpness.

    Args:
        pos: [N_pos, D] positive activations.
        neg: [N_neg, D] negative activations.
        q_neg: Optional [N_neg, d_k] query projections from negative.
        k_pos: Optional [N_pos, d_k] key projections from positive.
        temperature: Softmax temperature for EOT.

    Returns:
        EOTEditabilityResult with all metrics.
    """
    if q_neg is not None and k_pos is not None:
        d_k = q_neg.shape[-1]
        cost = -(q_neg.float() @ k_pos.float().T) / math.sqrt(d_k)
    else:
        cost = torch.cdist(neg.float(), pos.float())

    entropy = compute_attention_entropy(cost, temperature)
    jacobian = compute_jacobian_sensitivity(cost, temperature)
    survival = compute_steering_survival(cost, temperature)
    concentration, sharpness = compute_spectral_metrics(cost)

    cost_effect = jacobian * survival
    composite = 0.50 * survival + 0.25 * concentration + 0.25 * sharpness

    return EOTEditabilityResult(
        attention_entropy=entropy,
        jacobian_sensitivity=jacobian,
        cost_perturbation_effect=cost_effect,
        steering_survival=survival,
        spectral_concentration=concentration,
        spectral_sharpness=sharpness,
        composite_editability=composite,
    )
