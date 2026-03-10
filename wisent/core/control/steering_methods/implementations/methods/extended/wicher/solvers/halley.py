"""
Third-order Halley solver for WICHER steering in the SVD concept subspace.

Extends the Newton step with a Halley correction using the directional
Hessian of the residual F(z). Achieves cubic convergence rate versus
quadratic for Newton, potentially reaching the fixed point in fewer
iterations.

Algorithm per step:
  d_N = -J^{-inv} F                          (Newton step)
  dJ_dN = (lam/n^sq) * [DOT_FACTOR*(u^T d_N)*d_N
           + (||d_N||^sq - NORM_FACTOR*(u^T d_N)^sq)*u]
  halley_corr = HALF * J^{-inv} @ dJ_dN
  z_new = z + eta * (d_N - halley_corr)

Safety: clamp ||halley_corr|| <= ||d_N|| to prevent overshoot.
"""

from __future__ import annotations

import torch
from wisent.core.utils.config_tools.constants import (
    NORM_EPS,
    SHERMAN_MORRISON_EPS,
    STEERING_SCALE_IDENTITY,
    INDEX_FIRST,
    NDIM_VECTOR,
    BINARY_CLASS_NEGATIVE,
    ZERO_THRESHOLD,
    HALLEY_CORRECTION_HALF,
    HALLEY_HESSIAN_DOT_FACTOR,
    HALLEY_HESSIAN_NORM_FACTOR,
)
from ._residual import compute_residual
from .newton import _jacobian_inv_apply


def _directional_hessian(
    d_N: torch.Tensor,
    u: torch.Tensor,
    lam: float,
    z_norm: float,
) -> torch.Tensor:
    """
    Compute Hessian-vector product H(F) * d_N along the Newton direction.

    The Hessian of the norm-penalty term gives:
    dJ * d_N = (lam/n^sq) * [DOT_FACTOR*(u^T d_N)*d_N
               + (||d_N||^sq - NORM_FACTOR*(u^T d_N)^sq)*u]
    """
    n_sq = z_norm * z_norm
    if n_sq < ZERO_THRESHOLD:
        return torch.zeros_like(d_N)

    u_dot_dN = torch.dot(u, d_N)
    dN_norm_sq = torch.dot(d_N, d_N)

    scale = lam / n_sq
    term_a = HALLEY_HESSIAN_DOT_FACTOR * u_dot_dN * d_N
    term_b = (dN_norm_sq - HALLEY_HESSIAN_NORM_FACTOR * u_dot_dN * u_dot_dN) * u
    return scale * (term_a + term_b)


def wicher_halley_step(
    hidden: torch.Tensor,
    concept_dir: torch.Tensor,
    concept_basis: torch.Tensor,
    component_variances: torch.Tensor,
    num_steps: int,
    alpha: float,
    eta: float,
    beta: float,
    alpha_decay: float,
) -> torch.Tensor:
    """
    Third-order Halley solver in SVD concept subspace.

    Newton step plus Halley correction using directional Hessian.
    Safeguard clamps correction norm to prevent overshoot.

    Args:
        hidden: [D] or [B, D] hidden state.
        concept_dir: [D] concept direction.
        concept_basis: [k, D] SVD basis vectors.
        component_variances: [k] singular values squared.
        num_steps: Halley iterations.
        alpha: base steering strength.
        eta: step-size multiplier.
        beta: EMA momentum coefficient (zero = disabled).
        alpha_decay: per-step decay for alpha.

    Returns:
        Updated hidden state, same shape as input.
    """
    squeeze = hidden.dim() == NDIM_VECTOR
    if squeeze:
        hidden = hidden.unsqueeze(INDEX_FIRST)

    B = hidden.shape[INDEX_FIRST]
    k = concept_basis.shape[INDEX_FIRST]
    device = hidden.device
    dtype = hidden.dtype

    w = concept_basis @ concept_dir
    lam = float(k)

    results = []
    for i in range(B):
        h = hidden[i].float()
        z_0 = concept_basis @ h
        z = z_0.clone()
        alpha_t = alpha
        v_prev = torch.zeros(k, device=device, dtype=torch.float32)

        for _step in range(num_steps):
            F_val = compute_residual(z, z_0, w, alpha_t, lam)

            z_norm = z.norm().clamp(min=NORM_EPS)
            z0_norm = z_0.norm().clamp(min=NORM_EPS)
            u = z / z_norm

            a = (STEERING_SCALE_IDENTITY
                 + lam / z0_norm - lam / z_norm)
            b = lam / z_norm

            d_N = -_jacobian_inv_apply(F_val, u, a, b)

            dJ_dN = _directional_hessian(d_N, u, lam, z_norm.item())
            halley_raw = _jacobian_inv_apply(dJ_dN, u, a, b)
            halley_corr = HALLEY_CORRECTION_HALF * halley_raw

            dN_norm = d_N.norm()
            corr_norm = halley_corr.norm()
            if corr_norm > dN_norm and dN_norm > NORM_EPS:
                halley_corr = halley_corr * (dN_norm / corr_norm)

            direction = d_N - halley_corr
            if beta > BINARY_CLASS_NEGATIVE:
                direction = (beta * v_prev
                             + (STEERING_SCALE_IDENTITY - beta) * direction)

            z = z + eta * direction
            v_prev = direction
            alpha_t = alpha_t * alpha_decay

        h_new = h + concept_basis.T @ (z - z_0)
        results.append(h_new.to(dtype))

    out = torch.stack(results, dim=INDEX_FIRST)
    return out.squeeze(INDEX_FIRST) if squeeze else out
