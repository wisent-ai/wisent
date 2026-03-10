"""
Exact Newton solver for WICHER steering in the SVD concept subspace.

The Jacobian of F(z) has structure J = a*I + b*u*u^T where:
  u = z / ||z||
  a = identity + lam/||z_0|| - lam/||z||
  b = lam / ||z||

This admits an analytical inverse via Sherman-Morrison:
  J^{-inv} = (inv_a)*I - b/(a*(a+b)) * u*u^T

Same complexity as Broyden but exact.
Since k is typically small, the exact Jacobian is trivial to compute.
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
)
from ._residual import compute_residual


def _jacobian_inv_apply(
    F_vec: torch.Tensor,
    u: torch.Tensor,
    a: float,
    b: float,
) -> torch.Tensor:
    """
    Apply analytical J^{-inv} to vector F_vec.

    J = a*I + b*u*u^T  =>  J^{-inv} via Sherman-Morrison formula.
    """
    inv_a = STEERING_SCALE_IDENTITY / a
    denom = a * (a + b)
    if abs(denom) < SHERMAN_MORRISON_EPS:
        return inv_a * F_vec
    coeff = b / denom
    u_dot_F = torch.dot(u, F_vec)
    return inv_a * F_vec - coeff * u_dot_F * u


def wicher_newton_step(
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
    Exact Newton solver in SVD concept subspace.

    Uses analytical Jacobian inverse via Sherman-Morrison on
    the structured matrix J = a*I + b*u*u^T.

    Args:
        hidden: [D] or [B, D] hidden state.
        concept_dir: [D] concept direction.
        concept_basis: [k, D] SVD basis vectors.
        component_variances: [k] singular values squared.
        num_steps: Newton iterations.
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

            direction = -_jacobian_inv_apply(F_val, u, a, b)
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
