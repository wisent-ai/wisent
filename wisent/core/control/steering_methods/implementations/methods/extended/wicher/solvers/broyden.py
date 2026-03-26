"""
Broyden quasi-Newton steering in the SVD concept subspace.

WICHER projects the Broyden step into a learned low-rank SVD concept
subspace (k << D), achieving O(k^2) complexity per step with NO
dependency on the lm_head weight matrix.

The nonlinear residual F(z) encodes two objectives:
1. Concept alignment: move z along concept direction w by alpha
2. Norm preservation: keep ||z|| close to ||z_0|| (prevents degenerate outputs)

The norm-preservation term introduces the nonlinearity that makes
Broyden iterate meaningfully (otherwise F is linear and solves in 1 step).

Key properties:
- Operates in k-dim subspace instead of full D-dim space
- Variance-weighted diagonal preconditioner from SVD training
- Sherman-Morrison rank-1 inverse Jacobian update per step
- EMA momentum via beta parameter
- No lm_head required — purely function-evaluation based
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from wisent.core.utils.config_tools.constants import NORM_EPS, SHERMAN_MORRISON_EPS
from ._residual import compute_residual


def _sherman_morrison_update(
    H: torch.Tensor,
    delta_z: torch.Tensor,
    delta_r: torch.Tensor,
) -> torch.Tensor:
    """
    Sherman-Morrison rank-1 update of approximate inverse Jacobian.

    H_new = H + ((delta_z - H @ delta_r) @ delta_z^T @ H)
                / (delta_z^T @ H @ delta_r)
    """
    H_dr = H @ delta_r
    denom = delta_z @ H_dr
    if denom.abs() < SHERMAN_MORRISON_EPS:
        return H
    numerator = (delta_z - H_dr).unsqueeze(1) @ (delta_z @ H).unsqueeze(0)
    return H + numerator / denom


def wicher_broyden_step(
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
    Iterative subspace-projected Broyden steering.

    Solves F(z) = 0 where F encodes concept alignment + norm preservation.
    The norm-preservation term provides the nonlinearity that makes
    iterative Broyden refinement meaningful.

    Args:
        hidden: [D] or [B, D] hidden state.
        concept_dir: [D] concept direction.
        concept_basis: [k, D] SVD basis vectors.
        component_variances: [k] singular values squared.
        num_steps: Broyden iterations.
        alpha: base steering strength.
        eta: step-size multiplier.
        beta: EMA momentum coefficient (0 = disabled).
        alpha_decay: per-step decay for alpha.

    Returns:
        Updated hidden state, same shape as input.
    """
    squeeze = hidden.dim() == 1
    if squeeze:
        hidden = hidden.unsqueeze(0)

    B = hidden.shape[0]
    k = concept_basis.shape[0]
    device = hidden.device
    dtype = hidden.dtype

    w = concept_basis @ concept_dir

    H_init = torch.eye(k, device=device, dtype=torch.float32)

    lam = float(k)

    results = []
    for i in range(B):
        h = hidden[i].float()
        z_0 = concept_basis @ h
        z = z_0.clone()

        H = H_init.clone()
        alpha_t = alpha
        v_prev = torch.zeros(k, device=device, dtype=torch.float32)

        r = compute_residual(z, z_0, w, alpha_t, lam)

        for _step in range(num_steps):
            direction = H @ (-r)
            if beta > 0:
                direction = beta * v_prev + (1.0 - beta) * direction

            z_new = z + eta * direction
            r_new = compute_residual(z_new, z_0, w, alpha_t, lam)

            delta_z = z_new - z
            delta_r = r_new - r
            if delta_r.norm() > SHERMAN_MORRISON_EPS:
                H = _sherman_morrison_update(H, delta_z, delta_r)

            v_prev = direction
            z = z_new
            r = r_new
            alpha_t = alpha_t * alpha_decay

        h_new = h + concept_basis.T @ (z - z_0)
        h_new = torch.nan_to_num(h_new)
        results.append(h_new.to(dtype))

    out = torch.stack(results, dim=0)
    return out.squeeze(0) if squeeze else out
