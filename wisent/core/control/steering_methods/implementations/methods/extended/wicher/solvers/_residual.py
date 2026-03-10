"""Shared nonlinear residual for all WICHER solvers."""

from __future__ import annotations

import torch
from wisent.core.utils.config_tools.constants import (
    NORM_EPS,
    STEERING_SCALE_IDENTITY,
)


def compute_residual(
    z: torch.Tensor,
    z_0: torch.Tensor,
    w: torch.Tensor,
    alpha: float,
    lam: float,
) -> torch.Tensor:
    """
    Nonlinear residual F(z) for WICHER solvers.

    F(z) = displacement_gap + norm_penalty where:
    - displacement_gap = (z - z_0) - alpha * w
    - norm_penalty = lam * (||z||/||z_0|| - IDENTITY) * z/||z||

    First term: alignment gap (how far from target displacement).
    Second term: norm penalty (nonlinear via ||z||). This creates the
    curvature that iterative solvers need.

    F(z*) = zero when z* is displaced by alpha*w from z_0 AND ||z*|| = ||z_0||.
    """
    displacement_gap = (z - z_0) - alpha * w
    z_norm = z.norm().clamp(min=NORM_EPS)
    z0_norm = z_0.norm().clamp(min=NORM_EPS)
    norm_ratio = z_norm / z0_norm
    norm_penalty = lam * (norm_ratio - STEERING_SCALE_IDENTITY) * (z / z_norm)
    return displacement_gap + norm_penalty
