"""Null baseline comparison for effective dimensionality.

Compares real effective dimensions against random vectors to determine
if low-dimensional structure is statistically significant.
"""
import numpy as np
import torch
from typing import Dict

from ...analysis.intrinsic_dim import (
    participation_ratio,
    effective_rank,
    stable_rank,
    compute_effective_dimensions,
)


def compute_null_effective_dimensions(
    n_samples: int,
    n_dims: int,
    n_bootstrap: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Compute effective dimensions for random (null) difference vectors.

    This establishes what effective dimension you'd expect by chance.
    If your real data has similar effective_dim to null, there's no structure.
    """
    rng = np.random.RandomState(random_state)

    pr_nulls, er_nulls, sr_nulls = [], [], []

    for i in range(n_bootstrap):
        random_diff = rng.randn(n_samples, n_dims)

        pr_nulls.append(participation_ratio(random_diff))
        er_nulls.append(effective_rank(random_diff))
        sr_nulls.append(stable_rank(random_diff))

    return {
        "participation_ratio_null": float(np.mean(pr_nulls)),
        "effective_rank_null": float(np.mean(er_nulls)),
        "stable_rank_null": float(np.mean(sr_nulls)),
        "participation_ratio_null_std": float(np.std(pr_nulls)),
        "effective_rank_null_std": float(np.std(er_nulls)),
        "stable_rank_null_std": float(np.std(sr_nulls)),
    }


def compute_effective_dimensions_vs_null(
    pos: torch.Tensor,
    neg: torch.Tensor,
    n_bootstrap: int = 10,
    random_state: int = 42,
) -> Dict[str, any]:
    """
    Compute effective dimensions with comparison to null baseline.

    Returns real estimates, null estimates, and z-scores comparing them.
    Significant compression vs null indicates real low-dimensional structure.

    Z-score interpretation:
    - z < -2: Real effective dim is significantly LOWER than null (good - structure exists)
    - z ≈ 0: Real effective dim similar to null (no structure)
    - z > 2: Real effective dim is HIGHER than null (unusual)
    """
    real = compute_effective_dimensions(pos, neg)

    n_samples = real["n_samples"]
    n_dims = real["ambient_dim"]

    null = compute_null_effective_dimensions(n_samples, n_dims, n_bootstrap, random_state)

    def z_score(real_val, null_mean, null_std):
        if null_std < 1e-10:
            return 0.0
        return (real_val - null_mean) / null_std

    pr_z = z_score(real["participation_ratio"], null["participation_ratio_null"], null["participation_ratio_null_std"])
    er_z = z_score(real["effective_rank"], null["effective_rank_null"], null["effective_rank_null_std"])
    sr_z = z_score(real["stable_rank"], null["stable_rank_null"], null["stable_rank_null_std"])

    compression_vs_null = null["effective_rank_null"] / real["effective_rank"] if real["effective_rank"] > 0 else 1.0

    return {
        "real": real,
        "null": null,
        "z_scores": {
            "participation_ratio_z": pr_z,
            "effective_rank_z": er_z,
            "stable_rank_z": sr_z,
        },
        "compression_vs_null": compression_vs_null,
        "significant_structure": pr_z < -2.0 or er_z < -2.0 or sr_z < -2.0,
    }
