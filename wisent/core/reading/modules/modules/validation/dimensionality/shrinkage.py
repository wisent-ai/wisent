"""Ledoit-Wolf shrinkage covariance estimation for high-dimensional data."""

import numpy as np
from typing import Dict, Any, Tuple


def compute_shrinkage_covariance(
    X: np.ndarray,
    return_shrinkage: bool = True,
    *,
    shrinkage_low: float | None = None,
    shrinkage_moderate: float | None = None,
    shrinkage_high: float | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute shrinkage-regularized covariance using Ledoit-Wolf.

    In high dimensions, sample covariance is poorly estimated.
    Ledoit-Wolf shrinks toward a structured target (scaled identity).

    Args:
        shrinkage_low: Threshold below which shrinkage is considered low.
        shrinkage_moderate: Threshold below which shrinkage is moderate.
        shrinkage_high: Threshold below which shrinkage is high.

    Returns:
        (covariance_matrix, metadata_dict)
    """
    if shrinkage_low is None:
        raise ValueError("shrinkage_low is required")
    if shrinkage_moderate is None:
        raise ValueError("shrinkage_moderate is required")
    if shrinkage_high is None:
        raise ValueError("shrinkage_high is required")
    try:
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf()
        lw.fit(X)

        cov = lw.covariance_
        shrinkage = lw.shrinkage_

        metadata = {
            "shrinkage_intensity": float(shrinkage),
            "covariance_estimator": "ledoit_wolf",
            "interpretation": _interpret_shrinkage(shrinkage, shrinkage_low, shrinkage_moderate, shrinkage_high),
        }

    except Exception:
        # Simple regularized covariance when sklearn unavailable
        n, d = X.shape
        X_centered = X - X.mean(axis=0)
        sample_cov = np.cov(X_centered, rowvar=False)

        trace_cov = np.trace(sample_cov)
        shrinkage = min(1.0, d / n)

        target = np.eye(d) * trace_cov / d
        cov = (1 - shrinkage) * sample_cov + shrinkage * target

        metadata = {
            "shrinkage_intensity": float(shrinkage),
            "covariance_estimator": "simple_shrinkage",
            "interpretation": _interpret_shrinkage(shrinkage, shrinkage_low, shrinkage_moderate, shrinkage_high),
        }

    return cov, metadata


def _interpret_shrinkage(shrinkage: float, shrinkage_low: float, shrinkage_moderate: float, shrinkage_high: float) -> str:
    """Interpret shrinkage intensity."""
    if shrinkage < shrinkage_low:
        return "Low shrinkage: sample covariance is reliable"
    elif shrinkage < shrinkage_moderate:
        return "Moderate shrinkage: some regularization needed"
    elif shrinkage < shrinkage_high:
        return "High shrinkage: sample covariance unreliable"
    else:
        return "Very high shrinkage: severe high-dimensional effects"
