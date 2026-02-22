"""Ledoit-Wolf shrinkage covariance estimation for high-dimensional data."""

import numpy as np
from typing import Dict, Any, Tuple


def compute_shrinkage_covariance(
    X: np.ndarray,
    return_shrinkage: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute shrinkage-regularized covariance using Ledoit-Wolf.

    In high dimensions, sample covariance is poorly estimated.
    Ledoit-Wolf shrinks toward a structured target (scaled identity).

    Returns:
        (covariance_matrix, metadata_dict)
    """
    try:
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf()
        lw.fit(X)

        cov = lw.covariance_
        shrinkage = lw.shrinkage_

        metadata = {
            "shrinkage_intensity": float(shrinkage),
            "covariance_estimator": "ledoit_wolf",
            "interpretation": _interpret_shrinkage(shrinkage),
        }

    except Exception:
        # Fallback to simple regularized covariance
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
            "interpretation": _interpret_shrinkage(shrinkage),
        }

    return cov, metadata


def _interpret_shrinkage(shrinkage: float) -> str:
    """Interpret shrinkage intensity."""
    if shrinkage < 0.1:
        return "Low shrinkage: sample covariance is reliable"
    elif shrinkage < 0.3:
        return "Moderate shrinkage: some regularization needed"
    elif shrinkage < 0.6:
        return "High shrinkage: sample covariance unreliable"
    else:
        return "Very high shrinkage: severe high-dimensional effects"
