"""Metrics computation for contrastive pair diagnostics."""

from .divergence import compute_divergence_metrics
from .duplicates import compute_duplicate_metrics
from .coverage import compute_coverage_metrics
from .activations import compute_activation_metrics

__all__ = [
    "compute_divergence_metrics",
    "compute_duplicate_metrics",
    "compute_coverage_metrics",
    "compute_activation_metrics",
]
