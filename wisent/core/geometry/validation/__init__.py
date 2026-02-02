"""Validation tests and diagnostics for RepScan.

Includes signal/null tests, effective dimension validation, and curse of dimensionality diagnostics.
"""

from .dimensionality import (
    compute_sample_dimension_ratio,
    compute_effective_sample_size,
    recommend_sample_size,
    recommend_sample_size_from_data,
    compare_extraction_strategies,
    compute_statistical_power,
    compute_shrinkage_covariance,
    DimensionalityDiagnostics,
    run_dimensionality_diagnostics,
    format_diagnostics_report,
)

__all__ = [
    "compute_sample_dimension_ratio",
    "compute_effective_sample_size",
    "recommend_sample_size",
    "recommend_sample_size_from_data",
    "compare_extraction_strategies",
    "compute_statistical_power",
    "compute_shrinkage_covariance",
    "DimensionalityDiagnostics",
    "run_dimensionality_diagnostics",
    "format_diagnostics_report",
]
