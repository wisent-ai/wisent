"""Curse of dimensionality diagnostics and power analysis.

Provides statistical safeguards against high-dimensional pathologies:
1. Sample-to-dimension ratio checks
2. Effective sample size estimation
3. Statistical power analysis
4. Shrinkage covariance estimation
5. Degrees of freedom tracking
"""

from .sample_adequacy import (
    compute_sample_dimension_ratio,
    compute_effective_sample_size,
    recommend_sample_size,
    recommend_sample_size_from_data,
    compare_extraction_strategies,
)
from .power_analysis import compute_statistical_power
from .shrinkage import compute_shrinkage_covariance
from .diagnostics import (
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
