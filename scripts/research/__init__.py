"""
Research Analysis Package

Investigates four key research questions using the collected activation data:
1. Which extraction strategy outperforms all the others?
2. Is RepScan effective at predicting steering?
3. What is the maximum we can achieve per benchmark?
4. Is there a unified direction that improves performance over all benchmarks?
"""

from .common import (
    DB_CONFIG,
    ActivationData,
    BenchmarkResults,
    bytes_to_vector,
    load_activations_from_db,
    compute_geometry_metrics,
    compute_steering_accuracy,
)

from .q1_strategy_comparison import analyze_strategy_performance
from .q2_repscan_correlation import analyze_repscan_correlation
from .q3_benchmark_maximum import analyze_per_benchmark_maximum
from .q4_unified_direction import analyze_unified_direction

__all__ = [
    "DB_CONFIG",
    "ActivationData",
    "BenchmarkResults",
    "bytes_to_vector",
    "load_activations_from_db",
    "compute_geometry_metrics",
    "compute_steering_accuracy",
    "analyze_strategy_performance",
    "analyze_repscan_correlation",
    "analyze_per_benchmark_maximum",
    "analyze_unified_direction",
]
