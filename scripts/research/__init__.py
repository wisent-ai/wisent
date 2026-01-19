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
    STEERING_METHODS,
    RESEARCH_MODELS,
    ActivationData,
    BenchmarkResults,
    bytes_to_vector,
    get_model_info,
    get_all_models_with_activations,
    load_activations_from_db,
    compute_geometry_metrics,
    compute_steering_accuracy,
    compute_steering_vector,
)

from .q1_strategy_comparison import analyze_strategy_performance, summarize_strategy_results
from .q2_repscan_correlation import analyze_repscan_correlation, interpret_correlations
from .q3_benchmark_maximum import analyze_per_benchmark_maximum
from .q4_unified_direction import analyze_unified_direction
from .run_all import run_layer_analysis, run_model_analysis, run_all_models

__all__ = [
    "DB_CONFIG",
    "STEERING_METHODS",
    "RESEARCH_MODELS",
    "ActivationData",
    "BenchmarkResults",
    "bytes_to_vector",
    "get_model_info",
    "get_all_models_with_activations",
    "load_activations_from_db",
    "compute_geometry_metrics",
    "compute_steering_accuracy",
    "compute_steering_vector",
    "analyze_strategy_performance",
    "summarize_strategy_results",
    "analyze_repscan_correlation",
    "interpret_correlations",
    "analyze_per_benchmark_maximum",
    "analyze_unified_direction",
    "run_layer_analysis",
    "run_model_analysis",
    "run_all_models",
]
