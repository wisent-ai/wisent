"""
Research Analysis Package

Investigates key research questions using the collected activation data.
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

from .q1_strategy_comparison import (
    analyze_strategy_performance,
    summarize_strategy_results,
)

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
]
