"""
Optuna-based Optimization Framework for Wisent

This module provides Optuna-based hyperparameter optimization for both steering and classifier systems:

STEERING OPTIMIZATION:
- Use wisent.core.utils.cli.optimize_steering for steering optimization
- Use wisent.core.utils.cli.optimize for full model optimization

CLASSIFIER OPTIMIZATION:
1. Activation Pre-generation: Efficient caching of model activations
2. Model Training: Optimized logistic regression and MLP classifiers
3. Intelligent Caching: Avoid retraining identical configurations
4. Cross-validation: Robust performance evaluation

Key components:
- Steering metrics: calculate_comprehensive_metrics, evaluate_benchmark_performance
- Classifier: OptunaClassifierOptimizer, GenerationConfig, CacheConfig
"""

# Steering optimization metrics
from wisent.core.utils.config_tools.optuna.steering.metrics import (
    calculate_comprehensive_metrics,
    evaluate_benchmark_performance,
    evaluate_probe_performance,
    generate_performance_summary,
)

__all__ = [
    # Steering metrics
    "calculate_comprehensive_metrics",
    "evaluate_benchmark_performance",
    "evaluate_probe_performance",
    "generate_performance_summary",
]

# Lazy loading for classifier components (may have missing dependencies)
def __getattr__(name):
    """Lazy import for classifier components."""
    classifier_exports = {
        "OptunaClassifierOptimizer": ("wisent.core.utils.config_tools.optuna.classifier", "OptunaClassifierOptimizer"),
        "ClassifierOptimizationConfig": ("wisent.core.utils.config_tools.optuna.classifier", "ClassifierOptimizationConfig"),
        "GenerationConfig": ("wisent.core.utils.config_tools.optuna.classifier", "GenerationConfig"),
        "CacheConfig": ("wisent.core.utils.config_tools.optuna.classifier", "CacheConfig"),
        "ActivationGenerator": ("wisent.core.utils.config_tools.optuna.classifier", "ActivationGenerator"),
        "ClassifierCache": ("wisent.core.utils.config_tools.optuna.classifier", "ClassifierCache"),
        "OptimizationResult": ("wisent.core.utils.config_tools.optuna.classifier", "OptimizationResult"),
    }
    if name in classifier_exports:
        module_path, attr = classifier_exports[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
