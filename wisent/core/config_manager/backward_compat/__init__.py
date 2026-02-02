"""Backward compatibility layer for legacy config manager APIs."""

from .optimization_result import (
    OptimizationResult,
    store_optimization,
    get_cached_optimization,
)
from .weight_modification_result import (
    WeightModificationResult,
    store_weight_modification,
    get_cached_weight_modification,
    get_weight_modification_cache,
)
from .optimization_cache import OptimizationCache, get_cache
from .model_config_manager import (
    ModelConfigManager,
    get_default_manager,
    save_model_config,
    load_model_config,
    get_optimal_parameters,
)

__all__ = [
    # OptimizationResult
    "OptimizationResult",
    "store_optimization",
    "get_cached_optimization",
    # WeightModificationResult
    "WeightModificationResult",
    "store_weight_modification",
    "get_cached_weight_modification",
    "get_weight_modification_cache",
    # OptimizationCache
    "OptimizationCache",
    "get_cache",
    # ModelConfigManager
    "ModelConfigManager",
    "get_default_manager",
    "save_model_config",
    "load_model_config",
    "get_optimal_parameters",
]
