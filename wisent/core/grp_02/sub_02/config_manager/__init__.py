"""Unified configuration manager for Wisent.

This package provides a centralized system for managing model configurations
including classification, steering, and weight modification settings.
"""

from .types import (
    SerializableConfig,
    NumpyEncoder,
    ClassificationConfig,
    SteeringConfig,
    WeightModificationConfig,
    NestedConfigMixin,
    TaskConfig,
    TraitConfig,
    ModelConfig,
)
from .manager import WisentConfigManager
from .convenience import (
    get_config_manager,
    save_classification_config,
    get_classification_config,
    save_steering_config,
    get_steering_config,
    save_weight_modification_config,
    get_weight_modification_config,
    save_trait_classification_config,
    get_trait_classification_config,
    save_trait_steering_config,
    get_trait_steering_config,
    save_trait_weight_modification_config,
    get_trait_weight_modification_config,
)

# Backward compatibility re-exports
from .backward_compat import (
    OptimizationResult,
    store_optimization,
    get_cached_optimization,
    WeightModificationResult,
    store_weight_modification,
    get_cached_weight_modification,
    get_weight_modification_cache,
    OptimizationCache,
    get_cache,
    ModelConfigManager,
    get_default_manager,
    save_model_config,
    load_model_config,
    get_optimal_parameters,
)

__all__ = [
    # Types
    "SerializableConfig",
    "NumpyEncoder",
    "ClassificationConfig",
    "SteeringConfig",
    "WeightModificationConfig",
    "NestedConfigMixin",
    "TaskConfig",
    "TraitConfig",
    "ModelConfig",
    # Manager
    "WisentConfigManager",
    # Convenience functions
    "get_config_manager",
    "save_classification_config",
    "get_classification_config",
    "save_steering_config",
    "get_steering_config",
    "save_weight_modification_config",
    "get_weight_modification_config",
    "save_trait_classification_config",
    "get_trait_classification_config",
    "save_trait_steering_config",
    "get_trait_steering_config",
    "save_trait_weight_modification_config",
    "get_trait_weight_modification_config",
    # Backward compatibility
    "OptimizationResult",
    "store_optimization",
    "get_cached_optimization",
    "WeightModificationResult",
    "store_weight_modification",
    "get_cached_weight_modification",
    "get_weight_modification_cache",
    "OptimizationCache",
    "get_cache",
    "ModelConfigManager",
    "get_default_manager",
    "save_model_config",
    "load_model_config",
    "get_optimal_parameters",
]
