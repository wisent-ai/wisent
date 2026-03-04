"""Validated method-level defaults for steering methods.

These configs provide base-layer defaults: empirically validated
general-purpose parameter values. They replace the old pattern of
using arbitrary constants as silent defaults.

Model+task overrides from ~/.wisent/configs/ and explicit user
kwargs take priority over these validated defaults.
"""

from typing import Any, Dict, Optional

from .validated_defaults import VALIDATED_METHOD_DEFAULTS


def load_validated_method_defaults(method_name: str) -> Dict[str, Any]:
    """Load validated defaults for a steering method (base layer).

    Returns only parameters that have been empirically validated as
    good general defaults. Missing parameters must be supplied by
    the optimizer config or the user.
    """
    return dict(VALIDATED_METHOD_DEFAULTS.get(method_name.lower(), {}))


def load_model_task_config(
    model_name: Optional[str],
    task_name: Optional[str],
) -> Dict[str, Any]:
    """Load saved model+task steering config (override layer).

    Reads from ~/.wisent/configs/ via the unified config manager.
    Returns an empty dict when no saved config exists.
    """
    if not model_name:
        return {}

    try:
        from wisent.core.utils.config_tools.config.convenience import (
            get_steering_config,
        )
        config = get_steering_config(model_name, task_name)
        if config is None:
            return {}
        result: Dict[str, Any] = {}
        # Extract all non-None typed fields from SteeringConfig
        _TYPED_FIELDS = [
            "num_directions", "retain_weight", "independence_weight",
            "condition_threshold", "gate_temperature", "max_alpha",
            "gate_hidden_dim", "intensity_hidden_dim", "behavior_weight",
            "sparse_weight", "grom_optimization_steps", "grom_learning_rate",
            "tecza_optimization_steps", "tetno_optimization_steps",
            "sensor_layer", "steering_layers", "min_cosine_similarity",
            "max_cosine_similarity",
        ]
        for field_name in _TYPED_FIELDS:
            val = getattr(config, field_name, None)
            if val is not None:
                result[field_name] = val
        # method_params dict takes priority (most specific)
        if hasattr(config, "method_params") and config.method_params:
            result.update(config.method_params)
        return result
    except Exception:
        return {}


__all__ = [
    "load_validated_method_defaults",
    "load_model_task_config",
    "VALIDATED_METHOD_DEFAULTS",
]
