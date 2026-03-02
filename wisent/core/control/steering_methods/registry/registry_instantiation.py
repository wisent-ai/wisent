"""Three-layer config merge for steering method instantiation.

Merge order (later layers override earlier):
  base: validated method defaults from configs/
  override: model+task config from ~/.wisent/configs/
  explicit: user-provided kwargs

Raises ValueError if any required parameter is missing after merge.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_merged_params(
    definition: Any,
    model_name: Optional[str],
    task_name: Optional[str],
    user_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Build final params via three-layer merge + validation.

    Args:
        definition: SteeringMethodDefinition instance.
        model_name: Optional model identifier for config lookup.
        task_name: Optional task identifier for config lookup.
        user_kwargs: Explicit parameters from the caller.

    Returns:
        Merged parameter dict ready for method class constructor.

    Raises:
        ValueError: If required parameters are missing after merge.
    """
    from wisent.core.control.steering_methods.configs import (
        load_validated_method_defaults,
        load_model_task_config,
    )

    # Base layer: validated method defaults
    params = load_validated_method_defaults(definition.name)

    # Override layer: model+task saved config
    model_task_overrides = load_model_task_config(model_name, task_name)
    if model_task_overrides:
        params.update(model_task_overrides)

    # Inline definition defaults (for params with has_default=True)
    inline_defaults = definition.get_default_params()
    for key, val in inline_defaults.items():
        if key not in params:
            params[key] = val

    # Explicit layer: user-provided kwargs (highest priority)
    params.update(user_kwargs)

    # Validate that all required params are present
    _validate_required_params(definition, params)

    return params


def _validate_required_params(
    definition: Any,
    params: Dict[str, Any],
) -> None:
    """Raise ValueError if any required parameter is missing."""
    missing = []
    for p in definition.parameters:
        if p.required and p.name not in params:
            missing.append(p.name)
    if missing:
        names = ", ".join(missing)
        raise ValueError(
            f"Steering method '{definition.name}' is missing required "
            f"parameter(s): {names}. Supply them explicitly, run the "
            f"optimizer to save a config, or check validated defaults."
        )


def get_method_info_impl(
    definitions: List[Any],
) -> List[Dict[str, Any]]:
    """Build method info dicts (extracted from registry for size)."""
    return [
        {
            "name": d.name,
            "description": d.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type.__name__,
                    "default": p.default if p.has_default else None,
                    "required": p.required,
                    "help": p.help,
                }
                for p in d.parameters
            ],
            "default_strength": d.default_strength,
            "strength_range": d.strength_range,
        }
        for d in definitions
    ]
