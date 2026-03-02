"""Extracted multi-direction helpers: train_and_bake_tecza, train_and_bake_tetno, train_and_bake."""

from __future__ import annotations

import torch
from typing import Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

from wisent.core.weight_modification.multi.multi_direction import (
    MultiDirectionConfig,
    MultiDirectionResult,
    bake_multi_directions,
)


def train_and_bake_tecza(
    model: "Module",
    pair_set: "ContrastivePairSet",
    config: Optional[MultiDirectionConfig] = None,
    tecza_config: Optional[Dict[str, Any]] = None,
) -> MultiDirectionResult:
    """
    Train TECZA and bake directions into model weights.

    Args:
        model: Model to modify
        pair_set: ContrastivePairSet with activations
        config: Multi-direction config
        tecza_config: Additional TECZA-specific config

    Returns:
        MultiDirectionResult
    """
    from wisent.core.control.steering_methods.methods.advanced import (
        TECZAMethod, TECZAConfig)

    cfg = config or MultiDirectionConfig(method="tecza")

    # Build TECZA config
    p_cfg = TECZAConfig(
        num_directions=cfg.num_directions,
        optimization_steps=cfg.optimization_steps,
    )
    if tecza_config:
        for k, v in tecza_config.items():
            if hasattr(p_cfg, k):
                setattr(p_cfg, k, v)

    # Train TECZA
    print(f"\nTraining TECZA with {cfg.num_directions} directions...")
    tecza = TECZAMethod(config=p_cfg)
    result = tecza.train_tecza(pair_set)

    # Extract directions (TECZA has no learned weights, use uniform)
    directions = result.directions

    print(f"Trained {len(directions)} layers")
    for layer, dirs in directions.items():
        print(f"  {layer}: {dirs.shape[0]} directions")

    # Bake into weights (no learned weights, use strategy)
    print(f"\nBaking into weights with strategy: {cfg.combination_strategy}")
    return bake_multi_directions(model, directions, None, cfg)


def train_and_bake_tetno(
    model: "Module",
    pair_set: "ContrastivePairSet",
    config: Optional[MultiDirectionConfig] = None,
    tetno_config: Optional[Dict[str, Any]] = None,
) -> MultiDirectionResult:
    """
    Train TETNO and bake directions into model weights.

    TETNO uses single direction per layer with conditional gating.
    When baking, we lose the gating but keep the behavior vectors.

    Args:
        model: Model to modify
        pair_set: ContrastivePairSet with activations
        config: Multi-direction config
        tetno_config: Additional TETNO-specific config

    Returns:
        MultiDirectionResult
    """
    from wisent.core.control.steering_methods.methods.advanced import (
        TETNOMethod, TETNOConfig)

    cfg = config or MultiDirectionConfig(method="tetno")

    # Build TETNO config
    pu_cfg = TETNOConfig(
        optimization_steps=cfg.optimization_steps,
    )
    if tetno_config:
        for k, v in tetno_config.items():
            if hasattr(pu_cfg, k):
                setattr(pu_cfg, k, v)

    # Train TETNO
    print(f"\nTraining TETNO...")
    tetno = TETNOMethod(config=pu_cfg)
    result = tetno.train_tetno(pair_set)

    # Extract behavior vectors and layer weights
    behavior_vectors = result.behavior_vectors
    layer_scales = result.layer_scales

    print(f"Trained {len(behavior_vectors)} layers")

    # Convert to multi-direction format (single direction per layer)
    directions = {}
    weights = {}
    for layer, vec in behavior_vectors.items():
        directions[layer] = vec.unsqueeze(0)  # [1, hidden_dim]
        if layer_scales and layer in layer_scales:
            weights[layer] = torch.tensor([layer_scales[layer]])
        else:
            weights[layer] = torch.tensor([1.0])

    # Bake into weights
    print(f"\nBaking into weights...")
    return bake_multi_directions(model, directions, weights, cfg)


def train_and_bake(
    model: "Module",
    pair_set: "ContrastivePairSet",
    method: str,
    config: Optional[MultiDirectionConfig] = None,
    **method_kwargs,
) -> MultiDirectionResult:
    """
    Train specified method and bake into weights.

    Args:
        model: Model to modify
        pair_set: ContrastivePairSet with activations
        method: 'grom', 'tecza', or 'tetno'
        config: Multi-direction config
        **method_kwargs: Additional method-specific config

    Returns:
        MultiDirectionResult
    """
    # Import train_and_bake_grom locally to avoid circular imports
    from wisent.core.weight_modification.multi.multi_direction import (
        train_and_bake_grom,
    )

    if config is None:
        config = MultiDirectionConfig(method=method)
    else:
        config.method = method

    if method == "grom":
        return train_and_bake_grom(model, pair_set, config, method_kwargs)
    elif method == "tecza":
        return train_and_bake_tecza(model, pair_set, config, method_kwargs)
    elif method == "tetno":
        return train_and_bake_tetno(model, pair_set, config, method_kwargs)
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'grom', 'tecza', or 'tetno'")
