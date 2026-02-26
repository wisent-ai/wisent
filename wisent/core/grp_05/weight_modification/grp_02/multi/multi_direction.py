"""
Multi-direction weight modification: Bake GROM/TECZA/TETNO directions into weights.

These methods learn multiple steering directions per layer. When baking into weights,
we lose the input-dependent gating but keep the multi-directional coverage.

Baking formula:
    W_new = W + S(weight_i * direction_i)

Where weights are either:
1. Learned static weights from the method
2. Average activation weights across training data
3. Uniform weights (simple average)
"""

from __future__ import annotations

import torch
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass

from wisent.core.weight_modification.methods.additive import bake_steering_into_weights
from wisent.core.constants import DEFAULT_STRENGTH, GROM_NUM_DIRECTIONS, DEFAULT_OPTIMIZATION_STEPS
from wisent.core.cli.cli_logger import setup_logger, bind
from wisent.core.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

__all__ = [
    "MultiDirectionConfig",
    "MultiDirectionResult",
    "train_and_bake_grom",
    "train_and_bake_tecza",
    "train_and_bake_tetno",
    "bake_multi_directions",
    "combine_directions",
]

_LOG = setup_logger(__name__)


@dataclass
class MultiDirectionConfig:
    """Configuration for multi-direction weight modification."""

    method: str = "grom"
    """Which method to use: 'grom', 'tecza', or 'tetno'"""

    combination_strategy: str = "learned"
    """How to combine directions: 'learned', 'uniform', 'pca_weighted'"""

    alpha: float = DEFAULT_STRENGTH
    """Global steering strength multiplier"""

    components: List[str] = None
    """Components to modify. Default: ['self_attn.o_proj', 'mlp.down_proj']"""

    num_directions: int = GROM_NUM_DIRECTIONS
    """Number of directions per layer"""

    optimization_steps: int = DEFAULT_OPTIMIZATION_STEPS
    """Training steps for direction optimization"""

    bake_method: str = "bias"
    """How to bake: 'bias' or 'weight'"""

    def __post_init__(self):
        if self.components is None:
            self.components = ["self_attn.o_proj", "mlp.down_proj"]


@dataclass
class MultiDirectionResult:
    """Result from multi-direction training and baking."""

    method: str
    num_directions: int
    layers_modified: int
    directions_per_layer: Dict[str, int]
    combination_weights: Dict[str, List[float]]
    effective_vectors: Dict[int, "Tensor"]
    metadata: Dict[str, Any]


def combine_directions(
    directions: "Tensor",
    weights: Optional["Tensor"] = None,
    strategy: str = "learned",
) -> "Tensor":
    """
    Combine multiple directions into a single effective direction.

    Args:
        directions: [num_directions, hidden_dim] tensor
        weights: Optional [num_directions] weights
        strategy: 'learned', 'uniform', or 'pca_weighted'

    Returns:
        [hidden_dim] combined direction
    """
    num_dirs = directions.shape[0]

    if strategy == "uniform":
        return directions.mean(dim=0)

    elif strategy == "learned":
        if weights is None:
            weights = torch.ones(num_dirs) / num_dirs
        else:
            weights = torch.softmax(weights, dim=0)
        return (weights.unsqueeze(1) * directions).sum(dim=0)

    elif strategy == "pca_weighted":
        variance_weights = torch.tensor(
            [1.0 / (i + 1) for i in range(num_dirs)])
        variance_weights = variance_weights / variance_weights.sum()
        return (variance_weights.unsqueeze(1).to(directions.device)
                * directions).sum(dim=0)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def bake_multi_directions(
    model: "Module",
    directions: Dict[str, "Tensor"],
    weights: Optional[Dict[str, "Tensor"]] = None,
    config: Optional[MultiDirectionConfig] = None,
) -> MultiDirectionResult:
    """Bake multi-direction steering into model weights."""
    cfg = config or MultiDirectionConfig()
    log = bind(_LOG, method=cfg.method)

    effective_vectors: Dict[int, "Tensor"] = {}
    combination_weights: Dict[str, List[float]] = {}
    directions_per_layer: Dict[str, int] = {}

    for layer_name, layer_dirs in directions.items():
        try:
            layer_idx = int(layer_name.replace("layer_", ""))
        except ValueError:
            layer_idx = int(layer_name)

        layer_weights = weights.get(layer_name) if weights else None

        effective = combine_directions(
            layer_dirs, layer_weights, strategy=cfg.combination_strategy)

        effective_vectors[layer_idx] = effective
        directions_per_layer[layer_name] = layer_dirs.shape[0]

        if layer_weights is not None:
            combination_weights[layer_name] = (
                torch.softmax(layer_weights, dim=0).tolist())
        else:
            num_dirs = layer_dirs.shape[0]
            combination_weights[layer_name] = [1.0 / num_dirs] * num_dirs

    stats = bake_steering_into_weights(
        model=model, steering_vectors=effective_vectors,
        components=cfg.components, alpha=cfg.alpha,
        method=cfg.bake_method, verbose=True)

    return MultiDirectionResult(
        method=cfg.method, num_directions=cfg.num_directions,
        layers_modified=stats.get("layers_modified", len(effective_vectors)),
        directions_per_layer=directions_per_layer,
        combination_weights=combination_weights,
        effective_vectors=effective_vectors,
        metadata={"bake_stats": stats})


def train_and_bake_grom(
    model: "Module",
    pair_set: "ContrastivePairSet",
    config: Optional[MultiDirectionConfig] = None,
    grom_config: Optional[Dict[str, Any]] = None,
) -> MultiDirectionResult:
    """Train GROM and bake directions into model weights."""
    from wisent.core.steering_methods.methods.grom import GROMMethod, GROMConfig

    cfg = config or MultiDirectionConfig(method="grom")

    t_cfg = GROMConfig(
        num_directions=cfg.num_directions,
        optimization_steps=cfg.optimization_steps)
    if grom_config:
        for k, v in grom_config.items():
            if hasattr(t_cfg, k):
                setattr(t_cfg, k, v)

    print(f"\nTraining GROM with {cfg.num_directions} directions...")
    grom = GROMMethod(config=t_cfg)
    result = grom.train_grom(pair_set)

    directions = result.directions
    weights = result.direction_weights

    print(f"Trained {len(directions)} layers")
    for layer, dirs in directions.items():
        print(f"  {layer}: {dirs.shape[0]} directions")

    print(f"\nBaking into weights with strategy: {cfg.combination_strategy}")
    return bake_multi_directions(model, directions, weights, cfg)


# Re-export from helpers
from wisent.core.weight_modification.multi._multi_direction_helpers import (
    train_and_bake_tecza,
    train_and_bake_tetno,
    train_and_bake,
)
