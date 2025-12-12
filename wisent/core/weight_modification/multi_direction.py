"""
Multi-direction weight modification: Bake TITAN/PRISM/PULSE directions into weights.

These methods learn multiple steering directions per layer. When baking into weights,
we lose the input-dependent gating but keep the multi-directional coverage.

Baking formula:
    W_new = W + Σ(weight_i * direction_i)

Where weights are either:
1. Learned static weights from the method
2. Average activation weights across training data
3. Uniform weights (simple average)
"""

from __future__ import annotations

import torch
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass

from wisent.core.weight_modification.additive import bake_steering_into_weights
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

__all__ = [
    "MultiDirectionConfig",
    "MultiDirectionResult", 
    "train_and_bake_titan",
    "train_and_bake_prism",
    "train_and_bake_pulse",
    "bake_multi_directions",
    "combine_directions",
]

_LOG = setup_logger(__name__)


@dataclass
class MultiDirectionConfig:
    """Configuration for multi-direction weight modification."""
    
    # Method selection
    method: str = "titan"
    """Which method to use: 'titan', 'prism', or 'pulse'"""
    
    # Direction combination
    combination_strategy: str = "learned"
    """How to combine directions: 'learned', 'uniform', 'pca_weighted'"""
    
    # Global scaling
    alpha: float = 1.0
    """Global steering strength multiplier"""
    
    # Components to modify
    components: List[str] = None
    """Components to modify. Default: ['self_attn.o_proj', 'mlp.down_proj']"""
    
    # Method-specific configs (passed through)
    num_directions: int = 5
    """Number of directions per layer"""
    
    optimization_steps: int = 100
    """Training steps for direction optimization"""
    
    # Baking method
    bake_method: str = "bias"
    """How to bake: 'bias' or 'weight'"""
    
    def __post_init__(self):
        if self.components is None:
            self.components = ["self_attn.o_proj", "mlp.down_proj"]


@dataclass
class MultiDirectionResult:
    """Result from multi-direction training and baking."""
    
    method: str
    """Method used (titan/prism/pulse)"""
    
    num_directions: int
    """Number of directions per layer"""
    
    layers_modified: int
    """Number of layers modified"""
    
    directions_per_layer: Dict[str, int]
    """Number of directions found per layer"""
    
    combination_weights: Dict[str, List[float]]
    """Weights used to combine directions per layer"""
    
    effective_vectors: Dict[int, Tensor]
    """Final combined vectors per layer (what was baked)"""
    
    metadata: Dict[str, Any]
    """Additional metadata from training"""


def combine_directions(
    directions: Tensor,
    weights: Optional[Tensor] = None,
    strategy: str = "learned",
) -> Tensor:
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
        # Simple average
        return directions.mean(dim=0)
    
    elif strategy == "learned":
        # Use provided weights (softmax normalized)
        if weights is None:
            weights = torch.ones(num_dirs) / num_dirs
        else:
            weights = torch.softmax(weights, dim=0)
        return (weights.unsqueeze(1) * directions).sum(dim=0)
    
    elif strategy == "pca_weighted":
        # Weight by PCA variance (first direction gets most weight)
        # Assumes directions are ordered by importance
        variance_weights = torch.tensor([1.0 / (i + 1) for i in range(num_dirs)])
        variance_weights = variance_weights / variance_weights.sum()
        return (variance_weights.unsqueeze(1).to(directions.device) * directions).sum(dim=0)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def bake_multi_directions(
    model: Module,
    directions: Dict[str, Tensor],
    weights: Optional[Dict[str, Tensor]] = None,
    config: Optional[MultiDirectionConfig] = None,
) -> MultiDirectionResult:
    """
    Bake multi-direction steering into model weights.
    
    Args:
        model: Model to modify (in-place)
        directions: Per-layer directions {layer_name: [num_dirs, hidden_dim]}
        weights: Optional per-layer weights {layer_name: [num_dirs]}
        config: Configuration
        
    Returns:
        MultiDirectionResult with modification details
    """
    cfg = config or MultiDirectionConfig()
    log = bind(_LOG, method=cfg.method)
    
    # Combine directions into effective vectors
    effective_vectors: Dict[int, Tensor] = {}
    combination_weights: Dict[str, List[float]] = {}
    directions_per_layer: Dict[str, int] = {}
    
    for layer_name, layer_dirs in directions.items():
        # Get layer index
        try:
            layer_idx = int(layer_name.replace("layer_", ""))
        except ValueError:
            layer_idx = int(layer_name)
        
        # Get weights for this layer
        layer_weights = weights.get(layer_name) if weights else None
        
        # Combine directions
        effective = combine_directions(
            layer_dirs,
            layer_weights,
            strategy=cfg.combination_strategy,
        )
        
        effective_vectors[layer_idx] = effective
        directions_per_layer[layer_name] = layer_dirs.shape[0]
        
        # Record combination weights used
        if layer_weights is not None:
            combination_weights[layer_name] = torch.softmax(layer_weights, dim=0).tolist()
        else:
            num_dirs = layer_dirs.shape[0]
            combination_weights[layer_name] = [1.0 / num_dirs] * num_dirs
    
    # Bake into weights
    stats = bake_steering_into_weights(
        model=model,
        steering_vectors=effective_vectors,
        components=cfg.components,
        alpha=cfg.alpha,
        method=cfg.bake_method,
        verbose=True,
    )
    
    return MultiDirectionResult(
        method=cfg.method,
        num_directions=cfg.num_directions,
        layers_modified=stats.get("layers_modified", len(effective_vectors)),
        directions_per_layer=directions_per_layer,
        combination_weights=combination_weights,
        effective_vectors=effective_vectors,
        metadata={"bake_stats": stats},
    )


def train_and_bake_titan(
    model: Module,
    pair_set: "ContrastivePairSet",
    config: Optional[MultiDirectionConfig] = None,
    titan_config: Optional[Dict[str, Any]] = None,
) -> MultiDirectionResult:
    """
    Train TITAN and bake directions into model weights.
    
    Args:
        model: Model to modify
        pair_set: ContrastivePairSet with activations
        config: Multi-direction config
        titan_config: Additional TITAN-specific config
        
    Returns:
        MultiDirectionResult
    """
    from wisent.core.steering_methods.methods.titan import TITANMethod, TITANConfig
    
    cfg = config or MultiDirectionConfig(method="titan")
    
    # Build TITAN config
    t_cfg = TITANConfig(
        num_directions=cfg.num_directions,
        optimization_steps=cfg.optimization_steps,
    )
    if titan_config:
        for k, v in titan_config.items():
            if hasattr(t_cfg, k):
                setattr(t_cfg, k, v)
    
    # Train TITAN
    print(f"\nTraining TITAN with {cfg.num_directions} directions...")
    titan = TITANMethod(config=t_cfg)
    result = titan.train_titan(pair_set)
    
    # Extract directions and weights
    directions = result.directions
    weights = result.direction_weights
    
    print(f"Trained {len(directions)} layers")
    for layer, dirs in directions.items():
        print(f"  {layer}: {dirs.shape[0]} directions")
    
    # Bake into weights
    print(f"\nBaking into weights with strategy: {cfg.combination_strategy}")
    return bake_multi_directions(model, directions, weights, cfg)


def train_and_bake_prism(
    model: Module,
    pair_set: "ContrastivePairSet",
    config: Optional[MultiDirectionConfig] = None,
    prism_config: Optional[Dict[str, Any]] = None,
) -> MultiDirectionResult:
    """
    Train PRISM and bake directions into model weights.
    
    Args:
        model: Model to modify
        pair_set: ContrastivePairSet with activations
        config: Multi-direction config
        prism_config: Additional PRISM-specific config
        
    Returns:
        MultiDirectionResult
    """
    from wisent.core.steering_methods.methods.prism import PRISMMethod, PRISMConfig
    
    cfg = config or MultiDirectionConfig(method="prism")
    
    # Build PRISM config
    p_cfg = PRISMConfig(
        num_directions=cfg.num_directions,
        optimization_steps=cfg.optimization_steps,
    )
    if prism_config:
        for k, v in prism_config.items():
            if hasattr(p_cfg, k):
                setattr(p_cfg, k, v)
    
    # Train PRISM
    print(f"\nTraining PRISM with {cfg.num_directions} directions...")
    prism = PRISMMethod(config=p_cfg)
    result = prism.train_prism(pair_set)
    
    # Extract directions (PRISM doesn't have learned weights, use uniform)
    directions = result.directions
    
    print(f"Trained {len(directions)} layers")
    for layer, dirs in directions.items():
        print(f"  {layer}: {dirs.shape[0]} directions")
    
    # Bake into weights (no learned weights, use strategy)
    print(f"\nBaking into weights with strategy: {cfg.combination_strategy}")
    return bake_multi_directions(model, directions, None, cfg)


def train_and_bake_pulse(
    model: Module,
    pair_set: "ContrastivePairSet",
    config: Optional[MultiDirectionConfig] = None,
    pulse_config: Optional[Dict[str, Any]] = None,
) -> MultiDirectionResult:
    """
    Train PULSE and bake directions into model weights.
    
    PULSE uses single direction per layer with conditional gating.
    When baking, we lose the gating but keep the behavior vectors.
    
    Args:
        model: Model to modify
        pair_set: ContrastivePairSet with activations
        config: Multi-direction config
        pulse_config: Additional PULSE-specific config
        
    Returns:
        MultiDirectionResult
    """
    from wisent.core.steering_methods.methods.pulse import PULSEMethod, PULSEConfig
    
    cfg = config or MultiDirectionConfig(method="pulse")
    
    # Build PULSE config
    pu_cfg = PULSEConfig(
        optimization_steps=cfg.optimization_steps,
    )
    if pulse_config:
        for k, v in pulse_config.items():
            if hasattr(pu_cfg, k):
                setattr(pu_cfg, k, v)
    
    # Train PULSE
    print(f"\nTraining PULSE...")
    pulse = PULSEMethod(config=pu_cfg)
    result = pulse.train_pulse(pair_set)
    
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
    model: Module,
    pair_set: "ContrastivePairSet",
    method: str = "titan",
    config: Optional[MultiDirectionConfig] = None,
    **method_kwargs,
) -> MultiDirectionResult:
    """
    Train specified method and bake into weights.
    
    Args:
        model: Model to modify
        pair_set: ContrastivePairSet with activations
        method: 'titan', 'prism', or 'pulse'
        config: Multi-direction config
        **method_kwargs: Additional method-specific config
        
    Returns:
        MultiDirectionResult
    """
    if config is None:
        config = MultiDirectionConfig(method=method)
    else:
        config.method = method
    
    if method == "titan":
        return train_and_bake_titan(model, pair_set, config, method_kwargs)
    elif method == "prism":
        return train_and_bake_prism(model, pair_set, config, method_kwargs)
    elif method == "pulse":
        return train_and_bake_pulse(model, pair_set, config, method_kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'titan', 'prism', or 'pulse'")
