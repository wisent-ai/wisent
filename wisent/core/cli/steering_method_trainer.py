"""
Unified steering method training interface for CLI commands.

This module provides a unified interface for training any steering method
(CAA, PRISM, PULSE, TITAN) that can be used by the optimize-steering CLI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch

from wisent.core.steering_methods.registry import SteeringMethodRegistry
from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations.core.atoms import ActivationAggregationStrategy, LayerActivations
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.models.core.atoms import SteeringPlan, SteeringVector


def get_method_params_from_args(method_name: str, args) -> Dict[str, Any]:
    """
    Extract method-specific parameters from CLI args.
    
    Args:
        method_name: Name of the steering method (caa, prism, pulse, titan)
        args: Parsed CLI arguments
        
    Returns:
        Dict of method parameters
    """
    params = SteeringMethodRegistry.extract_method_params(method_name, args)
    
    # Handle steering_layers which may be a comma-separated string
    if "steering_layers" in params and isinstance(params["steering_layers"], str):
        params["steering_layers"] = [int(x.strip()) for x in params["steering_layers"].split(",")]
    
    return params


def train_steering_vector_for_layer(
    method_name: str,
    pos_acts: List[torch.Tensor],
    neg_acts: List[torch.Tensor],
    method_params: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Train a steering vector for a single layer using the specified method.
    
    This is the simple interface used by CAA and can be adapted for other methods
    that support per-layer training.
    
    Args:
        method_name: Name of the steering method
        pos_acts: List of positive activation tensors
        neg_acts: List of negative activation tensors
        method_params: Optional method-specific parameters
        
    Returns:
        Steering vector tensor
    """
    method_name = method_name.lower()
    params = method_params or {}
    
    if method_name == "caa":
        from wisent.core.steering_methods.methods.caa import CAAMethod
        method = CAAMethod(**params)
        return method.train_for_layer(pos_acts, neg_acts)
    
    elif method_name == "prism":
        # PRISM can work per-layer by training on stacked tensors
        from wisent.core.steering_methods.methods.prism import PRISMMethod
        method = PRISMMethod(**params)
        
        # Stack activations and train
        pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_acts], dim=0)
        neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_acts], dim=0)
        
        # Train PRISM for this layer
        directions, _ = method._train_layer_directions(pos_tensor, neg_tensor, layer_name="layer")
        
        # Return primary direction (first direction, weighted)
        if directions.shape[0] > 1:
            # Use first direction as primary
            primary = directions[0]
        else:
            primary = directions.squeeze(0)
        
        # Normalize if requested
        if params.get("normalize", True):
            primary = primary / (primary.norm() + 1e-8)
        
        return primary
    
    elif method_name in ("pulse", "titan"):
        # PULSE and TITAN require full ContrastivePairSet - fall back to CAA for per-layer
        # This is a simplification; for full PULSE/TITAN training, use train_steering_vectors
        from wisent.core.steering_methods.methods.caa import CAAMethod
        method = CAAMethod(normalize=True)
        return method.train_for_layer(pos_acts, neg_acts)
    
    else:
        # Default to CAA
        from wisent.core.steering_methods.methods.caa import CAAMethod
        method = CAAMethod(**params)
        return method.train_for_layer(pos_acts, neg_acts)


def train_steering_vectors(
    method_name: str,
    pair_set: ContrastivePairSet,
    method_params: Optional[Dict[str, Any]] = None,
    layers: Optional[List[str]] = None,
) -> LayerActivations:
    """
    Train steering vectors using the full ContrastivePairSet interface.
    
    This is the advanced interface used by PRISM, PULSE, and TITAN that
    operates on entire ContrastivePairSets and returns multi-layer results.
    
    Args:
        method_name: Name of the steering method
        pair_set: ContrastivePairSet with collected activations
        method_params: Optional method-specific parameters
        layers: Optional list of layer names to train on
        
    Returns:
        LayerActivations with steering vectors for each layer
    """
    method_name = method_name.lower()
    params = method_params or {}
    
    # Filter to specified layers if provided
    if layers:
        params["steering_layers"] = [int(l.replace("layer_", "").replace("layer.", "")) for l in layers]
    
    method = SteeringMethodRegistry.create_method_instance(method_name, **params)
    return method.train(pair_set)


def collect_activations_for_pair_set(
    model,
    pair_set: ContrastivePairSet,
    layers: List[str],
    aggregation: ActivationAggregationStrategy = ActivationAggregationStrategy.LAST_TOKEN,
) -> ContrastivePairSet:
    """
    Collect activations for all pairs in a ContrastivePairSet.
    
    Args:
        model: The model to collect activations from
        pair_set: ContrastivePairSet to process
        layers: List of layer names to collect
        aggregation: Token aggregation strategy
        
    Returns:
        Updated ContrastivePairSet with activations attached
    """
    collector = ActivationCollector(model=model, store_device="cpu")
    
    updated_pairs = []
    for pair in pair_set.pairs:
        updated_pair = collector.collect_for_pair(
            pair,
            layers=layers,
            aggregation=aggregation,
            return_full_sequence=False,
            normalize_layers=False,
        )
        updated_pairs.append(updated_pair)
    
    return ContrastivePairSet(name=pair_set.name, pairs=updated_pairs)


def create_steering_plan_from_activations(
    layer_activations: LayerActivations,
    strength: float = 1.0,
    description: str = "Steering",
) -> SteeringPlan:
    """
    Create a SteeringPlan from LayerActivations.
    
    Args:
        layer_activations: LayerActivations containing steering vectors
        strength: Global steering strength multiplier
        description: Description for the steering plan
        
    Returns:
        SteeringPlan ready to apply to model
    """
    layers_dict = {}
    
    for layer_name, vector in layer_activations.to_dict().items():
        if vector is not None:
            # Ensure vector is 1D
            if vector.dim() > 1:
                vector = vector.squeeze()
            layers_dict[layer_name] = SteeringVector(vector=vector, scale=strength)
    
    return SteeringPlan(
        layers=layers_dict,
        layers_description=[f"{description} (strength={strength})"],
    )


def create_steering_plan_single_layer(
    steering_vector: torch.Tensor,
    layer: str,
    strength: float = 1.0,
    description: str = "Steering",
) -> SteeringPlan:
    """
    Create a SteeringPlan for a single layer.
    
    Args:
        steering_vector: The steering vector tensor
        layer: Layer name/index
        strength: Steering strength
        description: Description
        
    Returns:
        SteeringPlan
    """
    layer_str = str(layer)
    steering_vec = SteeringVector(vector=steering_vector, scale=strength)
    
    return SteeringPlan(
        layers={layer_str: steering_vec},
        layers_description=[f"{description} L{layer} S{strength}"],
    )


class SteeringMethodWrapper:
    """
    A wrapper that provides a CAAMethod-compatible interface for any steering method.
    
    This allows drop-in replacement of CAAMethod in existing code:
    
        # Before:
        caa_method = CAAMethod(kwargs={"normalize": True})
        steering_vector = caa_method.train_for_layer(pos_acts, neg_acts)
        
        # After:
        method = create_steering_method(method_name, args)
        steering_vector = method.train_for_layer(pos_acts, neg_acts)
    """
    
    def __init__(self, method_name: str, method_params: Optional[Dict[str, Any]] = None):
        self.method_name = method_name.lower()
        self.method_params = method_params or {}
    
    def train_for_layer(
        self,
        pos_acts: List[torch.Tensor],
        neg_acts: List[torch.Tensor],
    ) -> torch.Tensor:
        """Train steering vector for a single layer - CAAMethod compatible interface."""
        return train_steering_vector_for_layer(
            self.method_name,
            pos_acts,
            neg_acts,
            self.method_params,
        )


def create_steering_method(method_name: str, args=None) -> SteeringMethodWrapper:
    """
    Create a steering method with CAAMethod-compatible interface.
    
    This is a drop-in replacement factory that returns a method object
    that works exactly like CAAMethod but can be any steering method.
    
    Args:
        method_name: Name of the steering method (CAA, PRISM, PULSE, TITAN)
        args: Optional CLI args to extract method-specific parameters
        
    Returns:
        SteeringMethodWrapper with train_for_layer method
        
    Example:
        # Drop-in replacement for CAAMethod
        method = create_steering_method("PRISM", args)
        steering_vector = method.train_for_layer(pos_acts, neg_acts)
    """
    method_name_lower = method_name.lower()
    
    if args is not None:
        method_params = get_method_params_from_args(method_name_lower, args)
    else:
        method_params = {}
    
    return SteeringMethodWrapper(method_name_lower, method_params)


class UnifiedSteeringTrainer:
    """
    Unified trainer that handles all steering methods transparently.
    
    Usage:
        trainer = UnifiedSteeringTrainer(model, method_name="prism", method_params={...})
        
        # Per-layer training (works for all methods)
        vector = trainer.train_for_layer(layer, pos_acts, neg_acts)
        
        # Full training (for PRISM, PULSE, TITAN)
        activations = trainer.train_full(pair_set)
    """
    
    def __init__(
        self,
        model,
        method_name: str = "caa",
        method_params: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.method_name = method_name.lower()
        self.method_params = method_params or {}
        self._collector = None
    
    @property
    def collector(self) -> ActivationCollector:
        if self._collector is None:
            self._collector = ActivationCollector(model=self.model, store_device="cpu")
        return self._collector
    
    def train_for_layer(
        self,
        layer: str,
        pos_acts: List[torch.Tensor],
        neg_acts: List[torch.Tensor],
    ) -> torch.Tensor:
        """Train steering vector for a single layer."""
        return train_steering_vector_for_layer(
            self.method_name,
            pos_acts,
            neg_acts,
            self.method_params,
        )
    
    def train_full(
        self,
        pair_set: ContrastivePairSet,
        layers: Optional[List[str]] = None,
    ) -> LayerActivations:
        """Train steering vectors for all layers using full method."""
        return train_steering_vectors(
            self.method_name,
            pair_set,
            self.method_params,
            layers,
        )
    
    def collect_and_train(
        self,
        pair_set: ContrastivePairSet,
        layers: List[str],
        aggregation: ActivationAggregationStrategy = ActivationAggregationStrategy.LAST_TOKEN,
    ) -> LayerActivations:
        """Collect activations and train in one step."""
        # Collect activations
        updated_set = collect_activations_for_pair_set(
            self.model,
            pair_set,
            layers,
            aggregation,
        )
        
        # Train
        return self.train_full(updated_set, layers)
    
    def create_plan(
        self,
        layer_activations: LayerActivations,
        strength: float = 1.0,
    ) -> SteeringPlan:
        """Create steering plan from trained vectors."""
        return create_steering_plan_from_activations(
            layer_activations,
            strength,
            f"{self.method_name.upper()} steering",
        )
    
    def supports_full_training(self) -> bool:
        """Check if this method supports full ContrastivePairSet training."""
        return self.method_name in ("prism", "pulse", "titan")
    
    def get_method_description(self) -> str:
        """Get description of the current method."""
        definition = SteeringMethodRegistry.get(self.method_name)
        return definition.description
