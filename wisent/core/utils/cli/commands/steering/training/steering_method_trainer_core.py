"""Core training functions for steering methods."""
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from wisent.core.primitives.model_interface.core.activations.activations_collector import ActivationCollector
from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.primitives.models.core.atoms import SteeringPlan, SteeringVector
from wisent.core.control.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.utils.config_tools.constants import NORM_EPS, ARCHITECTURE_MODULE_LIMIT

logger = logging.getLogger(__name__)


def get_method_params_from_args(method_name: str, args) -> Dict[str, Any]:
    """
    Extract method-specific parameters from CLI args.
    
    Args:
        method_name: Name of the steering method (caa, tecza, tetno, grom)
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
        from wisent.core.control.steering_methods.methods.caa import CAAMethod
        method = CAAMethod(**params)
        return method.train_for_layer(pos_acts, neg_acts)
    
    elif method_name == "tecza":
        # TECZA can work per-layer by training on stacked tensors
        from wisent.core.control.steering_methods.methods.advanced import TECZAMethod
        method = TECZAMethod(**params)
        
        # Stack activations and train
        pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_acts], dim=0)
        neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_acts], dim=0)
        
        # Train TECZA for this layer
        directions, _ = method._train_layer_directions(pos_tensor, neg_tensor, layer_name="layer", log_interval=method.log_interval)
        
        # Return primary direction (first direction, weighted)
        if directions.shape[0] > 1:
            # Use first direction as primary
            primary = directions[0]
        else:
            primary = directions.squeeze(0)
        
        # Normalize if requested
        if params.get("normalize", True):
            primary = primary / (primary.norm() + NORM_EPS)
        
        return primary
    
    elif method_name in ("tetno", "grom"):
        # TETNO and GROM require full ContrastivePairSet - fall back to CAA for per-layer
        # This is a simplification; for full TETNO/GROM training, use train_steering_vectors
        from wisent.core.control.steering_methods.methods.caa import CAAMethod
        method = CAAMethod(normalize=True)
        return method.train_for_layer(pos_acts, neg_acts)
    
    else:
        # Default to CAA
        from wisent.core.control.steering_methods.methods.caa import CAAMethod
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
    
    This is the advanced interface used by TECZA, TETNO, and GROM that
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
    aggregation: ExtractionStrategy = ExtractionStrategy.CHAT_LAST,
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
    collector = ActivationCollector(model=model, architecture_module_limit=ARCHITECTURE_MODULE_LIMIT)
    
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
    strength: float,
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
    strength: float,
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

