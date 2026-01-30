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
from wisent.core.activations.extraction_strategy import ExtractionStrategy
from wisent.core.activations.core.atoms import LayerActivations
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
    collector = ActivationCollector(model=model)
    
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
            self._collector = ActivationCollector(model=self.model)
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
        aggregation: ExtractionStrategy = ExtractionStrategy.CHAT_LAST,
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


def load_optimal_steering_config(
    model_name: str,
    task_name: str,
    method: str = "*",
) -> Optional[Dict[str, Any]]:
    """
    Load the optimal steering configuration for a model/task from cache.
    
    Args:
        model_name: Name of the model (e.g., "Qwen/Qwen2-7B")
        task_name: Name of the task (e.g., "truthfulqa_mc1")
        method: Steering method to load ("*" for any, or specific like "CAA", "PRISM")
        
    Returns:
        Dict with optimal steering parameters including method-specific params, or None if not found
        
    Example:
        >>> config = load_optimal_steering_config("Qwen/Qwen2-7B", "truthfulqa_mc1")
        >>> if config:
        ...     print(f"Best: {config['method']} at layer {config['layer']} with strength {config['strength']}")
    """
    from wisent.core.config_manager import get_cached_optimization
    
    result = get_cached_optimization(model_name, task_name, method)
    if result is None:
        return None
    
    # Convert OptimizationResult to dict with all params
    config = {
        "model": result.model,
        "task": result.task,
        "method": result.method,
        "layer": result.layer,
        "strength": result.strength,
        "strategy": result.strategy,
        "token_aggregation": result.token_aggregation,
        "prompt_strategy": result.prompt_strategy,
        "score": result.score,
        "metric": result.metric,
    }
    
    # Add method-specific parameters based on method
    if result.method.upper() == "PRISM":
        config.update({
            "num_directions": result.num_directions,
            "direction_weighting": result.direction_weighting,
            "retain_weight": result.retain_weight,
            "independence_weight": result.independence_weight,
            "optimization_steps": result.prism_optimization_steps,
            "use_caa_init": result.use_caa_init,
            "cone_constraint": result.cone_constraint,
            "min_cosine_similarity": result.min_cosine_similarity,
            "max_cosine_similarity": result.max_cosine_similarity,
        })
    elif result.method.upper() == "PULSE":
        config.update({
            "sensor_layer": result.sensor_layer,
            "steering_layers": result.steering_layers,
            "condition_threshold": result.condition_threshold,
            "gate_temperature": result.gate_temperature,
            "per_layer_scaling": result.per_layer_scaling,
            "use_entropy_scaling": result.use_entropy_scaling,
            "max_alpha": result.max_alpha,
            "learn_threshold": result.learn_threshold,
            "optimization_steps": result.pulse_optimization_steps,
        })
    elif result.method.upper() == "TITAN":
        config.update({
            "num_directions": result.num_directions,
            "sensor_layer": result.sensor_layer,
            "steering_layers": result.steering_layers,
            "gate_hidden_dim": result.gate_hidden_dim,
            "intensity_hidden_dim": result.intensity_hidden_dim,
            "behavior_weight": result.behavior_weight,
            "retain_weight": result.retain_weight,
            "sparse_weight": result.sparse_weight,
            "max_alpha": result.max_alpha,
            "optimization_steps": result.titan_optimization_steps,
            "learning_rate": result.titan_learning_rate,
        })
    
    # Also include raw method_params if any
    if result.method_params:
        config["method_params"] = result.method_params
    
    return config


def create_steering_method_from_config(config: Dict[str, Any]) -> Any:
    """
    Create a steering method instance from a loaded config.
    
    Args:
        config: Config dict from load_optimal_steering_config()
        
    Returns:
        Instantiated steering method ready for training
        
    Example:
        >>> config = load_optimal_steering_config("Qwen/Qwen2-7B", "truthfulqa_mc1")
        >>> method = create_steering_method_from_config(config)
        >>> vector = method.train_for_layer(pos_acts, neg_acts)
    """
    method_name = config.get("method", "CAA").lower()
    
    # Extract method-specific params
    params = {}
    
    if method_name == "caa":
        params = {"normalize": True}
        
    elif method_name == "prism":
        params = {
            "num_directions": config.get("num_directions", 1),
            "direction_weighting": config.get("direction_weighting", "primary_only"),
            "retain_weight": config.get("retain_weight", 0.0),
            "independence_weight": config.get("independence_weight", 0.05),
            "optimization_steps": config.get("optimization_steps", 100),
            "use_caa_init": config.get("use_caa_init", True),
            "cone_constraint": config.get("cone_constraint", True),
            "min_cosine_similarity": config.get("min_cosine_similarity", 0.3),
            "max_cosine_similarity": config.get("max_cosine_similarity", 0.95),
        }
        
    elif method_name == "pulse":
        params = {
            "sensor_layer": config.get("sensor_layer", -1),
            "steering_layers": config.get("steering_layers", ""),
            "condition_threshold": config.get("condition_threshold", 0.5),
            "gate_temperature": config.get("gate_temperature", 0.5),
            "per_layer_scaling": config.get("per_layer_scaling", True),
            "use_entropy_scaling": config.get("use_entropy_scaling", False),
            "max_alpha": config.get("max_alpha", 2.0),
            "learn_threshold": config.get("learn_threshold", True),
            "optimization_steps": config.get("optimization_steps", 100),
        }
        
    elif method_name == "titan":
        params = {
            "num_directions": config.get("num_directions", 3),
            "sensor_layer": config.get("sensor_layer", -1),
            "steering_layers": config.get("steering_layers", ""),
            "gate_hidden_dim": config.get("gate_hidden_dim", 64),
            "intensity_hidden_dim": config.get("intensity_hidden_dim", 32),
            "behavior_weight": config.get("behavior_weight", 1.0),
            "retain_weight": config.get("retain_weight", 0.2),
            "sparse_weight": config.get("sparse_weight", 0.05),
            "max_alpha": config.get("max_alpha", 2.0),
            "optimization_steps": config.get("optimization_steps", 200),
            "learning_rate": config.get("learning_rate", 0.005),
        }
    
    # Create the method
    method_class = SteeringMethodRegistry.get(method_name).method_class
    return method_class(**params)


def get_optimal_steering_plan(
    model,
    task_name: str,
    train_pairs: "ContrastivePairSet",
    method: str = "*",
    aggregation: ExtractionStrategy = ExtractionStrategy.CHAT_LAST,
) -> Optional[Tuple["SteeringPlan", Dict[str, Any]]]:
    """
    Load optimal config and create a ready-to-use steering plan.
    
    This is the main convenience function for using optimized steering.
    It loads the best config for a task, trains a steering vector, and
    returns a SteeringPlan ready for generation.
    
    Args:
        model: WisentModel instance
        task_name: Name of the task to get optimal steering for
        train_pairs: ContrastivePairSet for training the steering vector
        method: Steering method to use ("*" for best, or specific method name)
        aggregation: Token aggregation strategy
        
    Returns:
        Tuple of (SteeringPlan, config_dict) or None if no config found
        
    Example:
        >>> from wisent.core.cli.steering_method_trainer import get_optimal_steering_plan
        >>> 
        >>> # Load optimal config and create plan
        >>> plan, config = get_optimal_steering_plan(model, "truthfulqa_mc1", train_pairs)
        >>> 
        >>> # Use for generation
        >>> output = model.generate(
        ...     [{"role": "user", "content": "Is the earth flat?"}],
        ...     use_steering=True,
        ...     steering_plan=plan,
        ... )
    """
    # Load optimal config
    config = load_optimal_steering_config(model.model_name, task_name, method)
    if config is None:
        return None
    
    # Get parameters
    layer = config["layer"]
    strength = config["strength"]
    method_name = config["method"]
    
    # Collect activations for the optimal layer
    collector = ActivationCollector(model=model)
    layer_str = str(layer)
    
    pos_acts = []
    neg_acts = []
    
    for pair in train_pairs.pairs:
        updated_pair = collector.collect_for_pair(
            pair,
            layers=[layer_str],
            aggregation=aggregation,
            return_full_sequence=False,
            normalize_layers=False,
        )
        
        if (updated_pair.positive_response.layers_activations and 
            layer_str in updated_pair.positive_response.layers_activations):
            act = updated_pair.positive_response.layers_activations[layer_str]
            if act is not None:
                pos_acts.append(act)
        
        if (updated_pair.negative_response.layers_activations and 
            layer_str in updated_pair.negative_response.layers_activations):
            act = updated_pair.negative_response.layers_activations[layer_str]
            if act is not None:
                neg_acts.append(act)
    
    if not pos_acts or not neg_acts:
        return None
    
    # Create steering method with optimal params
    steering_method = create_steering_method_from_config(config)
    
    # Train steering vector
    steering_vector = steering_method.train_for_layer(pos_acts, neg_acts)
    
    # Create steering plan
    plan = SteeringPlan.from_raw(
        {layer_str: steering_vector},
        scale=strength,
        normalize=True,
        description=f"Optimal {method_name} steering for {task_name}",
    )
    
    return plan, config
