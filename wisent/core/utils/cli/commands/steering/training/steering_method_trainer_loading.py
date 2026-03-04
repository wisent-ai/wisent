"""Steering config loading and plan creation."""
import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch

from wisent.core.primitives.models.core.atoms import SteeringPlan, SteeringVector
from wisent.core.utils.config_tools.constants import ARCHITECTURE_MODULE_LIMIT

logger = logging.getLogger(__name__)


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
        method: Steering method to load ("*" for any, or specific like "CAA", "TECZA")
        
    Returns:
        Dict with optimal steering parameters including method-specific params, or None if not found
        
    Example:
        >>> config = load_optimal_steering_config("Qwen/Qwen2-7B", "truthfulqa_mc1")
        >>> if config:
        ...     print(f"Best: {config['method']} at layer {config['layer']} with strength {config['strength']}")
    """
    from wisent.core.utils.config_tools.config import get_cached_optimization
    
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
    if result.method.upper() == "TECZA":
        config.update({
            "num_directions": result.num_directions,
            "direction_weighting": result.direction_weighting,
            "retain_weight": result.retain_weight,
            "independence_weight": result.independence_weight,
            "optimization_steps": result.tecza_optimization_steps,
            "use_caa_init": result.use_caa_init,
            "cone_constraint": result.cone_constraint,
            "min_cosine_similarity": result.min_cosine_similarity,
            "max_cosine_similarity": result.max_cosine_similarity,
        })
    elif result.method.upper() == "TETNO":
        config.update({
            "sensor_layer": result.sensor_layer,
            "steering_layers": result.steering_layers,
            "condition_threshold": result.condition_threshold,
            "gate_temperature": result.gate_temperature,
            "per_layer_scaling": result.per_layer_scaling,
            "use_entropy_scaling": result.use_entropy_scaling,
            "max_alpha": result.max_alpha,
            "learn_threshold": result.learn_threshold,
            "optimization_steps": result.tetno_optimization_steps,
        })
    elif result.method.upper() == "GROM":
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
            "optimization_steps": result.grom_optimization_steps,
            "learning_rate": result.grom_learning_rate,
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

    elif method_name == "tecza":
        params = {}
        for key in [
            "num_directions", "retain_weight", "independence_weight",
            "optimization_steps", "min_cosine_similarity", "max_cosine_similarity",
        ]:
            if key in config:
                params[key] = config[key]
        # Boolean flags with sensible defaults
        params["direction_weighting"] = config.get("direction_weighting", "primary_only")
        params["use_caa_init"] = config.get("use_caa_init", True)
        params["cone_constraint"] = config.get("cone_constraint", True)

    elif method_name == "tetno":
        params = {}
        for key in [
            "condition_threshold", "gate_temperature", "max_alpha",
            "optimization_steps", "learning_rate", "sensor_layer",
            "steering_layers",
        ]:
            if key in config:
                params[key] = config[key]
        # Boolean flags with sensible defaults
        params["per_layer_scaling"] = config.get("per_layer_scaling", True)
        params["use_entropy_scaling"] = config.get("use_entropy_scaling", False)
        params["learn_threshold"] = config.get("learn_threshold", True)

    elif method_name == "grom":
        params = {}
        for key in [
            "num_directions", "optimization_steps", "learning_rate",
            "warmup_steps", "behavior_weight", "retain_weight",
            "sparse_weight", "smooth_weight", "independence_weight",
            "max_alpha", "gate_temperature", "min_cosine_similarity",
            "max_cosine_similarity", "sensor_layer", "steering_layers",
            "gate_hidden_dim", "intensity_hidden_dim",
        ]:
            if key in config:
                params[key] = config[key]
    
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
        >>> from wisent.core.utils.cli.steering_method_trainer import get_optimal_steering_plan
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
    collector = ActivationCollector(model=model, architecture_module_limit=ARCHITECTURE_MODULE_LIMIT)
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
