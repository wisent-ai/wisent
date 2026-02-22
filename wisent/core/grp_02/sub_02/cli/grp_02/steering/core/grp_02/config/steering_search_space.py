"""
Search space definitions for steering method optimization.

Defines the full parameter search space for each steering method:
- CAA: Basic search (layer, strength, strategy, token_aggregation)
- TECZA: + num_directions, direction_weighting, retain_weight
- TETNO: + sensor_layer, steering_layers, threshold, gate_temperature, per_layer_scaling
- GROM: + num_directions, network dimensions, loss weights
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Iterator
from enum import Enum
import itertools



from wisent.core.cli.steering.core.config.steering_search_space_classes import (
    DirectionWeighting, SteeringLayerConfig, SensorLayerConfig,
    BaseSearchSpace, CAASearchSpace, TECZASearchSpace,
    TETNOSearchSpace, GROMSearchSpace,
)


def get_search_space(method_name: str, num_layers: int, quick: bool = False) -> BaseSearchSpace:
    """
    Get the search space for a given method.
    
    Args:
        method_name: Name of steering method (CAA, TECZA, TETNO, GROM)
        num_layers: Number of layers in the model
        quick: If True, use reduced search space for faster testing
        
    Returns:
        Search space instance for the method
    """
    method = method_name.upper()
    
    # Full search uses ALL layers
    all_layers = list(range(num_layers))
    
    # Quick search uses subset of layers
    if num_layers > 20:
        quick_layers = list(range(num_layers // 2, num_layers - 2, 2))
    elif num_layers > 12:
        quick_layers = [4, 6, 8, 10, 12]
    else:
        quick_layers = list(range(2, num_layers, 2))
    
    if quick:
        # Reduced search space for quick testing
        if method == "CAA":
            return CAASearchSpace(
                layers=quick_layers[:3],
                strengths=[0.5, 1.0, 1.5],
                strategies=["constant"],
                token_aggregations=["last_token"],
                prompt_constructions=["chat_template"],
            )
        elif method == "TECZA":
            return TECZASearchSpace(
                layers=quick_layers[:3],
                strengths=[0.5, 1.0, 1.5],
                strategies=["constant"],
                token_aggregations=["last_token"],
                prompt_constructions=["chat_template"],
                num_directions=[2, 3],
                direction_weighting=["primary_only", "equal"],
                retain_weight=[0.1],
                optimization_steps=[50],
            )
        elif method == "TETNO":
            return TETNOSearchSpace(
                strengths=[1.0, 1.5],
                strategies=["constant"],
                token_aggregations=["last_token"],
                prompt_constructions=["chat_template"],
                sensor_layer_config=["late"],
                steering_layer_config=["range_3"],
                condition_threshold=[0.5],
                gate_temperature=[0.5],
                per_layer_scaling=[True],
                use_entropy_scaling=[False],
                max_alpha=[2.0],
            )
        elif method == "GROM":
            return GROMSearchSpace(
                strengths=[1.0, 1.5],
                token_aggregations=["last_token"],
                num_directions=[3],
                sensor_layer_config=["late"],
                steering_layer_config=["range_3"],
                gate_hidden_dim=[64],
                intensity_hidden_dim=[32],
                behavior_weight=[1.0],
                retain_weight=[0.2],
                sparse_weight=[0.05],
                max_alpha=[2.0],
                optimization_steps=[100],
            )
    
    # Full search space - uses ALL layers
    if method == "CAA":
        return CAASearchSpace(layers=all_layers)
    elif method == "TECZA":
        return TECZASearchSpace(layers=all_layers)
    elif method == "TETNO":
        return TETNOSearchSpace(strengths=[0.5, 1.0, 1.5, 2.0])
    elif method == "GROM":
        return GROMSearchSpace(strengths=[0.5, 1.0, 1.5, 2.0])
    else:
        # Default to CAA search space
        return CAASearchSpace(layers=all_layers)


def get_search_space_from_args(method_name: str, args, num_layers: int) -> BaseSearchSpace:
    """
    Create a search space with values from CLI arguments.
    
    Args:
        method_name: Name of steering method
        args: Parsed CLI arguments
        num_layers: Number of layers in the model
        
    Returns:
        Search space configured from arguments
    """
    method = method_name.upper()
    quick = getattr(args, 'quick_search', False)
    
    # Get base search space
    search_space = get_search_space(method, num_layers, quick=quick)
    
    # Override with any explicit CLI arguments
    if hasattr(args, 'search_layers') and args.search_layers:
        if isinstance(args.search_layers, str):
            search_space.layers = [int(x) for x in args.search_layers.split(',')]
        else:
            search_space.layers = args.search_layers
    
    if hasattr(args, 'search_strengths') and args.search_strengths:
        if isinstance(args.search_strengths, str):
            search_space.strengths = [float(x) for x in args.search_strengths.split(',')]
        else:
            search_space.strengths = args.search_strengths
    
    if hasattr(args, 'search_strategies') and args.search_strategies:
        search_space.strategies = args.search_strategies
    
    # Method-specific overrides
    if method == "TECZA" and isinstance(search_space, TECZASearchSpace):
        if hasattr(args, 'search_num_directions') and args.search_num_directions:
            search_space.num_directions = args.search_num_directions
        if hasattr(args, 'search_direction_weighting') and args.search_direction_weighting:
            search_space.direction_weighting = args.search_direction_weighting
        if hasattr(args, 'search_retain_weight') and args.search_retain_weight:
            search_space.retain_weight = args.search_retain_weight
    
    elif method == "TETNO" and isinstance(search_space, TETNOSearchSpace):
        if hasattr(args, 'search_sensor_layer') and args.search_sensor_layer:
            search_space.sensor_layer_config = args.search_sensor_layer
        if hasattr(args, 'search_steering_layers') and args.search_steering_layers:
            search_space.steering_layer_config = args.search_steering_layers
        if hasattr(args, 'search_threshold') and args.search_threshold:
            search_space.condition_threshold = args.search_threshold
        if hasattr(args, 'search_gate_temp') and args.search_gate_temp:
            search_space.gate_temperature = args.search_gate_temp
        if hasattr(args, 'search_max_alpha') and args.search_max_alpha:
            search_space.max_alpha = args.search_max_alpha
    
    elif method == "GROM" and isinstance(search_space, GROMSearchSpace):
        if hasattr(args, 'search_num_directions') and args.search_num_directions:
            search_space.num_directions = args.search_num_directions
        if hasattr(args, 'search_gate_hidden') and args.search_gate_hidden:
            search_space.gate_hidden_dim = args.search_gate_hidden
        if hasattr(args, 'search_intensity_hidden') and args.search_intensity_hidden:
            search_space.intensity_hidden_dim = args.search_intensity_hidden
        if hasattr(args, 'search_behavior_weight') and args.search_behavior_weight:
            search_space.behavior_weight = args.search_behavior_weight
        if hasattr(args, 'search_retain_weight') and args.search_retain_weight:
            search_space.retain_weight = args.search_retain_weight
        if hasattr(args, 'search_sparse_weight') and args.search_sparse_weight:
            search_space.sparse_weight = args.search_sparse_weight
    
    return search_space


def print_search_space_summary(search_space: BaseSearchSpace, method_name: str):
    """Print a summary of the search space."""
    print(f"\n📊 Search Space for {method_name.upper()}:")
    print(f"   Total configurations: {search_space.get_total_configs():,}")
    
    if isinstance(search_space, CAASearchSpace):
        print(f"   Layers: {search_space.layers}")
        print(f"   Strengths: {search_space.strengths}")
        print(f"   Strategies: {search_space.strategies}")
        print(f"   Token aggregations: {search_space.token_aggregations}")
        
    elif isinstance(search_space, TECZASearchSpace):
        print(f"   Layers: {search_space.layers}")
        print(f"   Strengths: {search_space.strengths}")
        print(f"   Num directions: {search_space.num_directions}")
        print(f"   Direction weighting: {search_space.direction_weighting}")
        print(f"   Retain weights: {search_space.retain_weight}")
        print(f"   Optimization steps: {search_space.optimization_steps}")
        
    elif isinstance(search_space, TETNOSearchSpace):
        print(f"   Strengths: {search_space.strengths}")
        print(f"   Sensor layer configs: {search_space.sensor_layer_config}")
        print(f"   Steering layer configs: {search_space.steering_layer_config}")
        print(f"   Condition thresholds: {search_space.condition_threshold}")
        print(f"   Gate temperatures: {search_space.gate_temperature}")
        print(f"   Per-layer scaling: {search_space.per_layer_scaling}")
        print(f"   Entropy scaling: {search_space.use_entropy_scaling}")
        print(f"   Max alpha: {search_space.max_alpha}")
        
    elif isinstance(search_space, GROMSearchSpace):
        print(f"   Strengths: {search_space.strengths}")
        print(f"   Num directions: {search_space.num_directions}")
        print(f"   Sensor layer configs: {search_space.sensor_layer_config}")
        print(f"   Steering layer configs: {search_space.steering_layer_config}")
        print(f"   Gate hidden dims: {search_space.gate_hidden_dim}")
        print(f"   Intensity hidden dims: {search_space.intensity_hidden_dim}")
        print(f"   Behavior weights: {search_space.behavior_weight}")
        print(f"   Retain weights: {search_space.retain_weight}")
        print(f"   Sparse weights: {search_space.sparse_weight}")
        print(f"   Max alpha: {search_space.max_alpha}")
