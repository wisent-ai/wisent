"""
Search space definitions for steering method optimization.

Defines the full parameter search space for each steering method:
- CAA: Basic search (layer, strength, strategy, token_aggregation)
- PRISM: + num_directions, direction_weighting, retain_weight
- PULSE: + sensor_layer, steering_layers, threshold, gate_temperature, per_layer_scaling
- TITAN: + num_directions, network dimensions, loss weights
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Iterator
from enum import Enum
import itertools


class DirectionWeighting(str, Enum):
    """How to combine multiple directions in PRISM/TITAN."""
    PRIMARY_ONLY = "primary_only"  # Use only the first/strongest direction
    EQUAL = "equal"  # Equal weight to all directions
    LEARNED = "learned"  # Use learned weights
    DECAY = "decay"  # Exponentially decaying weights (1, 0.5, 0.25, ...)


class SteeringLayerConfig(str, Enum):
    """Predefined steering layer configurations."""
    SINGLE_BEST = "single_best"  # Only the optimal single layer
    RANGE_3 = "range_3"  # 3 consecutive layers around best
    RANGE_5 = "range_5"  # 5 consecutive layers around best
    ALL_LATE = "all_late"  # All layers in last quarter
    CUSTOM = "custom"  # User-specified layers


class SensorLayerConfig(str, Enum):
    """Predefined sensor layer positions."""
    MIDDLE = "middle"  # Middle of the network
    LATE = "late"  # 75% through the network
    LAST_QUARTER = "last_quarter"  # Start of last quarter
    CUSTOM = "custom"  # User-specified


@dataclass
class BaseSearchSpace:
    """Base search space common to all methods."""
    
    # layers MUST be set by get_search_space() to all layers (0 to num_layers-1)
    # Empty default ensures it's always explicitly set
    layers: List[int] = field(default_factory=list)
    strengths: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    strategies: List[str] = field(default_factory=lambda: ["constant", "initial_only", "diminishing", "increasing", "gaussian"])
    token_aggregations: List[str] = field(default_factory=lambda: ["last_token", "mean_pooling", "first_token", "max_pooling", "continuation_token"])
    prompt_constructions: List[str] = field(default_factory=lambda: ["chat_template", "direct_completion", "multiple_choice", "role_playing", "instruction_following"])
    
    def get_total_configs(self) -> int:
        return (
            len(self.layers) *
            len(self.strengths) *
            len(self.strategies) *
            len(self.token_aggregations) *
            len(self.prompt_constructions)
        )
    
    def iterate(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all base configurations."""
        for layer, strength, strategy, token_agg, prompt_const in itertools.product(
            self.layers, self.strengths, self.strategies,
            self.token_aggregations, self.prompt_constructions
        ):
            yield {
                "layer": layer,
                "strength": strength,
                "strategy": strategy,
                "token_aggregation": token_agg,
                "prompt_construction": prompt_const,
            }


@dataclass
class CAASearchSpace(BaseSearchSpace):
    """Search space for CAA method."""
    
    normalize: List[bool] = field(default_factory=lambda: [True])
    
    def get_total_configs(self) -> int:
        return super().get_total_configs() * len(self.normalize)
    
    def iterate(self) -> Iterator[Dict[str, Any]]:
        for base_config in super().iterate():
            for normalize in self.normalize:
                yield {
                    **base_config,
                    "normalize": normalize,
                }


@dataclass
class PRISMSearchSpace(BaseSearchSpace):
    """Search space for PRISM method."""
    
    num_directions: List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    direction_weighting: List[str] = field(default_factory=lambda: ["primary_only", "equal", "learned"])
    retain_weight: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3])
    independence_weight: List[float] = field(default_factory=lambda: [0.05])
    optimization_steps: List[int] = field(default_factory=lambda: [50, 100])
    learning_rate: List[float] = field(default_factory=lambda: [0.01])
    use_caa_init: List[bool] = field(default_factory=lambda: [True])
    cone_constraint: List[bool] = field(default_factory=lambda: [True])
    min_cosine_similarity: List[float] = field(default_factory=lambda: [0.3])
    max_cosine_similarity: List[float] = field(default_factory=lambda: [0.95])
    
    def get_total_configs(self) -> int:
        return (
            super().get_total_configs() *
            len(self.num_directions) *
            len(self.direction_weighting) *
            len(self.retain_weight) *
            len(self.optimization_steps)
        )
    
    def iterate(self) -> Iterator[Dict[str, Any]]:
        for base_config in super().iterate():
            for num_dirs, dir_weight, retain_w, opt_steps in itertools.product(
                self.num_directions, self.direction_weighting,
                self.retain_weight, self.optimization_steps
            ):
                yield {
                    **base_config,
                    "num_directions": num_dirs,
                    "direction_weighting": dir_weight,
                    "retain_weight": retain_w,
                    "optimization_steps": opt_steps,
                    "independence_weight": self.independence_weight[0],
                    "learning_rate": self.learning_rate[0],
                    "use_caa_init": self.use_caa_init[0],
                    "cone_constraint": self.cone_constraint[0],
                    "min_cosine_similarity": self.min_cosine_similarity[0],
                    "max_cosine_similarity": self.max_cosine_similarity[0],
                }


@dataclass
class PULSESearchSpace(BaseSearchSpace):
    """Search space for PULSE method."""
    
    # Override base - PULSE uses different layer logic
    layers: List[int] = field(default_factory=lambda: [])  # Not used directly
    
    sensor_layer_config: List[str] = field(default_factory=lambda: ["middle", "late", "last_quarter"])
    steering_layer_config: List[str] = field(default_factory=lambda: ["single_best", "range_3", "range_5"])
    condition_threshold: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    gate_temperature: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])
    per_layer_scaling: List[bool] = field(default_factory=lambda: [True, False])
    use_entropy_scaling: List[bool] = field(default_factory=lambda: [True, False])
    max_alpha: List[float] = field(default_factory=lambda: [1.5, 2.0, 3.0])
    learn_threshold: List[bool] = field(default_factory=lambda: [True])
    optimization_steps: List[int] = field(default_factory=lambda: [50, 100])
    
    def get_total_configs(self) -> int:
        return (
            len(self.strengths) *
            len(self.strategies) *
            len(self.token_aggregations) *
            len(self.prompt_constructions) *
            len(self.sensor_layer_config) *
            len(self.steering_layer_config) *
            len(self.condition_threshold) *
            len(self.gate_temperature) *
            len(self.per_layer_scaling) *
            len(self.use_entropy_scaling) *
            len(self.max_alpha)
        )
    
    def iterate(self) -> Iterator[Dict[str, Any]]:
        for (strength, strategy, token_agg, prompt_const,
             sensor_cfg, steering_cfg, threshold, temp,
             per_layer, entropy, max_a) in itertools.product(
            self.strengths, self.strategies, self.token_aggregations,
            self.prompt_constructions, self.sensor_layer_config,
            self.steering_layer_config, self.condition_threshold,
            self.gate_temperature, self.per_layer_scaling,
            self.use_entropy_scaling, self.max_alpha
        ):
            yield {
                "strength": strength,
                "strategy": strategy,
                "token_aggregation": token_agg,
                "prompt_construction": prompt_const,
                "sensor_layer_config": sensor_cfg,
                "steering_layer_config": steering_cfg,
                "condition_threshold": threshold,
                "gate_temperature": temp,
                "per_layer_scaling": per_layer,
                "use_entropy_scaling": entropy,
                "max_alpha": max_a,
                "learn_threshold": self.learn_threshold[0],
                "optimization_steps": self.optimization_steps[0],
            }
    
    def resolve_sensor_layer(self, config: str, num_layers: int) -> int:
        """Convert sensor layer config to actual layer index."""
        if config == "middle":
            return num_layers // 2
        elif config == "late":
            return int(num_layers * 0.75)
        elif config == "last_quarter":
            return int(num_layers * 0.75)
        else:
            return num_layers - 4  # Default
    
    def resolve_steering_layers(self, config: str, best_layer: int, num_layers: int) -> List[int]:
        """Convert steering layer config to actual layer indices."""
        if config == "single_best":
            return [best_layer]
        elif config == "range_3":
            return [max(0, best_layer - 1), best_layer, min(num_layers - 1, best_layer + 1)]
        elif config == "range_5":
            return list(range(max(0, best_layer - 2), min(num_layers, best_layer + 3)))
        elif config == "all_late":
            start = int(num_layers * 0.75)
            return list(range(start, num_layers - 1))
        else:
            return [best_layer]


@dataclass
class TITANSearchSpace(BaseSearchSpace):
    """Search space for TITAN method."""
    
    # Override base - TITAN uses different layer logic
    layers: List[int] = field(default_factory=lambda: [])  # Not used directly
    
    num_directions: List[int] = field(default_factory=lambda: [2, 3, 5])
    sensor_layer_config: List[str] = field(default_factory=lambda: ["middle", "late"])
    steering_layer_config: List[str] = field(default_factory=lambda: ["range_3", "range_5", "all_late"])
    gate_hidden_dim: List[int] = field(default_factory=lambda: [32, 64, 128])
    intensity_hidden_dim: List[int] = field(default_factory=lambda: [16, 32, 64])
    behavior_weight: List[float] = field(default_factory=lambda: [0.5, 1.0])
    retain_weight: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.5])
    sparse_weight: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1])
    max_alpha: List[float] = field(default_factory=lambda: [2.0, 3.0, 5.0])
    optimization_steps: List[int] = field(default_factory=lambda: [100, 200])
    learning_rate: List[float] = field(default_factory=lambda: [0.005])
    
    def get_total_configs(self) -> int:
        return (
            len(self.strengths) *
            len(self.token_aggregations) *
            len(self.num_directions) *
            len(self.sensor_layer_config) *
            len(self.steering_layer_config) *
            len(self.gate_hidden_dim) *
            len(self.intensity_hidden_dim) *
            len(self.behavior_weight) *
            len(self.retain_weight) *
            len(self.sparse_weight) *
            len(self.max_alpha)
        )
    
    def iterate(self) -> Iterator[Dict[str, Any]]:
        for (strength, token_agg, num_dirs, sensor_cfg, steering_cfg,
             gate_dim, intensity_dim, behavior_w, retain_w,
             sparse_w, max_a) in itertools.product(
            self.strengths, self.token_aggregations,
            self.num_directions, self.sensor_layer_config,
            self.steering_layer_config, self.gate_hidden_dim,
            self.intensity_hidden_dim, self.behavior_weight,
            self.retain_weight, self.sparse_weight, self.max_alpha
        ):
            yield {
                "strength": strength,
                "token_aggregation": token_agg,
                "strategy": "constant",  # TITAN handles this internally
                "prompt_construction": "chat_template",
                "num_directions": num_dirs,
                "sensor_layer_config": sensor_cfg,
                "steering_layer_config": steering_cfg,
                "gate_hidden_dim": gate_dim,
                "intensity_hidden_dim": intensity_dim,
                "behavior_weight": behavior_w,
                "retain_weight": retain_w,
                "sparse_weight": sparse_w,
                "max_alpha": max_a,
                "optimization_steps": self.optimization_steps[0],
                "learning_rate": self.learning_rate[0],
            }
    
    def resolve_sensor_layer(self, config: str, num_layers: int) -> int:
        """Convert sensor layer config to actual layer index."""
        if config == "middle":
            return num_layers // 2
        elif config == "late":
            return int(num_layers * 0.75)
        else:
            return num_layers - 4
    
    def resolve_steering_layers(self, config: str, best_layer: int, num_layers: int) -> List[int]:
        """Convert steering layer config to actual layer indices."""
        if config == "range_3":
            return [max(0, best_layer - 1), best_layer, min(num_layers - 1, best_layer + 1)]
        elif config == "range_5":
            return list(range(max(0, best_layer - 2), min(num_layers, best_layer + 3)))
        elif config == "all_late":
            start = int(num_layers * 0.75)
            return list(range(start, num_layers - 1))
        else:
            return [best_layer]


def get_search_space(method_name: str, num_layers: int, quick: bool = False) -> BaseSearchSpace:
    """
    Get the search space for a given method.
    
    Args:
        method_name: Name of steering method (CAA, PRISM, PULSE, TITAN)
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
        elif method == "PRISM":
            return PRISMSearchSpace(
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
        elif method == "PULSE":
            return PULSESearchSpace(
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
        elif method == "TITAN":
            return TITANSearchSpace(
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
    elif method == "PRISM":
        return PRISMSearchSpace(layers=all_layers)
    elif method == "PULSE":
        return PULSESearchSpace(strengths=[0.5, 1.0, 1.5, 2.0])
    elif method == "TITAN":
        return TITANSearchSpace(strengths=[0.5, 1.0, 1.5, 2.0])
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
    if method == "PRISM" and isinstance(search_space, PRISMSearchSpace):
        if hasattr(args, 'search_num_directions') and args.search_num_directions:
            search_space.num_directions = args.search_num_directions
        if hasattr(args, 'search_direction_weighting') and args.search_direction_weighting:
            search_space.direction_weighting = args.search_direction_weighting
        if hasattr(args, 'search_retain_weight') and args.search_retain_weight:
            search_space.retain_weight = args.search_retain_weight
    
    elif method == "PULSE" and isinstance(search_space, PULSESearchSpace):
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
    
    elif method == "TITAN" and isinstance(search_space, TITANSearchSpace):
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
        
    elif isinstance(search_space, PRISMSearchSpace):
        print(f"   Layers: {search_space.layers}")
        print(f"   Strengths: {search_space.strengths}")
        print(f"   Num directions: {search_space.num_directions}")
        print(f"   Direction weighting: {search_space.direction_weighting}")
        print(f"   Retain weights: {search_space.retain_weight}")
        print(f"   Optimization steps: {search_space.optimization_steps}")
        
    elif isinstance(search_space, PULSESearchSpace):
        print(f"   Strengths: {search_space.strengths}")
        print(f"   Sensor layer configs: {search_space.sensor_layer_config}")
        print(f"   Steering layer configs: {search_space.steering_layer_config}")
        print(f"   Condition thresholds: {search_space.condition_threshold}")
        print(f"   Gate temperatures: {search_space.gate_temperature}")
        print(f"   Per-layer scaling: {search_space.per_layer_scaling}")
        print(f"   Entropy scaling: {search_space.use_entropy_scaling}")
        print(f"   Max alpha: {search_space.max_alpha}")
        
    elif isinstance(search_space, TITANSearchSpace):
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
