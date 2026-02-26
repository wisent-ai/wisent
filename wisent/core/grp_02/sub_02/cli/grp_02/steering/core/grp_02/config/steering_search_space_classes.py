"""Search space classes for steering methods."""
from enum import Enum
from typing import List, Dict, Any, Optional
from wisent.core import constants as _C


class DirectionWeighting(str, Enum):
    """How to combine multiple directions in TECZA/GROM."""
    PRIMARY_ONLY = "primary_only"  # Use only the first/strongest direction
    EQUAL = "equal"  # Equal weight to all directions
    LEARNED = "learned"  # Use learned weights
    DECAY = "decay"  # Exponentially decaying weights (1, 0.5, 0.25, ...)


class SteeringLayerConfig(str, Enum):
    """Predefined steering layer configurations."""
    SINGLE_BEST = "single_best"  # Only the optimal single layer
    RANGE_3 = "range_3"  # 3 consecutive layers around best
    RANGE_5 = "range_5"  # 5 consecutive layers around best
    CUSTOM = "custom"  # User-specified layers


class SensorLayerConfig(str, Enum):
    """Predefined sensor layer positions."""
    MIDDLE = "middle"  # Middle of the network
    CUSTOM = "custom"  # User-specified


@dataclass
class BaseSearchSpace:
    """Base search space common to all methods."""
    
    # layers MUST be set by get_search_space() to all layers (0 to num_layers-1)
    # Empty default ensures it's always explicitly set
    layers: List[int] = field(default_factory=list)
    strengths: List[float] = field(default_factory=lambda: list(_C.AUTO_DEFAULT_STRENGTHS))
    strategies: List[str] = field(default_factory=lambda: list(_C.STEERING_STRATEGIES))
    token_aggregations: List[str] = field(default_factory=lambda: list(_C.TOKEN_AGGREGATIONS))
    prompt_constructions: List[str] = field(default_factory=lambda: list(_C.PROMPT_CONSTRUCTIONS))
    
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
class TECZASearchSpace(BaseSearchSpace):
    """Search space for TECZA method."""
    
    num_directions: List[int] = field(default_factory=lambda: list(_C.TECZA_SEARCH_NUM_DIRECTIONS))
    direction_weighting: List[str] = field(default_factory=lambda: list(_C.DIRECTION_WEIGHTING_OPTIONS))
    retain_weight: List[float] = field(default_factory=lambda: list(_C.TECZA_SEARCH_RETAIN_WEIGHTS))
    independence_weight: List[float] = field(default_factory=lambda: [_C.TECZA_INDEPENDENCE_WEIGHT])
    optimization_steps: List[int] = field(default_factory=lambda: list(_C.TECZA_SEARCH_OPT_STEPS))
    learning_rate: List[float] = field(default_factory=lambda: [_C.TECZA_LEARNING_RATE])
    use_caa_init: List[bool] = field(default_factory=lambda: [True])
    cone_constraint: List[bool] = field(default_factory=lambda: [True])
    min_cosine_similarity: List[float] = field(default_factory=lambda: [_C.CAA_MIN_COSINE_SIMILARITY])
    max_cosine_similarity: List[float] = field(default_factory=lambda: [_C.CAA_MAX_COSINE_SIMILARITY])
    
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
class TETNOSearchSpace(BaseSearchSpace):
    """Search space for TETNO method."""
    
    # Override base - TETNO uses different layer logic
    layers: List[int] = field(default_factory=lambda: [])  # Not used directly
    
    sensor_layer_config: List[str] = field(default_factory=lambda: list(_C.SENSOR_LAYER_CONFIGS))
    steering_layer_config: List[str] = field(default_factory=lambda: list(_C.TETNO_STEERING_LAYER_CONFIGS))
    condition_threshold: List[float] = field(default_factory=lambda: list(_C.TETNO_SEARCH_CONDITION_THRESHOLDS))
    gate_temperature: List[float] = field(default_factory=lambda: list(_C.TETNO_SEARCH_GATE_TEMPERATURES))
    per_layer_scaling: List[bool] = field(default_factory=lambda: [True, False])
    use_entropy_scaling: List[bool] = field(default_factory=lambda: [True, False])
    max_alpha: List[float] = field(default_factory=lambda: list(_C.TETNO_SEARCH_MAX_ALPHAS))
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
        elif isinstance(config, int):
            return config
        else:
            raise ValueError(
                f"Unknown sensor_layer config: {config}. "
                f"Use 'middle' or an explicit integer layer index."
            )

    def resolve_steering_layers(self, config: str, best_layer: int, num_layers: int) -> List[int]:
        """Convert steering layer config to actual layer indices."""
        if config == "single_best":
            return [best_layer]
        elif config == "range_3":
            return [max(0, best_layer - 1), best_layer, min(num_layers - 1, best_layer + 1)]
        elif config == "range_5":
            return list(range(max(0, best_layer - 2), min(num_layers, best_layer + 3)))
        else:
            return [best_layer]


@dataclass
class GROMSearchSpace(BaseSearchSpace):
    """Search space for GROM method."""

    # Override base - GROM uses different layer logic
    layers: List[int] = field(default_factory=lambda: [])  # Not used directly

    num_directions: List[int] = field(default_factory=lambda: list(_C.GROM_SEARCH_NUM_DIRECTIONS))
    sensor_layer_config: List[str] = field(default_factory=lambda: list(_C.SENSOR_LAYER_CONFIGS))
    steering_layer_config: List[str] = field(default_factory=lambda: list(_C.GROM_STEERING_LAYER_CONFIGS))
    gate_hidden_dim: List[int] = field(default_factory=lambda: list(_C.GROM_SEARCH_GATE_HIDDEN_DIMS))
    intensity_hidden_dim: List[int] = field(default_factory=lambda: list(_C.GROM_SEARCH_INTENSITY_HIDDEN_DIMS))
    behavior_weight: List[float] = field(default_factory=lambda: list(_C.GROM_SEARCH_BEHAVIOR_WEIGHTS))
    retain_weight: List[float] = field(default_factory=lambda: list(_C.GROM_SEARCH_RETAIN_WEIGHTS))
    sparse_weight: List[float] = field(default_factory=lambda: list(_C.GROM_SPARSE_WEIGHT_OPTIONS))
    max_alpha: List[float] = field(default_factory=lambda: list(_C.GROM_MAX_ALPHA_SEARCH))
    optimization_steps: List[int] = field(default_factory=lambda: list(_C.GROM_SEARCH_OPT_STEPS))
    learning_rate: List[float] = field(default_factory=lambda: [_C.GROM_LEARNING_RATE])
    
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
                "strategy": "constant",  # GROM handles this internally
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
        elif isinstance(config, int):
            return config
        else:
            raise ValueError(
                f"Unknown sensor_layer config: {config}. "
                f"Use 'middle' or an explicit integer layer index."
            )
    
    def resolve_steering_layers(self, config: str, best_layer: int, num_layers: int) -> List[int]:
        """Convert steering layer config to actual layer indices."""
        if config == "range_3":
            return [max(0, best_layer - 1), best_layer, min(num_layers - 1, best_layer + 1)]
        elif config == "range_5":
            return list(range(max(0, best_layer - 2), min(num_layers, best_layer + 3)))
        else:
            raise ValueError(
                f"Unknown steering_layer config: {config}. "
                f"Use 'range_3' or 'range_5'."
            )


