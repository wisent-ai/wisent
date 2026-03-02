"""Backward-compatible OptimizationResult and related functions."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any

from ..convenience import save_steering_config, get_steering_config

from wisent.core.utils.config_tools.constants import (
    DEFAULT_SCORE, TECZA_INDEPENDENCE_WEIGHT, DEFAULT_OPTIMIZATION_STEPS,
    TECZA_MIN_COSINE_SIM, TECZA_MAX_COSINE_SIM, TETNO_CONDITION_THRESHOLD,
    TETNO_MAX_ALPHA, DEFAULT_OPTIMIZATION_STEPS, TETNO_GATE_TEMPERATURE_LEGACY,
    GROM_ROUTER_HIDDEN_DIM, GROM_INTENSITY_HIDDEN_DIM, GROM_BEHAVIOR_WEIGHT,
    GROM_SPARSE_WEIGHT, GROM_OPTIMIZATION_STEPS, GROM_LEARNING_RATE,
)

@dataclass
class OptimizationResult:
    """Backward-compatible result class for steering optimization cache."""
    model: str
    task: str
    layer: int
    strength: float
    method: Optional[str] = None
    token_aggregation: Optional[str] = None
    prompt_strategy: Optional[str] = None
    strategy: Optional[str] = None
    score: float = DEFAULT_SCORE
    metric: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    # TECZA
    num_directions: int = 1
    direction_weighting: Optional[str] = None
    retain_weight: float = 0.0
    independence_weight: float = TECZA_INDEPENDENCE_WEIGHT
    tecza_optimization_steps: int = DEFAULT_OPTIMIZATION_STEPS
    use_caa_init: bool = True
    cone_constraint: bool = True
    min_cosine_similarity: float = TECZA_MIN_COSINE_SIM
    max_cosine_similarity: float = TECZA_MAX_COSINE_SIM
    # TETNO
    sensor_layer: int = -1
    steering_layers: str = ""
    condition_threshold: float = TETNO_CONDITION_THRESHOLD
    gate_temperature: float = TETNO_GATE_TEMPERATURE_LEGACY
    per_layer_scaling: bool = True
    use_entropy_scaling: bool = False
    max_alpha: float = TETNO_MAX_ALPHA
    learn_threshold: bool = True
    tetno_optimization_steps: int = DEFAULT_OPTIMIZATION_STEPS
    # GROM
    gate_hidden_dim: int = GROM_ROUTER_HIDDEN_DIM
    intensity_hidden_dim: int = GROM_INTENSITY_HIDDEN_DIM
    behavior_weight: float = GROM_BEHAVIOR_WEIGHT
    sparse_weight: float = GROM_SPARSE_WEIGHT
    grom_optimization_steps: int = GROM_OPTIMIZATION_STEPS
    grom_learning_rate: float = GROM_LEARNING_RATE
    method_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def store_optimization(
    model: str, task: str, layer: int, strength: float,
    method: str, token_aggregation: str,
    prompt_strategy: str, strategy: str,
    metric: str, direction_weighting: str,
    score: float = DEFAULT_SCORE,
    metadata: Optional[Dict[str, Any]] = None, set_as_default: bool = False,
    retain_weight: float = 0.0, independence_weight: float = TECZA_INDEPENDENCE_WEIGHT,
    tecza_optimization_steps: int = DEFAULT_OPTIMIZATION_STEPS, use_caa_init: bool = True,
    cone_constraint: bool = True, min_cosine_similarity: float = TECZA_MIN_COSINE_SIM,
    max_cosine_similarity: float = TECZA_MAX_COSINE_SIM, sensor_layer: int = -1,
    steering_layers: str = "", condition_threshold: float = TETNO_CONDITION_THRESHOLD,
    gate_temperature: float = TETNO_GATE_TEMPERATURE_LEGACY, per_layer_scaling: bool = True,
    use_entropy_scaling: bool = False, max_alpha: float = TETNO_MAX_ALPHA,
    learn_threshold: bool = True, tetno_optimization_steps: int = DEFAULT_OPTIMIZATION_STEPS,
    gate_hidden_dim: int = GROM_ROUTER_HIDDEN_DIM, intensity_hidden_dim: int = GROM_INTENSITY_HIDDEN_DIM,
    behavior_weight: float = GROM_BEHAVIOR_WEIGHT, sparse_weight: float = GROM_SPARSE_WEIGHT,
    grom_optimization_steps: int = GROM_OPTIMIZATION_STEPS, grom_learning_rate: float = GROM_LEARNING_RATE,
    method_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Backward-compatible function to store steering optimization result."""
    save_steering_config(
        model_name=model, task_name=task, layer=layer, strength=strength,
        method=method, token_aggregation=token_aggregation,
        prompt_strategy=prompt_strategy, strategy=strategy, score=score,
        metric=metric, optimization_method="optuna" if metadata else "manual",
        set_as_default=set_as_default, num_directions=num_directions,
        direction_weighting=direction_weighting, retain_weight=retain_weight,
        independence_weight=independence_weight, tecza_optimization_steps=tecza_optimization_steps,
        use_caa_init=use_caa_init, cone_constraint=cone_constraint,
        min_cosine_similarity=min_cosine_similarity, max_cosine_similarity=max_cosine_similarity,
        sensor_layer=sensor_layer, steering_layers=steering_layers,
        condition_threshold=condition_threshold, gate_temperature=gate_temperature,
        per_layer_scaling=per_layer_scaling, use_entropy_scaling=use_entropy_scaling,
        max_alpha=max_alpha, learn_threshold=learn_threshold,
        tetno_optimization_steps=tetno_optimization_steps, gate_hidden_dim=gate_hidden_dim,
        intensity_hidden_dim=intensity_hidden_dim, behavior_weight=behavior_weight,
        sparse_weight=sparse_weight, grom_optimization_steps=grom_optimization_steps,
        grom_learning_rate=grom_learning_rate, method_params=method_params,
    )
    model_normalized = model.replace("/", "_").replace("\\", "_")
    return f"{model_normalized}::{task}::{method}"


def get_cached_optimization(
    model: str, task: str, method: str, use_default: bool = True
) -> Optional[OptimizationResult]:
    """Backward-compatible function to get cached steering optimization result."""
    steering = get_steering_config(model, task)
    if steering is None:
        return None
    if method != "*" and steering.method != method:
        return None
    return OptimizationResult(
        model=model, task=task, layer=steering.layer, strength=steering.strength,
        method=steering.method, token_aggregation=steering.token_aggregation,
        prompt_strategy=steering.prompt_strategy, strategy=steering.strategy,
        score=steering.score, metric=steering.metric,
        num_directions=steering.num_directions, direction_weighting=steering.direction_weighting,
        retain_weight=steering.retain_weight, independence_weight=steering.independence_weight,
        tecza_optimization_steps=steering.tecza_optimization_steps, use_caa_init=steering.use_caa_init,
        cone_constraint=steering.cone_constraint, min_cosine_similarity=steering.min_cosine_similarity,
        max_cosine_similarity=steering.max_cosine_similarity, sensor_layer=steering.sensor_layer,
        steering_layers=steering.steering_layers, condition_threshold=steering.condition_threshold,
        gate_temperature=steering.gate_temperature, per_layer_scaling=steering.per_layer_scaling,
        use_entropy_scaling=steering.use_entropy_scaling, max_alpha=steering.max_alpha,
        learn_threshold=steering.learn_threshold, tetno_optimization_steps=steering.tetno_optimization_steps,
        gate_hidden_dim=steering.gate_hidden_dim, intensity_hidden_dim=steering.intensity_hidden_dim,
        behavior_weight=steering.behavior_weight, sparse_weight=steering.sparse_weight,
        grom_optimization_steps=steering.grom_optimization_steps, grom_learning_rate=steering.grom_learning_rate,
        method_params=steering.method_params,
    )
