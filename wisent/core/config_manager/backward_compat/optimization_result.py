"""Backward-compatible OptimizationResult and related functions."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any

from ..convenience import save_steering_config, get_steering_config


@dataclass
class OptimizationResult:
    """Backward-compatible result class for steering optimization cache."""
    model: str
    task: str
    layer: int
    strength: float
    method: str = "CAA"
    token_aggregation: str = "average"
    prompt_strategy: str = "question_only"
    strategy: str = "constant"
    score: float = 0.0
    metric: str = "accuracy"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    # PRISM
    num_directions: int = 1
    direction_weighting: str = "primary_only"
    retain_weight: float = 0.0
    independence_weight: float = 0.05
    prism_optimization_steps: int = 100
    use_caa_init: bool = True
    cone_constraint: bool = True
    min_cosine_similarity: float = 0.3
    max_cosine_similarity: float = 0.95
    # PULSE
    sensor_layer: int = -1
    steering_layers: str = ""
    condition_threshold: float = 0.5
    gate_temperature: float = 0.5
    per_layer_scaling: bool = True
    use_entropy_scaling: bool = False
    max_alpha: float = 2.0
    learn_threshold: bool = True
    pulse_optimization_steps: int = 100
    # TITAN
    gate_hidden_dim: int = 64
    intensity_hidden_dim: int = 32
    behavior_weight: float = 1.0
    sparse_weight: float = 0.05
    titan_optimization_steps: int = 200
    titan_learning_rate: float = 0.005
    method_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def store_optimization(
    model: str, task: str, layer: int, strength: float,
    method: str = "CAA", token_aggregation: str = "average",
    prompt_strategy: str = "question_only", strategy: str = "constant",
    score: float = 0.0, metric: str = "accuracy",
    metadata: Optional[Dict[str, Any]] = None, set_as_default: bool = False,
    num_directions: int = 1, direction_weighting: str = "primary_only",
    retain_weight: float = 0.0, independence_weight: float = 0.05,
    prism_optimization_steps: int = 100, use_caa_init: bool = True,
    cone_constraint: bool = True, min_cosine_similarity: float = 0.3,
    max_cosine_similarity: float = 0.95, sensor_layer: int = -1,
    steering_layers: str = "", condition_threshold: float = 0.5,
    gate_temperature: float = 0.5, per_layer_scaling: bool = True,
    use_entropy_scaling: bool = False, max_alpha: float = 2.0,
    learn_threshold: bool = True, pulse_optimization_steps: int = 100,
    gate_hidden_dim: int = 64, intensity_hidden_dim: int = 32,
    behavior_weight: float = 1.0, sparse_weight: float = 0.05,
    titan_optimization_steps: int = 200, titan_learning_rate: float = 0.005,
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
        independence_weight=independence_weight, prism_optimization_steps=prism_optimization_steps,
        use_caa_init=use_caa_init, cone_constraint=cone_constraint,
        min_cosine_similarity=min_cosine_similarity, max_cosine_similarity=max_cosine_similarity,
        sensor_layer=sensor_layer, steering_layers=steering_layers,
        condition_threshold=condition_threshold, gate_temperature=gate_temperature,
        per_layer_scaling=per_layer_scaling, use_entropy_scaling=use_entropy_scaling,
        max_alpha=max_alpha, learn_threshold=learn_threshold,
        pulse_optimization_steps=pulse_optimization_steps, gate_hidden_dim=gate_hidden_dim,
        intensity_hidden_dim=intensity_hidden_dim, behavior_weight=behavior_weight,
        sparse_weight=sparse_weight, titan_optimization_steps=titan_optimization_steps,
        titan_learning_rate=titan_learning_rate, method_params=method_params,
    )
    model_normalized = model.replace("/", "_").replace("\\", "_")
    return f"{model_normalized}::{task}::{method}"


def get_cached_optimization(
    model: str, task: str, method: str = "CAA", use_default: bool = True
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
        prism_optimization_steps=steering.prism_optimization_steps, use_caa_init=steering.use_caa_init,
        cone_constraint=steering.cone_constraint, min_cosine_similarity=steering.min_cosine_similarity,
        max_cosine_similarity=steering.max_cosine_similarity, sensor_layer=steering.sensor_layer,
        steering_layers=steering.steering_layers, condition_threshold=steering.condition_threshold,
        gate_temperature=steering.gate_temperature, per_layer_scaling=steering.per_layer_scaling,
        use_entropy_scaling=steering.use_entropy_scaling, max_alpha=steering.max_alpha,
        learn_threshold=steering.learn_threshold, pulse_optimization_steps=steering.pulse_optimization_steps,
        gate_hidden_dim=steering.gate_hidden_dim, intensity_hidden_dim=steering.intensity_hidden_dim,
        behavior_weight=steering.behavior_weight, sparse_weight=steering.sparse_weight,
        titan_optimization_steps=steering.titan_optimization_steps, titan_learning_rate=steering.titan_learning_rate,
        method_params=steering.method_params,
    )
