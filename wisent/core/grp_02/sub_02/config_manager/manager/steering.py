"""Steering and weight modification configuration mixin for WisentConfigManager."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from wisent.core.constants import (
    BLEND_DEFAULT,
    BROYDEN_DEFAULT_ALPHA,
    CONFIG_MAX_ALPHA,
    DEFAULT_LAYER_CONFIG,
    DEFAULT_SCORE,
    DEFAULT_STRENGTH,
    GROM_BEHAVIOR_WEIGHT,
    GROM_GATE_TEMPERATURE,
    GROM_INTENSITY_HIDDEN_DIM,
    GROM_LEARNING_RATE,
    GROM_OPTIMIZATION_STEPS,
    GROM_ROUTER_HIDDEN_DIM,
    GROM_SPARSE_WEIGHT,
    PAIRS_NUM_PAIRS,
    TECZA_INDEPENDENCE_WEIGHT,
    TECZA_MAX_COSINE_SIM,
    TECZA_MIN_COSINE_SIM,
    DEFAULT_OPTIMIZATION_STEPS,
    TETNO_CONDITION_THRESHOLD,
    DEFAULT_OPTIMIZATION_STEPS,
)
from ..types import SteeringConfig, WeightModificationConfig, TaskConfig


class SteeringMixin:
    """Mixin providing steering and weight modification config save/get methods."""

    def save_steering_config(
        self, model_name: str, task_name: Optional[str] = None,
        layer: int = DEFAULT_LAYER_CONFIG, strength: float = DEFAULT_STRENGTH,
        method: str = "CAA",
        token_aggregation: str = "average", prompt_strategy: str = "question_only",
        normalize_mode: str = "none", strategy: str = "constant",
        score: float = DEFAULT_SCORE, metric: str = "accuracy",
        optimization_method: str = "manual", set_as_default: bool = False,
        num_directions: int = 1, direction_weighting: str = "primary_only",
        retain_weight: float = 0.0,
        independence_weight: float = TECZA_INDEPENDENCE_WEIGHT,
        tecza_optimization_steps: int = DEFAULT_OPTIMIZATION_STEPS,
        use_caa_init: bool = True,
        cone_constraint: bool = True,
        min_cosine_similarity: float = TECZA_MIN_COSINE_SIM,
        max_cosine_similarity: float = TECZA_MAX_COSINE_SIM,
        sensor_layer: int = -1,
        steering_layers: str = "",
        condition_threshold: float = TETNO_CONDITION_THRESHOLD,
        gate_temperature: float = GROM_GATE_TEMPERATURE,
        per_layer_scaling: bool = True,
        use_entropy_scaling: bool = False,
        max_alpha: float = CONFIG_MAX_ALPHA,
        learn_threshold: bool = True,
        tetno_optimization_steps: int = DEFAULT_OPTIMIZATION_STEPS,
        gate_hidden_dim: int = GROM_ROUTER_HIDDEN_DIM,
        intensity_hidden_dim: int = GROM_INTENSITY_HIDDEN_DIM,
        behavior_weight: float = GROM_BEHAVIOR_WEIGHT,
        sparse_weight: float = GROM_SPARSE_WEIGHT,
        grom_optimization_steps: int = GROM_OPTIMIZATION_STEPS,
        grom_learning_rate: float = GROM_LEARNING_RATE,
        method_params: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save steering config for a model/task."""
        config = self._load_model_config(model_name)
        steering = SteeringConfig(
            layer=layer, strength=strength, method=method,
            token_aggregation=token_aggregation, prompt_strategy=prompt_strategy,
            normalize_mode=normalize_mode, strategy=strategy, score=score, metric=metric,
            num_directions=num_directions, direction_weighting=direction_weighting,
            retain_weight=retain_weight, independence_weight=independence_weight,
            tecza_optimization_steps=tecza_optimization_steps, use_caa_init=use_caa_init,
            cone_constraint=cone_constraint, min_cosine_similarity=min_cosine_similarity,
            max_cosine_similarity=max_cosine_similarity, sensor_layer=sensor_layer,
            steering_layers=steering_layers, condition_threshold=condition_threshold,
            gate_temperature=gate_temperature, per_layer_scaling=per_layer_scaling,
            use_entropy_scaling=use_entropy_scaling, max_alpha=max_alpha,
            learn_threshold=learn_threshold, tetno_optimization_steps=tetno_optimization_steps,
            gate_hidden_dim=gate_hidden_dim, intensity_hidden_dim=intensity_hidden_dim,
            behavior_weight=behavior_weight, sparse_weight=sparse_weight,
            grom_optimization_steps=grom_optimization_steps, grom_learning_rate=grom_learning_rate,
            method_params=method_params or {},
        )
        if task_name:
            if task_name not in config.tasks:
                config.tasks[task_name] = TaskConfig(task_name=task_name)
            config.tasks[task_name].steering = steering
            config.tasks[task_name].optimization_method = optimization_method
            config.tasks[task_name].updated_at = datetime.now().isoformat()
        if set_as_default or not task_name:
            config.default_steering = steering
        return self._save_model_config(config)

    def get_steering_config(self, model_name: str, task_name: Optional[str] = None) -> Optional[SteeringConfig]:
        """Get steering config for a model/task."""
        config = self._load_model_config(model_name)
        if task_name and task_name in config.tasks:
            if config.tasks[task_name].steering:
                return config.tasks[task_name].steering
        return config.default_steering

    def save_weight_modification_config(
        self, model_name: str, task_name: Optional[str] = None, trait_label: str = "",
        method: str = "directional", max_weight: float = DEFAULT_STRENGTH,
        min_weight: float = 0.0,
        max_weight_position: float = BLEND_DEFAULT,
        min_weight_distance: float = BLEND_DEFAULT,
        strength: float = DEFAULT_STRENGTH, num_pairs: int = PAIRS_NUM_PAIRS,
        alpha: float = BROYDEN_DEFAULT_ALPHA,
        additive_method: str = "bias", components: Optional[List[str]] = None,
        normalize_vectors: bool = True, norm_preserve: bool = True,
        use_biprojection: bool = True, use_kernel: bool = True,
        score: float = DEFAULT_SCORE, baseline_score: float = DEFAULT_SCORE,
        output_dir: str = "",
        optimization_method: str = "manual", set_as_default: bool = False,
    ) -> Path:
        """Save weight modification config for a model/task."""
        config = self._load_model_config(model_name)
        weight_mod = WeightModificationConfig(
            method=method, max_weight=max_weight, min_weight=min_weight,
            max_weight_position=max_weight_position, min_weight_distance=min_weight_distance,
            strength=strength, num_pairs=num_pairs, alpha=alpha,
            additive_method=additive_method, components=components or ["self_attn.o_proj", "mlp.down_proj"],
            normalize_vectors=normalize_vectors, norm_preserve=norm_preserve,
            use_biprojection=use_biprojection, use_kernel=use_kernel,
            score=score, baseline_score=baseline_score, output_dir=output_dir,
        )
        if task_name:
            if task_name not in config.tasks:
                config.tasks[task_name] = TaskConfig(task_name=task_name)
            config.tasks[task_name].weight_modification = weight_mod
            config.tasks[task_name].optimization_method = optimization_method
            config.tasks[task_name].updated_at = datetime.now().isoformat()
        if set_as_default or not task_name:
            config.default_weight_modification = weight_mod
        return self._save_model_config(config)

    def get_weight_modification_config(self, model_name: str, task_name: Optional[str] = None) -> Optional[WeightModificationConfig]:
        """Get weight modification config for a model/task."""
        config = self._load_model_config(model_name)
        if task_name and task_name in config.tasks:
            if config.tasks[task_name].weight_modification:
                return config.tasks[task_name].weight_modification
        return config.default_weight_modification
