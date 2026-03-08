"""Steering and weight modification configuration mixin for WisentConfigManager."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from wisent.core import constants as _C
from wisent.core.control.steering_methods.configs.optimal import get_optimal
from ..types import SteeringConfig, WeightModificationConfig, TaskConfig


class SteeringMixin:
    """Mixin providing steering and weight modification config save/get methods."""

    def save_steering_config(
        self, model_name: str, method: str, token_aggregation: str,
        prompt_strategy: str, normalize_mode: str, strategy: str,
        optimization_method: str, metric: str, direction_weighting: str,
        task_name: Optional[str] = None,
        layer: Optional[int] = None, strength: Optional[float] = None,
        score: float = None,
        set_as_default: bool = False, num_directions: Optional[int] = None,
        retain_weight: Optional[float] = None,
        independence_weight: Optional[float] = None,
        tecza_optimization_steps: Optional[int] = None,
        use_caa_init: bool = get_optimal("use_caa_init"),
        cone_constraint: bool = get_optimal("cone_constraint"),
        min_cosine_similarity: Optional[float] = None,
        max_cosine_similarity: Optional[float] = None,
        sensor_layer: Optional[int] = None,
        steering_layers: Optional[str] = None,
        condition_threshold: Optional[float] = None,
        gate_temperature: Optional[float] = None,
        per_layer_scaling: bool = get_optimal("per_layer_scaling"),
        use_entropy_scaling: bool = get_optimal("use_entropy_scaling"),
        max_alpha: Optional[float] = None,
        learn_threshold: bool = get_optimal("learn_threshold"),
        tetno_optimization_steps: Optional[int] = None,
        gate_hidden_dim: Optional[int] = None,
        intensity_hidden_dim: Optional[int] = None,
        behavior_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
        grom_optimization_steps: Optional[int] = None,
        grom_learning_rate: Optional[float] = None,
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
        self, model_name: str, method: str, additive_method: str,
        optimization_method: str, num_pairs: int,
        task_name: Optional[str] = None, trait_label: Optional[str] = None,
        max_weight: Optional[float] = None,
        min_weight: Optional[float] = None,
        max_weight_position: float = None,
        min_weight_distance: float = None,
        strength: Optional[float] = None,
        alpha: Optional[float] = None,
        components: Optional[List[str]] = None,
        normalize_vectors: bool = True, norm_preserve: bool = True,
        use_biprojection: bool = True, use_kernel: bool = True,
        score: float = None, baseline_score: float = None,
        output_dir: Optional[str] = None,
        set_as_default: bool = False,
    ) -> Path:
        """Save weight modification config for a model/task."""
        config = self._load_model_config(model_name)
        weight_mod = WeightModificationConfig(
            method=method, max_weight=max_weight, min_weight=min_weight,
            max_weight_position=max_weight_position, min_weight_distance=min_weight_distance,
            strength=strength, num_pairs=num_pairs,
            additive_method=additive_method, components=components or ["self_attn.o_proj", "mlp.down_proj"],
            normalize_vectors=normalize_vectors, norm_preserve=norm_preserve,
            use_biprojection=use_biprojection, use_kernel=use_kernel,
            score=score, baseline_score=baseline_score, output_dir=output_dir,
        )
        if alpha is not None:
            weight_mod.alpha = alpha
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
