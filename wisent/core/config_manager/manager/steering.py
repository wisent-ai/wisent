"""Steering and weight modification configuration mixin for WisentConfigManager."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from ..types import SteeringConfig, WeightModificationConfig, TaskConfig


class SteeringMixin:
    """Mixin providing steering and weight modification config save/get methods."""

    def save_steering_config(
        self, model_name: str, task_name: Optional[str] = None,
        layer: int = 12, strength: float = 1.0, method: str = "CAA",
        token_aggregation: str = "average", prompt_strategy: str = "question_only",
        normalize_mode: str = "none", strategy: str = "constant",
        score: float = 0.0, metric: str = "accuracy",
        optimization_method: str = "manual", set_as_default: bool = False,
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
    ) -> Path:
        """Save steering config for a model/task."""
        config = self._load_model_config(model_name)
        steering = SteeringConfig(
            layer=layer, strength=strength, method=method,
            token_aggregation=token_aggregation, prompt_strategy=prompt_strategy,
            normalize_mode=normalize_mode, strategy=strategy, score=score, metric=metric,
            num_directions=num_directions, direction_weighting=direction_weighting,
            retain_weight=retain_weight, independence_weight=independence_weight,
            prism_optimization_steps=prism_optimization_steps, use_caa_init=use_caa_init,
            cone_constraint=cone_constraint, min_cosine_similarity=min_cosine_similarity,
            max_cosine_similarity=max_cosine_similarity, sensor_layer=sensor_layer,
            steering_layers=steering_layers, condition_threshold=condition_threshold,
            gate_temperature=gate_temperature, per_layer_scaling=per_layer_scaling,
            use_entropy_scaling=use_entropy_scaling, max_alpha=max_alpha,
            learn_threshold=learn_threshold, pulse_optimization_steps=pulse_optimization_steps,
            gate_hidden_dim=gate_hidden_dim, intensity_hidden_dim=intensity_hidden_dim,
            behavior_weight=behavior_weight, sparse_weight=sparse_weight,
            titan_optimization_steps=titan_optimization_steps, titan_learning_rate=titan_learning_rate,
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
        method: str = "directional", max_weight: float = 1.0, min_weight: float = 0.0,
        max_weight_position: float = 0.5, min_weight_distance: float = 0.5,
        strength: float = 1.0, num_pairs: int = 100, alpha: float = 1.0,
        additive_method: str = "bias", components: Optional[List[str]] = None,
        normalize_vectors: bool = True, norm_preserve: bool = True,
        use_biprojection: bool = True, use_kernel: bool = True,
        score: float = 0.0, baseline_score: float = 0.0, output_dir: str = "",
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
