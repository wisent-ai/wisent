"""Trait configuration and general methods mixin for WisentConfigManager."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from ..types import ClassificationConfig, SteeringConfig, WeightModificationConfig, TraitConfig, ModelConfig


class TraitsMixin:
    """Mixin providing trait config and general management methods."""

    def save_trait_classification_config(
        self, model_name: str, trait_name: str, layer: int = 12,
        token_aggregation: str = "average", detection_threshold: float = 0.6,
        classifier_type: str = "logistic", prompt_construction_strategy: str = "multiple_choice",
        token_targeting_strategy: str = "last_token", accuracy: float = 0.0,
        f1_score: float = 0.0, precision: float = 0.0, recall: float = 0.0,
        optimization_method: str = "manual", set_as_default: bool = False,
    ) -> Path:
        """Save classification config for a trait."""
        config = self._load_model_config(model_name)
        classification = ClassificationConfig(
            layer=layer, token_aggregation=token_aggregation, detection_threshold=detection_threshold,
            classifier_type=classifier_type, prompt_construction_strategy=prompt_construction_strategy,
            token_targeting_strategy=token_targeting_strategy, accuracy=accuracy,
            f1_score=f1_score, precision=precision, recall=recall,
        )
        if trait_name not in config.traits:
            config.traits[trait_name] = TraitConfig(trait_name=trait_name)
        config.traits[trait_name].classification = classification
        config.traits[trait_name].optimization_method = optimization_method
        config.traits[trait_name].updated_at = datetime.now().isoformat()
        if set_as_default:
            config.default_classification = classification
        return self._save_model_config(config)

    def get_trait_classification_config(self, model_name: str, trait_name: str) -> Optional[ClassificationConfig]:
        """Get classification config for a trait."""
        config = self._load_model_config(model_name)
        if trait_name in config.traits and config.traits[trait_name].classification:
            return config.traits[trait_name].classification
        return config.default_classification

    def save_trait_steering_config(
        self, model_name: str, trait_name: str, layer: int = 12, strength: float = 1.0,
        method: str = "CAA", token_aggregation: str = "average",
        prompt_strategy: str = "question_only", normalize_mode: str = "none",
        score: float = 0.0, metric: str = "accuracy",
        optimization_method: str = "manual", set_as_default: bool = False,
    ) -> Path:
        """Save steering config for a trait."""
        config = self._load_model_config(model_name)
        steering = SteeringConfig(
            layer=layer, strength=strength, method=method, token_aggregation=token_aggregation,
            prompt_strategy=prompt_strategy, normalize_mode=normalize_mode, score=score, metric=metric,
        )
        if trait_name not in config.traits:
            config.traits[trait_name] = TraitConfig(trait_name=trait_name)
        config.traits[trait_name].steering = steering
        config.traits[trait_name].optimization_method = optimization_method
        config.traits[trait_name].updated_at = datetime.now().isoformat()
        if set_as_default:
            config.default_steering = steering
        return self._save_model_config(config)

    def get_trait_steering_config(self, model_name: str, trait_name: str) -> Optional[SteeringConfig]:
        """Get steering config for a trait."""
        config = self._load_model_config(model_name)
        if trait_name in config.traits and config.traits[trait_name].steering:
            return config.traits[trait_name].steering
        return config.default_steering

    def save_trait_weight_modification_config(
        self, model_name: str, trait_name: str, method: str = "directional",
        max_weight: float = 1.0, min_weight: float = 0.0, max_weight_position: float = 0.5,
        min_weight_distance: float = 0.5, strength: float = 1.0, num_pairs: int = 100,
        alpha: float = 1.0, additive_method: str = "bias", components: Optional[List[str]] = None,
        normalize_vectors: bool = True, norm_preserve: bool = True,
        use_biprojection: bool = True, use_kernel: bool = True, score: float = 0.0,
        baseline_score: float = 0.0, output_dir: str = "",
        optimization_method: str = "manual", set_as_default: bool = False,
    ) -> Path:
        """Save weight modification config for a trait."""
        config = self._load_model_config(model_name)
        weight_mod = WeightModificationConfig(
            method=method, max_weight=max_weight, min_weight=min_weight,
            max_weight_position=max_weight_position, min_weight_distance=min_weight_distance,
            strength=strength, num_pairs=num_pairs, alpha=alpha, additive_method=additive_method,
            components=components or ["self_attn.o_proj", "mlp.down_proj"],
            normalize_vectors=normalize_vectors, norm_preserve=norm_preserve,
            use_biprojection=use_biprojection, use_kernel=use_kernel,
            score=score, baseline_score=baseline_score, output_dir=output_dir,
        )
        if trait_name not in config.traits:
            config.traits[trait_name] = TraitConfig(trait_name=trait_name)
        config.traits[trait_name].weight_modification = weight_mod
        config.traits[trait_name].optimization_method = optimization_method
        config.traits[trait_name].updated_at = datetime.now().isoformat()
        if set_as_default:
            config.default_weight_modification = weight_mod
        return self._save_model_config(config)

    def get_trait_weight_modification_config(self, model_name: str, trait_name: str) -> Optional[WeightModificationConfig]:
        """Get weight modification config for a trait."""
        config = self._load_model_config(model_name)
        if trait_name in config.traits and config.traits[trait_name].weight_modification:
            return config.traits[trait_name].weight_modification
        return config.default_weight_modification

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get the full model config."""
        return self._load_model_config(model_name)

    def has_config(self, model_name: str) -> bool:
        """Check if a model has any saved configuration."""
        return self._get_config_path(model_name).exists()

    def list_models(self) -> List[str]:
        """List all models with saved configurations."""
        models = []
        for path in self.config_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                models.append(data.get("model_name", path.stem))
            except (json.JSONDecodeError, KeyError):
                continue
        return models

    def list_tasks(self, model_name: str) -> List[str]:
        """List all tasks with saved configurations for a model."""
        return list(self._load_model_config(model_name).tasks.keys())

    def list_traits(self, model_name: str) -> List[str]:
        """List all traits with saved configurations for a model."""
        return list(self._load_model_config(model_name).traits.keys())

    def delete_config(self, model_name: str) -> bool:
        """Delete all configuration for a model."""
        config_path = self._get_config_path(model_name)
        if config_path.exists():
            config_path.unlink()
            if model_name in self._cache:
                del self._cache[model_name]
            return True
        return False

    def delete_task_config(self, model_name: str, task_name: str) -> bool:
        """Delete configuration for a specific task."""
        config = self._load_model_config(model_name)
        if task_name in config.tasks:
            del config.tasks[task_name]
            self._save_model_config(config)
            return True
        return False

    def delete_trait_config(self, model_name: str, trait_name: str) -> bool:
        """Delete configuration for a specific trait."""
        config = self._load_model_config(model_name)
        if trait_name in config.traits:
            del config.traits[trait_name]
            self._save_model_config(config)
            return True
        return False
