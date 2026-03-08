"""
Configuration dataclasses and types for Wisent unified config manager.

Contains all configuration dataclasses including:
- ClassificationConfig: Classification optimization parameters
- SteeringConfig: Steering optimization parameters with method-specific settings
- WeightModificationConfig: Weight modification parameters
- TaskConfig: Task-specific configuration container
- TraitConfig: Trait-specific configuration container
- ModelConfig: Complete model configuration
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List, TypeVar, Type
import numpy as np

from wisent.core.utils.config_tools.constants import (
    NESTED_CONFIG_NAME_FIELD,
    TASK_CONFIG_NAME_FIELD, TRAIT_CONFIG_NAME_FIELD,
)
from wisent.core.control.steering_methods.configs.optimal import get_optimal


# Default config location
DEFAULT_CONFIG_DIR = "~/.wisent/configs"

# Type variable for SerializableConfig
T = TypeVar("T", bound="SerializableConfig")


class SerializableConfig:
    """Mixin providing to_dict() and from_dict() for dataclasses."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class ClassificationConfig(SerializableConfig):
    """Classification optimization parameters."""
    layer: Optional[int] = None
    token_aggregation: Optional[str] = None
    detection_threshold: Optional[float] = None
    classifier_type: Optional[str] = None
    prompt_construction_strategy: Optional[str] = None
    token_targeting_strategy: Optional[str] = None
    accuracy: float = None
    f1_score: float = None
    precision: float = None
    recall: float = None


@dataclass
class SteeringConfig(SerializableConfig):
    """Steering optimization parameters with method-specific settings."""
    layer: Optional[int] = None
    strength: Optional[float] = None
    method: Optional[str] = None
    token_aggregation: Optional[str] = None
    prompt_strategy: Optional[str] = None
    normalize_mode: Optional[str] = None
    strategy: Optional[str] = None
    score: float = None
    metric: Optional[str] = None
    # TECZA parameters
    num_directions: Optional[int] = None
    direction_weighting: Optional[str] = None
    retain_weight: Optional[float] = None
    independence_weight: Optional[float] = None
    tecza_optimization_steps: Optional[int] = None
    use_caa_init: bool = field(default_factory=lambda: get_optimal("use_caa_init"))
    cone_constraint: bool = field(default_factory=lambda: get_optimal("cone_constraint"))
    min_cosine_similarity: Optional[float] = None
    max_cosine_similarity: Optional[float] = None
    # TETNO parameters
    sensor_layer: Optional[int] = None
    steering_layers: Optional[str] = None
    condition_threshold: Optional[float] = None
    gate_temperature: Optional[float] = None
    per_layer_scaling: bool = field(default_factory=lambda: get_optimal("per_layer_scaling"))
    use_entropy_scaling: bool = field(default_factory=lambda: get_optimal("use_entropy_scaling"))
    max_alpha: Optional[float] = None
    learn_threshold: bool = field(default_factory=lambda: get_optimal("learn_threshold"))
    tetno_optimization_steps: Optional[int] = None
    # GROM parameters
    gate_hidden_dim: Optional[int] = None
    intensity_hidden_dim: Optional[int] = None
    behavior_weight: Optional[float] = None
    sparse_weight: Optional[float] = None
    grom_optimization_steps: Optional[int] = None
    grom_learning_rate: Optional[float] = None
    method_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightModificationConfig(SerializableConfig):
    """Weight modification (directional projection/additive) parameters."""
    method: Optional[str] = None
    max_weight: Optional[float] = None
    min_weight: float = None
    max_weight_position: float = None
    min_weight_distance: float = None
    strength: Optional[float] = None
    num_pairs: Optional[int] = None
    alpha: Optional[float] = None
    additive_method: Optional[str] = None
    components: List[str] = field(default_factory=lambda: ["self_attn.o_proj", "mlp.down_proj"])
    normalize_vectors: bool = True
    norm_preserve: bool = True
    use_biprojection: bool = True
    use_kernel: bool = True
    score: float = None
    baseline_score: float = None
    output_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightModificationConfig":
        if "components" in data and isinstance(data["components"], str):
            data["components"] = [data["components"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class NestedConfigMixin:
    """Mixin for configs with nested classification/steering/weight_modification configs."""
    _name_field: str = NESTED_CONFIG_NAME_FIELD
    classification: Optional[ClassificationConfig] = None
    steering: Optional[SteeringConfig] = None
    weight_modification: Optional[WeightModificationConfig] = None
    updated_at: str = ""
    optimization_method: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            self._name_field: getattr(self, self._name_field),
            "updated_at": self.updated_at,
            "optimization_method": self.optimization_method,
        }
        if self.classification:
            result["classification"] = self.classification.to_dict()
        if self.steering:
            result["steering"] = self.steering.to_dict()
        if self.weight_modification:
            result["weight_modification"] = self.weight_modification.to_dict()
        return result

    @classmethod
    def _from_dict_common(cls, config, data: Dict[str, Any]):
        if "classification" in data:
            config.classification = ClassificationConfig.from_dict(data["classification"])
        if "steering" in data:
            config.steering = SteeringConfig.from_dict(data["steering"])
        if "weight_modification" in data:
            config.weight_modification = WeightModificationConfig.from_dict(data["weight_modification"])
        return config


@dataclass
class TaskConfig(NestedConfigMixin):
    """Configuration for a specific benchmark task."""
    _name_field: str = field(default=TASK_CONFIG_NAME_FIELD, init=False, repr=False)
    task_name: str = ""
    classification: Optional[ClassificationConfig] = None
    steering: Optional[SteeringConfig] = None
    weight_modification: Optional[WeightModificationConfig] = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    optimization_method: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        config = cls(
            task_name=data.get("task_name", ""),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            optimization_method=data.get("optimization_method"),
        )
        return cls._from_dict_common(config, data)


@dataclass
class TraitConfig(NestedConfigMixin):
    """Configuration for a specific behavioral trait."""
    _name_field: str = field(default=TRAIT_CONFIG_NAME_FIELD, init=False, repr=False)
    trait_name: str = ""
    classification: Optional[ClassificationConfig] = None
    steering: Optional[SteeringConfig] = None
    weight_modification: Optional[WeightModificationConfig] = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    optimization_method: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraitConfig":
        config = cls(
            trait_name=data.get("trait_name", ""),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            optimization_method=data.get("optimization_method"),
        )
        return cls._from_dict_common(config, data)


@dataclass
class ModelConfig:
    """Complete configuration for a model."""
    model_name: str
    num_layers: int = 0
    default_classification: Optional[ClassificationConfig] = None
    default_steering: Optional[SteeringConfig] = None
    default_weight_modification: Optional[WeightModificationConfig] = None
    tasks: Dict[str, TaskConfig] = field(default_factory=dict)
    traits: Dict[str, TraitConfig] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    config_version: str = "2.1"

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "model_name": self.model_name, "num_layers": self.num_layers,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "config_version": self.config_version,
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "traits": {k: v.to_dict() for k, v in self.traits.items()},
        }
        if self.default_classification:
            result["default_classification"] = self.default_classification.to_dict()
        if self.default_steering:
            result["default_steering"] = self.default_steering.to_dict()
        if self.default_weight_modification:
            result["default_weight_modification"] = self.default_weight_modification.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        config = cls(
            model_name=data.get("model_name", ""), num_layers=data.get("num_layers", 0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            config_version=data.get("config_version", "2.1"),
        )
        if "default_classification" in data:
            config.default_classification = ClassificationConfig.from_dict(data["default_classification"])
        if "default_steering" in data:
            config.default_steering = SteeringConfig.from_dict(data["default_steering"])
        if "default_weight_modification" in data:
            config.default_weight_modification = WeightModificationConfig.from_dict(data["default_weight_modification"])
        if "tasks" in data:
            config.tasks = {k: TaskConfig.from_dict(v) for k, v in data["tasks"].items()}
        if "traits" in data:
            config.traits = {k: TraitConfig.from_dict(v) for k, v in data["traits"].items()}
        return config
