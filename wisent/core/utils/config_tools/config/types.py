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
    DEFAULT_LAYER_CONFIG, DEFAULT_SCORE,
    MOVEMENT_THRESHOLD, BROYDEN_DEFAULT_ALPHA, PAIRS_NUM_PAIRS,
    TECZA_INDEPENDENCE_WEIGHT, DEFAULT_OPTIMIZATION_STEPS,
    TECZA_MIN_COSINE_SIM, TECZA_MAX_COSINE_SIM,
    TETNO_CONDITION_THRESHOLD, TETNO_MAX_ALPHA,
    GROM_GATE_TEMPERATURE, GROM_ROUTER_HIDDEN_DIM, GROM_INTENSITY_HIDDEN_DIM,
    GROM_BEHAVIOR_WEIGHT, GROM_SPARSE_WEIGHT, GROM_OPTIMIZATION_STEPS,
    GROM_LEARNING_RATE, BLEND_DEFAULT, NESTED_CONFIG_NAME_FIELD,
    TASK_CONFIG_NAME_FIELD, TRAIT_CONFIG_NAME_FIELD,
)


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
    layer: int = DEFAULT_LAYER_CONFIG
    token_aggregation: Optional[str] = None
    detection_threshold: float = MOVEMENT_THRESHOLD
    classifier_type: Optional[str] = None
    prompt_construction_strategy: Optional[str] = None
    token_targeting_strategy: Optional[str] = None
    accuracy: float = DEFAULT_SCORE
    f1_score: float = DEFAULT_SCORE
    precision: float = DEFAULT_SCORE
    recall: float = DEFAULT_SCORE


@dataclass
class SteeringConfig(SerializableConfig):
    """Steering optimization parameters with method-specific settings."""
    layer: int = DEFAULT_LAYER_CONFIG
    strength: Optional[float] = None
    method: Optional[str] = None
    token_aggregation: Optional[str] = None
    prompt_strategy: Optional[str] = None
    normalize_mode: Optional[str] = None
    strategy: Optional[str] = None
    score: float = DEFAULT_SCORE
    metric: Optional[str] = None
    # TECZA parameters
    num_directions: int = 1
    direction_weighting: Optional[str] = None
    retain_weight: float = DEFAULT_SCORE
    independence_weight: float = TECZA_INDEPENDENCE_WEIGHT
    tecza_optimization_steps: int = DEFAULT_OPTIMIZATION_STEPS
    use_caa_init: bool = True
    cone_constraint: bool = True
    min_cosine_similarity: float = TECZA_MIN_COSINE_SIM
    max_cosine_similarity: float = TECZA_MAX_COSINE_SIM
    # TETNO parameters
    sensor_layer: int = -1
    steering_layers: Optional[str] = None
    condition_threshold: float = TETNO_CONDITION_THRESHOLD
    gate_temperature: float = GROM_GATE_TEMPERATURE
    per_layer_scaling: bool = True
    use_entropy_scaling: bool = False
    max_alpha: float = TETNO_MAX_ALPHA
    learn_threshold: bool = True
    tetno_optimization_steps: int = DEFAULT_OPTIMIZATION_STEPS
    # GROM parameters
    gate_hidden_dim: int = GROM_ROUTER_HIDDEN_DIM
    intensity_hidden_dim: int = GROM_INTENSITY_HIDDEN_DIM
    behavior_weight: float = GROM_BEHAVIOR_WEIGHT
    sparse_weight: float = GROM_SPARSE_WEIGHT
    grom_optimization_steps: int = GROM_OPTIMIZATION_STEPS
    grom_learning_rate: float = GROM_LEARNING_RATE
    method_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightModificationConfig(SerializableConfig):
    """Weight modification (directional projection/additive) parameters."""
    method: Optional[str] = None
    max_weight: Optional[float] = None
    min_weight: float = DEFAULT_SCORE
    max_weight_position: float = BLEND_DEFAULT
    min_weight_distance: float = BLEND_DEFAULT
    strength: Optional[float] = None
    num_pairs: int = PAIRS_NUM_PAIRS
    alpha: float = BROYDEN_DEFAULT_ALPHA
    additive_method: Optional[str] = None
    components: List[str] = field(default_factory=lambda: ["self_attn.o_proj", "mlp.down_proj"])
    normalize_vectors: bool = True
    norm_preserve: bool = True
    use_biprojection: bool = True
    use_kernel: bool = True
    score: float = DEFAULT_SCORE
    baseline_score: float = DEFAULT_SCORE
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
