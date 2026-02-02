"""
Unified Configuration Manager for Wisent.

Stores all optimized parameters in a single location with a consistent structure:
- Classification parameters (layer, threshold, aggregation, etc.)
- Steering parameters (layer, strength, method, etc.)
- Weight modification parameters (directional projection/additive settings)

All configs are stored per model, with task-specific overrides where applicable.

Location: ~/.wisent/configs/{model_name}.json
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, TypeVar, Type
import numpy as np


# Default config location
DEFAULT_CONFIG_DIR = os.path.expanduser("~/.wisent/configs")

# Type variable for SerializableConfig
T = TypeVar("T", bound="SerializableConfig")


class SerializableConfig:
    """Mixin providing to_dict() and from_dict() for dataclasses.

    All simple config dataclasses should inherit from this to avoid
    duplicating serialization logic.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create config from dictionary, filtering to valid fields only."""
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
    layer: int = 12
    token_aggregation: str = "average"
    detection_threshold: float = 0.6
    classifier_type: str = "logistic"
    prompt_construction_strategy: str = "multiple_choice"
    token_targeting_strategy: str = "last_token"

    # Metrics from optimization
    accuracy: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0


@dataclass
class SteeringConfig(SerializableConfig):
    """Steering optimization parameters."""
    layer: int = 12
    strength: float = 1.0
    method: str = "CAA"
    token_aggregation: str = "average"
    prompt_strategy: str = "question_only"
    normalize_mode: str = "none"
    strategy: str = "constant"  # Steering strategy: constant, initial_only, diminishing

    # Metrics from optimization
    score: float = 0.0
    metric: str = "accuracy"
    
    # ==========================================================================
    # METHOD-SPECIFIC PARAMETERS
    # ==========================================================================
    
    # PRISM parameters
    num_directions: int = 1  # Number of steering directions
    direction_weighting: str = "primary_only"  # primary_only, equal, learned, decay
    retain_weight: float = 0.0  # Weight for retaining original behavior
    independence_weight: float = 0.05  # Weight for direction independence loss
    prism_optimization_steps: int = 100  # Optimization steps for PRISM
    use_caa_init: bool = True  # Initialize from CAA direction
    cone_constraint: bool = True  # Use cone constraint
    min_cosine_similarity: float = 0.3  # Min cosine sim for cone
    max_cosine_similarity: float = 0.95  # Max cosine sim for cone
    
    # PULSE parameters
    sensor_layer: int = -1  # Layer to sense activation patterns (-1 = auto)
    steering_layers: str = ""  # Comma-separated steering layer indices
    condition_threshold: float = 0.5  # Threshold for conditional steering
    gate_temperature: float = 0.5  # Temperature for gating
    per_layer_scaling: bool = True  # Use per-layer scaling
    use_entropy_scaling: bool = False  # Scale by entropy
    max_alpha: float = 2.0  # Maximum steering intensity
    learn_threshold: bool = True  # Learn threshold during training
    pulse_optimization_steps: int = 100  # Optimization steps for PULSE
    
    # TITAN parameters
    gate_hidden_dim: int = 64  # Hidden dimension for gate network
    intensity_hidden_dim: int = 32  # Hidden dimension for intensity network
    behavior_weight: float = 1.0  # Weight for behavior loss
    sparse_weight: float = 0.05  # Weight for sparsity loss
    titan_optimization_steps: int = 200  # Optimization steps for TITAN
    titan_learning_rate: float = 0.005  # Learning rate for TITAN
    
    # Generic method parameters storage (for future methods)
    method_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightModificationConfig(SerializableConfig):
    """Weight modification (directional projection/additive) parameters."""
    method: str = "directional"  # directional or additive

    # Directional projection parameters
    max_weight: float = 1.0
    min_weight: float = 0.0
    max_weight_position: float = 0.5  # As ratio of total layers
    min_weight_distance: float = 0.5  # As ratio of total layers
    strength: float = 1.0
    num_pairs: int = 100

    # Additive parameters
    alpha: float = 1.0
    additive_method: str = "bias"  # bias, weight, or both

    # Common parameters
    components: List[str] = field(default_factory=lambda: ["self_attn.o_proj", "mlp.down_proj"])
    normalize_vectors: bool = True
    norm_preserve: bool = True
    use_biprojection: bool = True
    use_kernel: bool = True

    # Metrics
    score: float = 0.0
    baseline_score: float = 0.0
    output_dir: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightModificationConfig":
        # Handle components field specially since it's a list
        if "components" in data and isinstance(data["components"], str):
            data["components"] = [data["components"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class NestedConfigMixin:
    """Mixin for configs with nested classification/steering/weight_modification configs.

    Subclasses must define a `_name_field` class attribute specifying the primary name field
    (e.g., "task_name" or "trait_name").
    """
    _name_field: str = "name"  # Override in subclass

    classification: Optional[ClassificationConfig] = None
    steering: Optional[SteeringConfig] = None
    weight_modification: Optional[WeightModificationConfig] = None
    updated_at: str = ""
    optimization_method: str = "manual"

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
        """Populate nested configs from dict data."""
        if "classification" in data:
            config.classification = ClassificationConfig.from_dict(data["classification"])
        if "steering" in data:
            config.steering = SteeringConfig.from_dict(data["steering"])
        if "weight_modification" in data:
            config.weight_modification = WeightModificationConfig.from_dict(data["weight_modification"])
        return config


@dataclass
class TaskConfig(NestedConfigMixin):
    """Configuration for a specific benchmark task (e.g., hellaswag, gsm8k, mmlu)."""
    _name_field: str = field(default="task_name", init=False, repr=False)

    task_name: str = ""
    classification: Optional[ClassificationConfig] = None
    steering: Optional[SteeringConfig] = None
    weight_modification: Optional[WeightModificationConfig] = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    optimization_method: str = "manual"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        config = cls(
            task_name=data.get("task_name", ""),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            optimization_method=data.get("optimization_method", "manual"),
        )
        return cls._from_dict_common(config, data)


@dataclass
class TraitConfig(NestedConfigMixin):
    """Configuration for a specific behavioral trait (e.g., coding, honesty, helpfulness)."""
    _name_field: str = field(default="trait_name", init=False, repr=False)

    trait_name: str = ""
    classification: Optional[ClassificationConfig] = None
    steering: Optional[SteeringConfig] = None
    weight_modification: Optional[WeightModificationConfig] = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    optimization_method: str = "manual"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraitConfig":
        config = cls(
            trait_name=data.get("trait_name", ""),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            optimization_method=data.get("optimization_method", "manual"),
        )
        return cls._from_dict_common(config, data)


@dataclass
class ModelConfig:
    """Complete configuration for a model."""
    model_name: str
    num_layers: int = 0

    # Default configs (used when no task/trait-specific config exists)
    default_classification: Optional[ClassificationConfig] = None
    default_steering: Optional[SteeringConfig] = None
    default_weight_modification: Optional[WeightModificationConfig] = None

    # Task-specific configs (benchmark tasks: hellaswag, gsm8k, mmlu, etc.)
    tasks: Dict[str, TaskConfig] = field(default_factory=dict)

    # Trait-specific configs (behavioral traits: coding, honesty, helpfulness, etc.)
    traits: Dict[str, TraitConfig] = field(default_factory=dict)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    config_version: str = "2.1"  # Bumped version for traits support

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "model_name": self.model_name,
            "num_layers": self.num_layers,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
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
            model_name=data.get("model_name", ""),
            num_layers=data.get("num_layers", 0),
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
            config.tasks = {
                k: TaskConfig.from_dict(v) for k, v in data["tasks"].items()
            }

        if "traits" in data:
            config.traits = {
                k: TraitConfig.from_dict(v) for k, v in data["traits"].items()
            }

        return config


class WisentConfigManager:
    """
    Unified configuration manager for all Wisent optimization parameters.

    Stores one JSON file per model at ~/.wisent/configs/{model_name}.json

    Structure:
    {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "num_layers": 16,
        "config_version": "2.1",

        // Default configs (fallback when no specific config exists)
        "default_classification": { ... },
        "default_steering": { ... },
        "default_weight_modification": { ... },

        // Task-specific configs (benchmark tasks)
        "tasks": {
            "hellaswag": {
                "task_name": "hellaswag",
                "classification": { ... },
                "steering": { ... },
                "weight_modification": { ... }
            },
            "gsm8k": { ... }
        },

        // Trait-specific configs (behavioral traits)
        "traits": {
            "coding": {
                "trait_name": "coding",
                "classification": { ... },
                "steering": { ... },
                "weight_modification": { ... }
            },
            "honesty": { ... },
            "helpfulness": { ... }
        }
    }
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the config manager.

        Args:
            config_dir: Directory to store config files. Defaults to ~/.wisent/configs/
        """
        self.config_dir = Path(config_dir or DEFAULT_CONFIG_DIR)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, ModelConfig] = {}

    def _sanitize_model_name(self, model_name: str) -> str:
        """Convert model name to a safe filename."""
        sanitized = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        sanitized = "".join(c for c in sanitized if c.isalnum() or c in "._-")
        return sanitized

    def _get_config_path(self, model_name: str) -> Path:
        """Get the full path to the config file for a model."""
        sanitized_name = self._sanitize_model_name(model_name)
        return self.config_dir / f"{sanitized_name}.json"

    def _load_model_config(self, model_name: str) -> ModelConfig:
        """Load or create a model config."""
        if model_name in self._cache:
            return self._cache[model_name]

        config_path = self._get_config_path(model_name)

        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = json.load(f)
                config = ModelConfig.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load config for {model_name}: {e}")
                config = ModelConfig(model_name=model_name)
        else:
            config = ModelConfig(model_name=model_name)

        self._cache[model_name] = config
        return config

    def _save_model_config(self, config: ModelConfig) -> Path:
        """Save a model config to disk."""
        config.updated_at = datetime.now().isoformat()
        config_path = self._get_config_path(config.model_name)

        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2, cls=NumpyEncoder)

        self._cache[config.model_name] = config
        return config_path

    # ========== Classification Methods ==========

    def save_classification_config(
        self,
        model_name: str,
        task_name: Optional[str] = None,
        layer: int = 12,
        token_aggregation: str = "average",
        detection_threshold: float = 0.6,
        classifier_type: str = "logistic",
        prompt_construction_strategy: str = "multiple_choice",
        token_targeting_strategy: str = "last_token",
        accuracy: float = 0.0,
        f1_score: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
        optimization_method: str = "manual",
        set_as_default: bool = False,
    ) -> Path:
        """
        Save classification config for a model/task.

        Args:
            model_name: Model name/path
            task_name: Task name (None for default config)
            set_as_default: If True, also set as the default config
            ... other classification parameters

        Returns:
            Path to the saved config file
        """
        config = self._load_model_config(model_name)

        classification = ClassificationConfig(
            layer=layer,
            token_aggregation=token_aggregation,
            detection_threshold=detection_threshold,
            classifier_type=classifier_type,
            prompt_construction_strategy=prompt_construction_strategy,
            token_targeting_strategy=token_targeting_strategy,
            accuracy=accuracy,
            f1_score=f1_score,
            precision=precision,
            recall=recall,
        )

        if task_name:
            # Save to task-specific config
            if task_name not in config.tasks:
                config.tasks[task_name] = TaskConfig(task_name=task_name)
            config.tasks[task_name].classification = classification
            config.tasks[task_name].optimization_method = optimization_method
            config.tasks[task_name].updated_at = datetime.now().isoformat()

        if set_as_default or not task_name:
            config.default_classification = classification

        return self._save_model_config(config)

    def get_classification_config(
        self,
        model_name: str,
        task_name: Optional[str] = None,
    ) -> Optional[ClassificationConfig]:
        """
        Get classification config for a model/task.

        Args:
            model_name: Model name/path
            task_name: Task name (returns task-specific if exists, else default)

        Returns:
            ClassificationConfig or None
        """
        config = self._load_model_config(model_name)

        # Try task-specific first
        if task_name and task_name in config.tasks:
            task_config = config.tasks[task_name]
            if task_config.classification:
                return task_config.classification

        # Fall back to default
        return config.default_classification

    # ========== Steering Methods ==========

    def save_steering_config(
        self,
        model_name: str,
        task_name: Optional[str] = None,
        layer: int = 12,
        strength: float = 1.0,
        method: str = "CAA",
        token_aggregation: str = "average",
        prompt_strategy: str = "question_only",
        normalize_mode: str = "none",
        strategy: str = "constant",
        score: float = 0.0,
        metric: str = "accuracy",
        optimization_method: str = "manual",
        set_as_default: bool = False,
        # PRISM parameters
        num_directions: int = 1,
        direction_weighting: str = "primary_only",
        retain_weight: float = 0.0,
        independence_weight: float = 0.05,
        prism_optimization_steps: int = 100,
        use_caa_init: bool = True,
        cone_constraint: bool = True,
        min_cosine_similarity: float = 0.3,
        max_cosine_similarity: float = 0.95,
        # PULSE parameters
        sensor_layer: int = -1,
        steering_layers: str = "",
        condition_threshold: float = 0.5,
        gate_temperature: float = 0.5,
        per_layer_scaling: bool = True,
        use_entropy_scaling: bool = False,
        max_alpha: float = 2.0,
        learn_threshold: bool = True,
        pulse_optimization_steps: int = 100,
        # TITAN parameters
        gate_hidden_dim: int = 64,
        intensity_hidden_dim: int = 32,
        behavior_weight: float = 1.0,
        sparse_weight: float = 0.05,
        titan_optimization_steps: int = 200,
        titan_learning_rate: float = 0.005,
        # Generic method params
        method_params: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save steering config for a model/task."""
        config = self._load_model_config(model_name)

        steering = SteeringConfig(
            layer=layer,
            strength=strength,
            method=method,
            token_aggregation=token_aggregation,
            prompt_strategy=prompt_strategy,
            normalize_mode=normalize_mode,
            strategy=strategy,
            score=score,
            metric=metric,
            # PRISM
            num_directions=num_directions,
            direction_weighting=direction_weighting,
            retain_weight=retain_weight,
            independence_weight=independence_weight,
            prism_optimization_steps=prism_optimization_steps,
            use_caa_init=use_caa_init,
            cone_constraint=cone_constraint,
            min_cosine_similarity=min_cosine_similarity,
            max_cosine_similarity=max_cosine_similarity,
            # PULSE
            sensor_layer=sensor_layer,
            steering_layers=steering_layers,
            condition_threshold=condition_threshold,
            gate_temperature=gate_temperature,
            per_layer_scaling=per_layer_scaling,
            use_entropy_scaling=use_entropy_scaling,
            max_alpha=max_alpha,
            learn_threshold=learn_threshold,
            pulse_optimization_steps=pulse_optimization_steps,
            # TITAN
            gate_hidden_dim=gate_hidden_dim,
            intensity_hidden_dim=intensity_hidden_dim,
            behavior_weight=behavior_weight,
            sparse_weight=sparse_weight,
            titan_optimization_steps=titan_optimization_steps,
            titan_learning_rate=titan_learning_rate,
            # Generic
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

    def get_steering_config(
        self,
        model_name: str,
        task_name: Optional[str] = None,
    ) -> Optional[SteeringConfig]:
        """Get steering config for a model/task."""
        config = self._load_model_config(model_name)

        if task_name and task_name in config.tasks:
            task_config = config.tasks[task_name]
            if task_config.steering:
                return task_config.steering

        return config.default_steering

    # ========== Weight Modification Methods ==========

    def save_weight_modification_config(
        self,
        model_name: str,
        task_name: Optional[str] = None,
        trait_label: str = "",
        method: str = "directional",
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        max_weight_position: float = 0.5,
        min_weight_distance: float = 0.5,
        strength: float = 1.0,
        num_pairs: int = 100,
        alpha: float = 1.0,
        additive_method: str = "bias",
        components: Optional[List[str]] = None,
        normalize_vectors: bool = True,
        norm_preserve: bool = True,
        use_biprojection: bool = True,
        use_kernel: bool = True,
        score: float = 0.0,
        baseline_score: float = 0.0,
        output_dir: str = "",
        optimization_method: str = "manual",
        set_as_default: bool = False,
    ) -> Path:
        """Save weight modification config for a model/task."""
        config = self._load_model_config(model_name)

        weight_mod = WeightModificationConfig(
            method=method,
            max_weight=max_weight,
            min_weight=min_weight,
            max_weight_position=max_weight_position,
            min_weight_distance=min_weight_distance,
            strength=strength,
            num_pairs=num_pairs,
            alpha=alpha,
            additive_method=additive_method,
            components=components or ["self_attn.o_proj", "mlp.down_proj"],
            normalize_vectors=normalize_vectors,
            norm_preserve=norm_preserve,
            use_biprojection=use_biprojection,
            use_kernel=use_kernel,
            score=score,
            baseline_score=baseline_score,
            output_dir=output_dir,
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

    def get_weight_modification_config(
        self,
        model_name: str,
        task_name: Optional[str] = None,
    ) -> Optional[WeightModificationConfig]:
        """Get weight modification config for a model/task."""
        config = self._load_model_config(model_name)

        if task_name and task_name in config.tasks:
            task_config = config.tasks[task_name]
            if task_config.weight_modification:
                return task_config.weight_modification

        return config.default_weight_modification

    # ========== Trait-specific Methods ==========

    def save_trait_classification_config(
        self,
        model_name: str,
        trait_name: str,
        layer: int = 12,
        token_aggregation: str = "average",
        detection_threshold: float = 0.6,
        classifier_type: str = "logistic",
        prompt_construction_strategy: str = "multiple_choice",
        token_targeting_strategy: str = "last_token",
        accuracy: float = 0.0,
        f1_score: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
        optimization_method: str = "manual",
        set_as_default: bool = False,
    ) -> Path:
        """Save classification config for a trait."""
        config = self._load_model_config(model_name)

        classification = ClassificationConfig(
            layer=layer,
            token_aggregation=token_aggregation,
            detection_threshold=detection_threshold,
            classifier_type=classifier_type,
            prompt_construction_strategy=prompt_construction_strategy,
            token_targeting_strategy=token_targeting_strategy,
            accuracy=accuracy,
            f1_score=f1_score,
            precision=precision,
            recall=recall,
        )

        if trait_name not in config.traits:
            config.traits[trait_name] = TraitConfig(trait_name=trait_name)
        config.traits[trait_name].classification = classification
        config.traits[trait_name].optimization_method = optimization_method
        config.traits[trait_name].updated_at = datetime.now().isoformat()

        if set_as_default:
            config.default_classification = classification

        return self._save_model_config(config)

    def get_trait_classification_config(
        self,
        model_name: str,
        trait_name: str,
    ) -> Optional[ClassificationConfig]:
        """Get classification config for a trait."""
        config = self._load_model_config(model_name)

        if trait_name in config.traits:
            trait_config = config.traits[trait_name]
            if trait_config.classification:
                return trait_config.classification

        return config.default_classification

    def save_trait_steering_config(
        self,
        model_name: str,
        trait_name: str,
        layer: int = 12,
        strength: float = 1.0,
        method: str = "CAA",
        token_aggregation: str = "average",
        prompt_strategy: str = "question_only",
        normalize_mode: str = "none",
        score: float = 0.0,
        metric: str = "accuracy",
        optimization_method: str = "manual",
        set_as_default: bool = False,
    ) -> Path:
        """Save steering config for a trait."""
        config = self._load_model_config(model_name)

        steering = SteeringConfig(
            layer=layer,
            strength=strength,
            method=method,
            token_aggregation=token_aggregation,
            prompt_strategy=prompt_strategy,
            normalize_mode=normalize_mode,
            score=score,
            metric=metric,
        )

        if trait_name not in config.traits:
            config.traits[trait_name] = TraitConfig(trait_name=trait_name)
        config.traits[trait_name].steering = steering
        config.traits[trait_name].optimization_method = optimization_method
        config.traits[trait_name].updated_at = datetime.now().isoformat()

        if set_as_default:
            config.default_steering = steering

        return self._save_model_config(config)

    def get_trait_steering_config(
        self,
        model_name: str,
        trait_name: str,
    ) -> Optional[SteeringConfig]:
        """Get steering config for a trait."""
        config = self._load_model_config(model_name)

        if trait_name in config.traits:
            trait_config = config.traits[trait_name]
            if trait_config.steering:
                return trait_config.steering

        return config.default_steering

    def save_trait_weight_modification_config(
        self,
        model_name: str,
        trait_name: str,
        method: str = "directional",
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        max_weight_position: float = 0.5,
        min_weight_distance: float = 0.5,
        strength: float = 1.0,
        num_pairs: int = 100,
        alpha: float = 1.0,
        additive_method: str = "bias",
        components: Optional[List[str]] = None,
        normalize_vectors: bool = True,
        norm_preserve: bool = True,
        use_biprojection: bool = True,
        use_kernel: bool = True,
        score: float = 0.0,
        baseline_score: float = 0.0,
        output_dir: str = "",
        optimization_method: str = "manual",
        set_as_default: bool = False,
    ) -> Path:
        """Save weight modification config for a trait."""
        config = self._load_model_config(model_name)

        weight_mod = WeightModificationConfig(
            method=method,
            max_weight=max_weight,
            min_weight=min_weight,
            max_weight_position=max_weight_position,
            min_weight_distance=min_weight_distance,
            strength=strength,
            num_pairs=num_pairs,
            alpha=alpha,
            additive_method=additive_method,
            components=components or ["self_attn.o_proj", "mlp.down_proj"],
            normalize_vectors=normalize_vectors,
            norm_preserve=norm_preserve,
            use_biprojection=use_biprojection,
            use_kernel=use_kernel,
            score=score,
            baseline_score=baseline_score,
            output_dir=output_dir,
        )

        if trait_name not in config.traits:
            config.traits[trait_name] = TraitConfig(trait_name=trait_name)
        config.traits[trait_name].weight_modification = weight_mod
        config.traits[trait_name].optimization_method = optimization_method
        config.traits[trait_name].updated_at = datetime.now().isoformat()

        if set_as_default:
            config.default_weight_modification = weight_mod

        return self._save_model_config(config)

    def get_trait_weight_modification_config(
        self,
        model_name: str,
        trait_name: str,
    ) -> Optional[WeightModificationConfig]:
        """Get weight modification config for a trait."""
        config = self._load_model_config(model_name)

        if trait_name in config.traits:
            trait_config = config.traits[trait_name]
            if trait_config.weight_modification:
                return trait_config.weight_modification

        return config.default_weight_modification

    # ========== General Methods ==========

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get the full model config."""
        return self._load_model_config(model_name)

    def has_config(self, model_name: str) -> bool:
        """Check if a model has any saved configuration."""
        config_path = self._get_config_path(model_name)
        return config_path.exists()

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
        config = self._load_model_config(model_name)
        return list(config.tasks.keys())

    def list_traits(self, model_name: str) -> List[str]:
        """List all traits with saved configurations for a model."""
        config = self._load_model_config(model_name)
        return list(config.traits.keys())

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


# Global instance
_config_manager: Optional[WisentConfigManager] = None


def get_config_manager() -> WisentConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = WisentConfigManager()
    return _config_manager


# Convenience functions
def save_classification_config(model_name: str, **kwargs) -> Path:
    """Save classification config using global manager."""
    return get_config_manager().save_classification_config(model_name, **kwargs)


def get_classification_config(model_name: str, task_name: Optional[str] = None) -> Optional[ClassificationConfig]:
    """Get classification config using global manager."""
    return get_config_manager().get_classification_config(model_name, task_name)


def save_steering_config(model_name: str, **kwargs) -> Path:
    """Save steering config using global manager."""
    return get_config_manager().save_steering_config(model_name, **kwargs)


def get_steering_config(model_name: str, task_name: Optional[str] = None) -> Optional[SteeringConfig]:
    """Get steering config using global manager."""
    return get_config_manager().get_steering_config(model_name, task_name)


def save_weight_modification_config(model_name: str, **kwargs) -> Path:
    """Save weight modification config using global manager."""
    return get_config_manager().save_weight_modification_config(model_name, **kwargs)


def get_weight_modification_config(
    model_name: str,
    task_name: Optional[str] = None,
) -> Optional[WeightModificationConfig]:
    """Get weight modification config using global manager."""
    return get_config_manager().get_weight_modification_config(model_name, task_name)


# Trait convenience functions
def save_trait_classification_config(model_name: str, trait_name: str, **kwargs) -> Path:
    """Save classification config for a trait using global manager."""
    return get_config_manager().save_trait_classification_config(model_name, trait_name, **kwargs)


def get_trait_classification_config(model_name: str, trait_name: str) -> Optional[ClassificationConfig]:
    """Get classification config for a trait using global manager."""
    return get_config_manager().get_trait_classification_config(model_name, trait_name)


def save_trait_steering_config(model_name: str, trait_name: str, **kwargs) -> Path:
    """Save steering config for a trait using global manager."""
    return get_config_manager().save_trait_steering_config(model_name, trait_name, **kwargs)


def get_trait_steering_config(model_name: str, trait_name: str) -> Optional[SteeringConfig]:
    """Get steering config for a trait using global manager."""
    return get_config_manager().get_trait_steering_config(model_name, trait_name)


def save_trait_weight_modification_config(model_name: str, trait_name: str, **kwargs) -> Path:
    """Save weight modification config for a trait using global manager."""
    return get_config_manager().save_trait_weight_modification_config(model_name, trait_name, **kwargs)


def get_trait_weight_modification_config(model_name: str, trait_name: str) -> Optional[WeightModificationConfig]:
    """Get weight modification config for a trait using global manager."""
    return get_config_manager().get_trait_weight_modification_config(model_name, trait_name)


# ========== Backward Compatibility Layer ==========
# These functions provide compatibility with the old OptimizationCache API

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
    
    # Method-specific parameters
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
    # Generic
    method_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def store_optimization(
    model: str,
    task: str,
    layer: int,
    strength: float,
    method: str = "CAA",
    token_aggregation: str = "average",
    prompt_strategy: str = "question_only",
    strategy: str = "constant",
    score: float = 0.0,
    metric: str = "accuracy",
    metadata: Optional[Dict[str, Any]] = None,
    set_as_default: bool = False,
    # PRISM parameters
    num_directions: int = 1,
    direction_weighting: str = "primary_only",
    retain_weight: float = 0.0,
    independence_weight: float = 0.05,
    prism_optimization_steps: int = 100,
    use_caa_init: bool = True,
    cone_constraint: bool = True,
    min_cosine_similarity: float = 0.3,
    max_cosine_similarity: float = 0.95,
    # PULSE parameters
    sensor_layer: int = -1,
    steering_layers: str = "",
    condition_threshold: float = 0.5,
    gate_temperature: float = 0.5,
    per_layer_scaling: bool = True,
    use_entropy_scaling: bool = False,
    max_alpha: float = 2.0,
    learn_threshold: bool = True,
    pulse_optimization_steps: int = 100,
    # TITAN parameters
    gate_hidden_dim: int = 64,
    intensity_hidden_dim: int = 32,
    behavior_weight: float = 1.0,
    sparse_weight: float = 0.05,
    titan_optimization_steps: int = 200,
    titan_learning_rate: float = 0.005,
    # Generic
    method_params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Backward-compatible function to store steering optimization result.
    Maps to the new unified config manager.
    """
    config_path = save_steering_config(
        model_name=model,
        task_name=task,
        layer=layer,
        strength=strength,
        method=method,
        token_aggregation=token_aggregation,
        prompt_strategy=prompt_strategy,
        strategy=strategy,
        score=score,
        metric=metric,
        optimization_method="optuna" if metadata else "manual",
        set_as_default=set_as_default,
        # PRISM
        num_directions=num_directions,
        direction_weighting=direction_weighting,
        retain_weight=retain_weight,
        independence_weight=independence_weight,
        prism_optimization_steps=prism_optimization_steps,
        use_caa_init=use_caa_init,
        cone_constraint=cone_constraint,
        min_cosine_similarity=min_cosine_similarity,
        max_cosine_similarity=max_cosine_similarity,
        # PULSE
        sensor_layer=sensor_layer,
        steering_layers=steering_layers,
        condition_threshold=condition_threshold,
        gate_temperature=gate_temperature,
        per_layer_scaling=per_layer_scaling,
        use_entropy_scaling=use_entropy_scaling,
        max_alpha=max_alpha,
        learn_threshold=learn_threshold,
        pulse_optimization_steps=pulse_optimization_steps,
        # TITAN
        gate_hidden_dim=gate_hidden_dim,
        intensity_hidden_dim=intensity_hidden_dim,
        behavior_weight=behavior_weight,
        sparse_weight=sparse_weight,
        titan_optimization_steps=titan_optimization_steps,
        titan_learning_rate=titan_learning_rate,
        # Generic
        method_params=method_params,
    )
    # Return a cache key for backward compatibility
    model_normalized = model.replace("/", "_").replace("\\", "_")
    return f"{model_normalized}::{task}::{method}"


def get_cached_optimization(
    model: str,
    task: str,
    method: str = "CAA",
    use_default: bool = True
) -> Optional[OptimizationResult]:
    """
    Backward-compatible function to get cached steering optimization result.
    Maps to the new unified config manager.
    """
    steering = get_steering_config(model, task)

    if steering is None:
        return None

    # Only return if method matches (or method is wildcard)
    if method != "*" and steering.method != method:
        return None

    return OptimizationResult(
        model=model,
        task=task,
        layer=steering.layer,
        strength=steering.strength,
        method=steering.method,
        token_aggregation=steering.token_aggregation,
        prompt_strategy=steering.prompt_strategy,
        strategy=steering.strategy,
        score=steering.score,
        metric=steering.metric,
        # PRISM
        num_directions=steering.num_directions,
        direction_weighting=steering.direction_weighting,
        retain_weight=steering.retain_weight,
        independence_weight=steering.independence_weight,
        prism_optimization_steps=steering.prism_optimization_steps,
        use_caa_init=steering.use_caa_init,
        cone_constraint=steering.cone_constraint,
        min_cosine_similarity=steering.min_cosine_similarity,
        max_cosine_similarity=steering.max_cosine_similarity,
        # PULSE
        sensor_layer=steering.sensor_layer,
        steering_layers=steering.steering_layers,
        condition_threshold=steering.condition_threshold,
        gate_temperature=steering.gate_temperature,
        per_layer_scaling=steering.per_layer_scaling,
        use_entropy_scaling=steering.use_entropy_scaling,
        max_alpha=steering.max_alpha,
        learn_threshold=steering.learn_threshold,
        pulse_optimization_steps=steering.pulse_optimization_steps,
        # TITAN
        gate_hidden_dim=steering.gate_hidden_dim,
        intensity_hidden_dim=steering.intensity_hidden_dim,
        behavior_weight=steering.behavior_weight,
        sparse_weight=steering.sparse_weight,
        titan_optimization_steps=steering.titan_optimization_steps,
        titan_learning_rate=steering.titan_learning_rate,
        # Generic
        method_params=steering.method_params,
    )


# ========== Weight Modification Backward Compatibility ==========
# These functions provide compatibility with the old WeightModificationCache API

@dataclass
class WeightModificationResult:
    """Backward-compatible result class for weight modification cache."""
    model: str
    task: str
    trait_label: str
    method: str = "directional"
    max_weight: float = 1.0
    min_weight: float = 0.0
    max_weight_position: float = 0.5
    min_weight_distance: float = 0.5
    strength: float = 1.0
    num_pairs: int = 100
    alpha: float = 1.0
    additive_method: str = "bias"
    components: List[str] = field(default_factory=lambda: ["self_attn.o_proj", "mlp.down_proj"])
    normalize_vectors: bool = True
    norm_preserve: bool = True
    use_biprojection: bool = True
    use_kernel: bool = True
    score: float = 0.0
    metric: str = "accuracy"
    baseline_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    output_dir: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def store_weight_modification(
    model: str,
    task: str,
    trait_label: str,
    method: str = "directional",
    max_weight: float = 1.0,
    min_weight: float = 0.0,
    max_weight_position: float = 0.5,
    min_weight_distance: float = 0.5,
    strength: float = 1.0,
    num_pairs: int = 100,
    alpha: float = 1.0,
    additive_method: str = "bias",
    components: Optional[List[str]] = None,
    normalize_vectors: bool = True,
    norm_preserve: bool = True,
    use_biprojection: bool = True,
    use_kernel: bool = True,
    score: float = 0.0,
    metric: str = "accuracy",
    baseline_score: float = 0.0,
    output_dir: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    set_as_default: bool = False,
) -> str:
    """
    Backward-compatible function to store weight modification result.
    Maps to the new unified config manager.

    If trait_label is provided, stores as a trait config.
    If task is provided without trait_label, stores as a task config.
    """
    # If trait_label is provided, store as trait config
    if trait_label:
        save_trait_weight_modification_config(
            model_name=model,
            trait_name=trait_label,
            method=method,
            max_weight=max_weight,
            min_weight=min_weight,
            max_weight_position=max_weight_position,
            min_weight_distance=min_weight_distance,
            strength=strength,
            num_pairs=num_pairs,
            alpha=alpha,
            additive_method=additive_method,
            components=components,
            normalize_vectors=normalize_vectors,
            norm_preserve=norm_preserve,
            use_biprojection=use_biprojection,
            use_kernel=use_kernel,
            score=score,
            baseline_score=baseline_score,
            output_dir=output_dir,
            optimization_method="optuna" if metadata else "manual",
            set_as_default=set_as_default,
        )
    else:
        # Store as task config
        save_weight_modification_config(
            model_name=model,
            task_name=task,
            method=method,
            max_weight=max_weight,
            min_weight=min_weight,
            max_weight_position=max_weight_position,
            min_weight_distance=min_weight_distance,
            strength=strength,
            num_pairs=num_pairs,
            alpha=alpha,
            additive_method=additive_method,
            components=components,
            normalize_vectors=normalize_vectors,
            norm_preserve=norm_preserve,
            use_biprojection=use_biprojection,
            use_kernel=use_kernel,
            score=score,
            baseline_score=baseline_score,
            output_dir=output_dir,
            optimization_method="optuna" if metadata else "manual",
            set_as_default=set_as_default,
        )

    # Return a cache key for backward compatibility
    model_normalized = model.replace("/", "_").replace("\\", "_")
    return f"{model_normalized}::{task}::{trait_label}::{method}"


def get_cached_weight_modification(
    model: str,
    task: str,
    trait_label: str,
    method: str = "directional",
    use_default: bool = True,
) -> Optional[WeightModificationResult]:
    """
    Backward-compatible function to get cached weight modification result.
    Maps to the new unified config manager.

    If trait_label is provided, gets from trait config.
    If task is provided without trait_label, gets from task config.
    """
    # If trait_label is provided, get from trait config
    if trait_label:
        weight_mod = get_trait_weight_modification_config(model, trait_label)
    else:
        weight_mod = get_weight_modification_config(model, task)

    if weight_mod is None:
        return None

    # Only return if method matches
    if method != "*" and weight_mod.method != method:
        return None

    return WeightModificationResult(
        model=model,
        task=task,
        trait_label=trait_label,
        method=weight_mod.method,
        max_weight=weight_mod.max_weight,
        min_weight=weight_mod.min_weight,
        max_weight_position=weight_mod.max_weight_position,
        min_weight_distance=weight_mod.min_weight_distance,
        strength=weight_mod.strength,
        num_pairs=weight_mod.num_pairs,
        alpha=weight_mod.alpha,
        additive_method=weight_mod.additive_method,
        components=weight_mod.components,
        normalize_vectors=weight_mod.normalize_vectors,
        norm_preserve=weight_mod.norm_preserve,
        use_biprojection=weight_mod.use_biprojection,
        use_kernel=weight_mod.use_kernel,
        score=weight_mod.score,
        baseline_score=weight_mod.baseline_score,
        output_dir=weight_mod.output_dir,
    )


def get_weight_modification_cache():
    """
    Backward-compatible function that returns the global config manager.
    This allows existing code expecting a cache object to work.
    """
    return get_config_manager()


# ========== Steering Cache Backward Compatibility ==========
# Provides compatibility with the old OptimizationCache class API

class OptimizationCache:
    """
    Backward-compatible wrapper class for the unified config manager.
    Provides the same interface as the old OptimizationCache class.
    """

    def __init__(self):
        self._manager = get_config_manager()
        self._defaults: Dict[str, str] = {}

    def _make_key(self, model: str, task: str, method: str = "CAA") -> str:
        model_normalized = model.replace("/", "_").replace("\\", "_")
        return f"{model_normalized}::{task}::{method}"

    def store(
        self,
        model: str,
        task: str,
        layer: int,
        strength: float,
        method: str = "CAA",
        token_aggregation: str = "average",
        prompt_strategy: str = "question_only",
        score: float = 0.0,
        metric: str = "accuracy",
        metadata: Optional[Dict[str, Any]] = None,
        set_as_default: bool = False,
    ) -> str:
        """Store an optimization result."""
        return store_optimization(
            model=model,
            task=task,
            layer=layer,
            strength=strength,
            method=method,
            token_aggregation=token_aggregation,
            prompt_strategy=prompt_strategy,
            score=score,
            metric=metric,
            metadata=metadata,
            set_as_default=set_as_default,
        )

    def get(
        self,
        model: str,
        task: str,
        method: str = "CAA",
    ) -> Optional[OptimizationResult]:
        """Get a cached optimization result."""
        return get_cached_optimization(model, task, method, use_default=False)

    def get_default(self, model: str, task: str) -> Optional[OptimizationResult]:
        """Get the default optimization result for a model/task."""
        return get_cached_optimization(model, task, "*", use_default=True)

    def set_default(self, model: str, task: str, method: str = "CAA") -> bool:
        """Set a cached result as the default."""
        # Get the existing steering config
        steering = get_steering_config(model, task)
        if steering is None:
            return False

        # Save it again with set_as_default=True
        save_steering_config(
            model_name=model,
            task_name=task,
            layer=steering.layer,
            strength=steering.strength,
            method=steering.method,
            token_aggregation=steering.token_aggregation,
            prompt_strategy=steering.prompt_strategy,
            score=steering.score,
            metric=steering.metric,
            set_as_default=True,
        )
        return True

    def exists(self, model: str, task: str, method: str = "CAA") -> bool:
        """Check if a cached result exists."""
        result = get_cached_optimization(model, task, method, use_default=False)
        return result is not None

    def list_cached(
        self,
        model: Optional[str] = None,
        task: Optional[str] = None,
    ) -> List[OptimizationResult]:
        """List cached results, optionally filtered."""
        results = []

        # If model specified, only look at that model
        if model:
            models = [model]
        else:
            models = self._manager.list_models()

        for m in models:
            config = self._manager.get_model_config(m)

            # Check default steering
            if config.default_steering:
                if not task:  # No task filter, include default
                    results.append(OptimizationResult(
                        model=m,
                        task="(default)",
                        layer=config.default_steering.layer,
                        strength=config.default_steering.strength,
                        method=config.default_steering.method,
                        token_aggregation=config.default_steering.token_aggregation,
                        prompt_strategy=config.default_steering.prompt_strategy,
                        score=config.default_steering.score,
                        metric=config.default_steering.metric,
                    ))

            # Check task-specific steering
            for task_name, task_config in config.tasks.items():
                if task and task_name != task:
                    continue
                if task_config.steering:
                    results.append(OptimizationResult(
                        model=m,
                        task=task_name,
                        layer=task_config.steering.layer,
                        strength=task_config.steering.strength,
                        method=task_config.steering.method,
                        token_aggregation=task_config.steering.token_aggregation,
                        prompt_strategy=task_config.steering.prompt_strategy,
                        score=task_config.steering.score,
                        metric=task_config.steering.metric,
                    ))

        return results

    def delete(self, model: str, task: str, method: str = "CAA") -> bool:
        """Delete a cached result."""
        config = self._manager.get_model_config(model)

        if task in config.tasks and config.tasks[task].steering:
            config.tasks[task].steering = None
            self._manager._save_model_config(config)
            return True
        return False

    def clear(self) -> int:
        """Clear all cached steering results."""
        count = 0
        for model in self._manager.list_models():
            config = self._manager.get_model_config(model)

            if config.default_steering:
                config.default_steering = None
                count += 1

            for task_config in config.tasks.values():
                if task_config.steering:
                    task_config.steering = None
                    count += 1

            self._manager._save_model_config(config)

        return count

    def _save(self) -> None:
        """No-op for compatibility - config manager auto-saves."""
        pass


# Global cache instance for backward compatibility
_legacy_cache: Optional[OptimizationCache] = None


def get_cache() -> OptimizationCache:
    """Get the global optimization cache instance (backward compatible)."""
    global _legacy_cache
    if _legacy_cache is None:
        _legacy_cache = OptimizationCache()
    return _legacy_cache


# ========== ModelConfigManager Backward Compatibility ==========
# Provides compatibility with the old ModelConfigManager class API

class ModelConfigManager:
    """
    Backward-compatible wrapper class for the unified config manager.
    Provides the same interface as the old ModelConfigManager class.
    """

    def __init__(self, config_dir: Optional[str] = None):
        self._manager = get_config_manager()
        # Ignore config_dir parameter since unified manager has fixed location

    def _sanitize_model_name(self, model_name: str) -> str:
        return self._manager._sanitize_model_name(model_name)

    def _get_config_path(self, model_name: str) -> str:
        return str(self._manager._get_config_path(model_name))

    def save_model_config(
        self,
        model_name: str,
        classification_layer: int,
        steering_layer: Optional[int] = None,
        token_aggregation: str = "average",
        detection_threshold: float = 0.6,
        optimization_method: str = "manual",
        optimization_metrics: Optional[Dict[str, Any]] = None,
        task_specific_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """Save model configuration using unified config manager."""
        if steering_layer is None:
            steering_layer = classification_layer

        # Save classification config
        save_classification_config(
            model_name=model_name,
            layer=classification_layer,
            token_aggregation=token_aggregation,
            detection_threshold=detection_threshold,
            optimization_method=optimization_method,
            set_as_default=True,
        )

        # Save steering config with same layer
        save_steering_config(
            model_name=model_name,
            layer=steering_layer,
            optimization_method=optimization_method,
            set_as_default=True,
        )

        return str(self._manager._get_config_path(model_name))

    def load_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load model configuration in legacy format."""
        config = self._manager.get_model_config(model_name)

        # Return None if no configs exist
        if not config.default_classification and not config.default_steering:
            return None

        # Build legacy format
        result = {
            "model_name": model_name,
            "optimal_parameters": {},
            "task_specific_overrides": {},
            "optimization_metrics": {},
            "config_version": "2.0",
        }

        if config.default_classification:
            result["optimal_parameters"]["classification_layer"] = config.default_classification.layer
            result["optimal_parameters"]["token_aggregation"] = config.default_classification.token_aggregation
            result["optimal_parameters"]["detection_threshold"] = config.default_classification.detection_threshold

        if config.default_steering:
            result["optimal_parameters"]["steering_layer"] = config.default_steering.layer

        return result

    def has_model_config(self, model_name: str) -> bool:
        """Check if a model has a saved configuration."""
        return self._manager.has_config(model_name)

    def get_optimal_parameters(
        self,
        model_name: str,
        task_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get optimal parameters for a model."""
        config = self.load_model_config(model_name)
        if not config:
            return None
        return config.get("optimal_parameters", {})

    def list_model_configs(self) -> List[Dict[str, Any]]:
        """List all available model configurations."""
        configs = []
        for model_name in self._manager.list_models():
            config = self.load_model_config(model_name)
            if config:
                configs.append({
                    "model_name": model_name,
                    "classification_layer": config.get("optimal_parameters", {}).get("classification_layer"),
                    "steering_layer": config.get("optimal_parameters", {}).get("steering_layer"),
                })
        return configs

    def remove_model_config(self, model_name: str) -> bool:
        """Remove a model configuration."""
        return self._manager.delete_config(model_name)


def get_default_manager() -> ModelConfigManager:
    """Get a default ModelConfigManager instance (backward compatible)."""
    return ModelConfigManager()


def save_model_config(model_name: str, **kwargs) -> str:
    """Save model configuration using default manager (backward compatible)."""
    return ModelConfigManager().save_model_config(model_name, **kwargs)


def load_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    """Load model configuration using default manager (backward compatible)."""
    return ModelConfigManager().load_model_config(model_name)


def get_optimal_parameters(model_name: str, task_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get optimal parameters using default manager (backward compatible)."""
    return ModelConfigManager().get_optimal_parameters(model_name, task_name)
