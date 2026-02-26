"""Method configuration dataclasses for steering optimization."""
from dataclasses import dataclass, field
from typing import List, Dict, Any
from wisent.core.constants import (
    BROYDEN_DEFAULT_ALPHA,
    BROYDEN_DEFAULT_ALPHA_DECAY,
    BROYDEN_DEFAULT_BETA,
    BROYDEN_DEFAULT_ETA,
    BROYDEN_DEFAULT_NUM_STEPS,
    DEFAULT_LAYER,
    DEFAULT_VARIANCE_THRESHOLD,
    GROM_BEHAVIOR_WEIGHT,
    GROM_INTENSITY_HIDDEN_DIM,
    GROM_OPTIMIZATION_STEPS,
    GROM_RETAIN_WEIGHT,
    GROM_ROUTER_HIDDEN_DIM,
    GROM_SPARSE_WEIGHT,
    MLP_HIDDEN_DIM,
    MLP_NUM_LAYERS,
    MLP_LEARNING_RATE,
    NURT_NUM_DIMS,
    NURT_NUM_INTEGRATION_STEPS,
    NURT_T_MAX,
    NURT_TRAINING_EPOCHS,
    SZLAK_INFERENCE_K,
    SZLAK_SINKHORN_REG,
    TECZA_NUM_DIRECTIONS,
    DEFAULT_OPTIMIZATION_STEPS,
    TECZA_RETAIN_WEIGHT,
    TETNO_CONDITION_THRESHOLD,
    TETNO_GATE_TEMPERATURE_LEGACY,
    TETNO_MAX_ALPHA,
    DEFAULT_STEERING_LAYERS,
    WICHER_CONCEPT_DIM,
)


# Steering application strategies
STEERING_STRATEGIES = ["constant", "initial_only", "diminishing", "increasing", "gaussian"]


@dataclass
class MethodConfig:
    """Base configuration for all methods."""
    method: str = ""
    extraction_strategy: str = "chat_last"
    steering_strategy: str = "constant"  # How steering is applied during generation


@dataclass
class CAAConfig(MethodConfig):
    """CAA-specific parameters."""
    layer: int = DEFAULT_LAYER
    
    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "CAA",
            "layer": self.layer,
            "steering_strategy": self.steering_strategy,
        }


@dataclass
class OstrzeConfig(MethodConfig):
    """Ostrze-specific parameters."""
    layer: int = DEFAULT_LAYER
    
    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "Ostrze",
            "layer": self.layer,
            "steering_strategy": self.steering_strategy,
        }


@dataclass
class MLPConfig(MethodConfig):
    """MLP-specific parameters."""
    layer: int = DEFAULT_LAYER
    hidden_dim: int = MLP_HIDDEN_DIM
    num_layers: int = MLP_NUM_LAYERS
    
    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "MLP",
            "layer": self.layer,
            "mlp_hidden_dim": self.hidden_dim,
            "mlp_num_layers": self.num_layers,
            "steering_strategy": self.steering_strategy,
        }


@dataclass
class TECZAConfig(MethodConfig):
    """TECZA-specific parameters."""
    layer: int = DEFAULT_LAYER
    num_directions: int = TECZA_NUM_DIRECTIONS
    direction_weighting: str = "primary_only"
    retain_weight: float = TECZA_RETAIN_WEIGHT
    optimization_steps: int = DEFAULT_OPTIMIZATION_STEPS
    
    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "TECZA",
            "layer": self.layer,
            "num_directions": self.num_directions,
            "direction_weighting": self.direction_weighting,
            "retain_weight": self.retain_weight,
            "tecza_optimization_steps": self.optimization_steps,
            "steering_strategy": self.steering_strategy,
        }


@dataclass
class TETNOConfig(MethodConfig):
    """TETNO-specific parameters."""
    sensor_layer: int = DEFAULT_LAYER
    steering_layers: List[int] = field(default_factory=lambda: list(DEFAULT_STEERING_LAYERS))
    condition_threshold: float = TETNO_CONDITION_THRESHOLD
    gate_temperature: float = TETNO_GATE_TEMPERATURE_LEGACY
    max_alpha: float = TETNO_MAX_ALPHA
    
    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "TETNO",
            "sensor_layer": self.sensor_layer,
            "steering_layers": ",".join(str(l) for l in self.steering_layers),
            "condition_threshold": self.condition_threshold,
            "gate_temperature": self.gate_temperature,
            "max_alpha": self.max_alpha,
            "steering_strategy": self.steering_strategy,
        }


@dataclass
class GROMConfig(MethodConfig):
    """GROM-specific parameters."""
    sensor_layer: int = DEFAULT_LAYER
    steering_layers: List[int] = field(default_factory=lambda: list(DEFAULT_STEERING_LAYERS))
    num_directions: int = TECZA_NUM_DIRECTIONS
    gate_hidden_dim: int = GROM_ROUTER_HIDDEN_DIM
    intensity_hidden_dim: int = GROM_INTENSITY_HIDDEN_DIM
    behavior_weight: float = GROM_BEHAVIOR_WEIGHT
    retain_weight: float = GROM_RETAIN_WEIGHT
    sparse_weight: float = GROM_SPARSE_WEIGHT
    max_alpha: float = TETNO_MAX_ALPHA
    optimization_steps: int = GROM_OPTIMIZATION_STEPS
    
    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "GROM",
            "sensor_layer": self.sensor_layer,
            "steering_layers": ",".join(str(l) for l in self.steering_layers),
            "num_directions": self.num_directions,
            "gate_hidden_dim": self.gate_hidden_dim,
            "intensity_hidden_dim": self.intensity_hidden_dim,
            "behavior_weight": self.behavior_weight,
            "retain_weight": self.retain_weight,
            "sparse_weight": self.sparse_weight,
            "max_alpha": self.max_alpha,
            "grom_optimization_steps": self.optimization_steps,
            "steering_strategy": self.steering_strategy,
        }


@dataclass
class NurtConfig(MethodConfig):
    """Nurt-specific parameters."""
    layer: int = DEFAULT_LAYER
    num_dims: int = NURT_NUM_DIMS
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD
    training_epochs: int = NURT_TRAINING_EPOCHS
    lr: float = MLP_LEARNING_RATE
    num_integration_steps: int = NURT_NUM_INTEGRATION_STEPS
    t_max: float = NURT_T_MAX
    
    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "nurt",
            "layer": self.layer,
            "steering_strategy": self.steering_strategy,
        }

@dataclass
class SzlakConfig(MethodConfig):
    """Szlak-specific parameters."""
    layer: int = DEFAULT_LAYER
    sinkhorn_reg: float = SZLAK_SINKHORN_REG
    inference_k: int = SZLAK_INFERENCE_K

    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "szlak",
            "layer": self.layer,
            "steering_strategy": self.steering_strategy,
            "szlak_sinkhorn_reg": self.sinkhorn_reg,
            "szlak_inference_k": self.inference_k,
        }


@dataclass
class WicherConfig(MethodConfig):
    """WICHER-specific parameters."""
    layer: int = DEFAULT_LAYER
    concept_dim: int = WICHER_CONCEPT_DIM
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD
    num_steps: int = BROYDEN_DEFAULT_NUM_STEPS
    alpha: float = BROYDEN_DEFAULT_ALPHA
    eta: float = BROYDEN_DEFAULT_ETA
    beta: float = BROYDEN_DEFAULT_BETA
    alpha_decay: float = BROYDEN_DEFAULT_ALPHA_DECAY

    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "wicher",
            "layer": self.layer,
            "steering_strategy": self.steering_strategy,
            "wicher_concept_dim": self.concept_dim,
            "wicher_variance_threshold": self.variance_threshold,
            "wicher_num_steps": self.num_steps,
            "wicher_alpha": self.alpha,
            "wicher_eta": self.eta,
            "wicher_beta": self.beta,
            "wicher_alpha_decay": self.alpha_decay,
        }
