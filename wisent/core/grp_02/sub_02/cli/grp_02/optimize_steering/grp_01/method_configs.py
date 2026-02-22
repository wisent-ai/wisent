"""Method configuration dataclasses for steering optimization."""
from dataclasses import dataclass, field
from typing import List, Dict, Any


# Steering application strategies
STEERING_STRATEGIES = ["constant", "initial_only", "diminishing", "increasing", "gaussian"]


class MethodConfig:
    """Base configuration for all methods."""
    method: str
    extraction_strategy: str = "chat_last"
    steering_strategy: str = "constant"  # How steering is applied during generation


@dataclass
class CAAConfig(MethodConfig):
    """CAA-specific parameters."""
    layer: int = 16
    
    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "CAA",
            "layer": self.layer,
            "steering_strategy": self.steering_strategy,
        }


@dataclass
class OstrzeConfig(MethodConfig):
    """Ostrze-specific parameters."""
    layer: int = 16
    
    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "Ostrze",
            "layer": self.layer,
            "steering_strategy": self.steering_strategy,
        }


@dataclass
class MLPConfig(MethodConfig):
    """MLP-specific parameters."""
    layer: int = 16
    hidden_dim: int = 256
    num_layers: int = 2
    
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
    layer: int = 16
    num_directions: int = 3
    direction_weighting: str = "primary_only"
    retain_weight: float = 0.1
    optimization_steps: int = 100
    
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
    sensor_layer: int = 16
    steering_layers: List[int] = field(default_factory=lambda: [20, 21, 22])
    condition_threshold: float = 0.5
    gate_temperature: float = 0.5
    max_alpha: float = 2.0
    
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
    sensor_layer: int = 16
    steering_layers: List[int] = field(default_factory=lambda: [20, 21, 22])
    num_directions: int = 3
    gate_hidden_dim: int = 64
    intensity_hidden_dim: int = 32
    behavior_weight: float = 1.0
    retain_weight: float = 0.2
    sparse_weight: float = 0.05
    max_alpha: float = 2.0
    optimization_steps: int = 200
    
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
    layer: int = 16
    num_dims: int = 0
    variance_threshold: float = 0.80
    training_epochs: int = 300
    lr: float = 0.001
    num_integration_steps: int = 4
    t_max: float = 1.0
    
    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "nurt",
            "layer": self.layer,
            "steering_strategy": self.steering_strategy,
        }

@dataclass
class SzlakConfig(MethodConfig):
    """Szlak-specific parameters."""
    layer: int = 16
    sinkhorn_reg: float = 0.1
    inference_k: int = 5

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
    layer: int = 16
    concept_dim: int = 0
    variance_threshold: float = 0.80
    num_steps: int = 3
    alpha: float = 5e-3
    eta: float = 0.5
    beta: float = 0.0
    alpha_decay: float = 1.0

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
