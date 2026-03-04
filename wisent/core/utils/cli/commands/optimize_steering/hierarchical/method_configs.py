"""Method configuration dataclasses for steering optimization."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from typing import Optional as _Optional


# Steering application strategies
STEERING_STRATEGIES = ["constant", "initial_only", "diminishing", "increasing", "gaussian"]


@dataclass
class MethodConfig:
    """Base configuration for all methods."""
    method: str = ""
    extraction_strategy: Optional[str] = None
    steering_strategy: Optional[str] = None


@dataclass
class CAAConfig(MethodConfig):
    """CAA-specific parameters."""
    layer: _Optional[int] = None

    def to_args(self) -> Dict[str, Any]:
        return {"method": "CAA", "layer": self.layer, "steering_strategy": self.steering_strategy}


@dataclass
class OstrzeConfig(MethodConfig):
    """Ostrze-specific parameters."""
    layer: _Optional[int] = None

    def to_args(self) -> Dict[str, Any]:
        return {"method": "Ostrze", "layer": self.layer, "steering_strategy": self.steering_strategy}


@dataclass
class MLPConfig(MethodConfig):
    """MLP-specific parameters."""
    layer: _Optional[int] = None
    hidden_dim: _Optional[int] = None
    num_layers: _Optional[int] = None
    mlp_input_divisor: _Optional[int] = None
    mlp_early_stopping_patience: _Optional[int] = None
    mlp_gating_hidden_dim_divisor: _Optional[int] = None

    def to_args(self) -> Dict[str, Any]:
        return {"method": "MLP", "layer": self.layer, "mlp_hidden_dim": self.hidden_dim,
                "mlp_num_layers": self.num_layers, "steering_strategy": self.steering_strategy,
                "mlp_input_divisor": self.mlp_input_divisor,
                "mlp_early_stopping_patience": self.mlp_early_stopping_patience,
                "mlp_gating_hidden_dim_divisor": self.mlp_gating_hidden_dim_divisor}


@dataclass
class TECZAConfig(MethodConfig):
    """TECZA-specific parameters."""
    layer: _Optional[int] = None
    num_directions: _Optional[int] = None
    direction_weighting: _Optional[str] = None
    retain_weight: _Optional[float] = None
    optimization_steps: _Optional[int] = None

    def to_args(self) -> Dict[str, Any]:
        return {"method": "TECZA", "layer": self.layer, "num_directions": self.num_directions,
                "direction_weighting": self.direction_weighting, "retain_weight": self.retain_weight,
                "tecza_optimization_steps": self.optimization_steps, "steering_strategy": self.steering_strategy}


@dataclass
class TETNOConfig(MethodConfig):
    """TETNO-specific parameters."""
    sensor_layer: _Optional[int] = None
    steering_layers: List[int] = field(default_factory=list)
    condition_threshold: _Optional[float] = None
    gate_temperature: _Optional[float] = None
    max_alpha: _Optional[float] = None

    def to_args(self) -> Dict[str, Any]:
        return {"method": "TETNO", "sensor_layer": self.sensor_layer,
                "steering_layers": ",".join(str(l) for l in self.steering_layers),
                "condition_threshold": self.condition_threshold, "gate_temperature": self.gate_temperature,
                "max_alpha": self.max_alpha, "steering_strategy": self.steering_strategy}


@dataclass
class GROMConfig(MethodConfig):
    """GROM-specific parameters."""
    sensor_layer: _Optional[int] = None
    steering_layers: List[int] = field(default_factory=list)
    num_directions: _Optional[int] = None
    gate_hidden_dim: _Optional[int] = None
    intensity_hidden_dim: _Optional[int] = None
    behavior_weight: _Optional[float] = None
    retain_weight: _Optional[float] = None
    sparse_weight: _Optional[float] = None
    max_alpha: _Optional[float] = None
    optimization_steps: _Optional[int] = None

    def to_args(self) -> Dict[str, Any]:
        return {"method": "GROM", "sensor_layer": self.sensor_layer,
                "steering_layers": ",".join(str(l) for l in self.steering_layers),
                "num_directions": self.num_directions, "gate_hidden_dim": self.gate_hidden_dim,
                "intensity_hidden_dim": self.intensity_hidden_dim, "behavior_weight": self.behavior_weight,
                "retain_weight": self.retain_weight, "sparse_weight": self.sparse_weight,
                "max_alpha": self.max_alpha, "grom_optimization_steps": self.optimization_steps,
                "steering_strategy": self.steering_strategy}


@dataclass
class NurtConfig(MethodConfig):
    """Nurt-specific parameters."""
    layer: _Optional[int] = None
    num_dims: _Optional[int] = None
    variance_threshold: _Optional[float] = None
    training_epochs: _Optional[int] = None
    lr: _Optional[float] = None
    num_integration_steps: _Optional[int] = None
    t_max: _Optional[float] = None

    def to_args(self) -> Dict[str, Any]:
        return {"method": "nurt", "layer": self.layer, "steering_strategy": self.steering_strategy}


@dataclass
class SzlakConfig(MethodConfig):
    """Szlak-specific parameters."""
    layer: _Optional[int] = None
    sinkhorn_reg: _Optional[float] = None
    inference_k: _Optional[int] = None

    def to_args(self) -> Dict[str, Any]:
        return {"method": "szlak", "layer": self.layer, "steering_strategy": self.steering_strategy,
                "szlak_sinkhorn_reg": self.sinkhorn_reg, "szlak_inference_k": self.inference_k}


@dataclass
class WicherConfig(MethodConfig):
    """WICHER-specific parameters."""
    layer: _Optional[int] = None
    concept_dim: _Optional[int] = None
    variance_threshold: _Optional[float] = None
    num_steps: _Optional[int] = None
    alpha: _Optional[float] = None
    eta: _Optional[float] = None
    beta: _Optional[float] = None
    alpha_decay: _Optional[float] = None

    def to_args(self) -> Dict[str, Any]:
        return {"method": "wicher", "layer": self.layer, "steering_strategy": self.steering_strategy,
                "wicher_concept_dim": self.concept_dim, "wicher_variance_threshold": self.variance_threshold,
                "wicher_num_steps": self.num_steps, "wicher_alpha": self.alpha, "wicher_eta": self.eta,
                "wicher_beta": self.beta, "wicher_alpha_decay": self.alpha_decay}
