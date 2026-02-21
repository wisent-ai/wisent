"""Method configuration for PRZELOM (attention-transport) steering optimization."""
from dataclasses import dataclass
from typing import Dict, Any

from wisent.core.cli.optimize_steering.method_configs import MethodConfig


@dataclass
class PrzelomConfig(MethodConfig):
    """PRZELOM attention-transport parameters for Optuna optimization."""
    layer: int = 16
    epsilon: float = 1.0
    target_mode: str = "uniform"
    regularization: float = 1e-4
    inference_k: int = 5

    def to_args(self) -> Dict[str, Any]:
        return {
            "method": "przelom",
            "layer": self.layer,
            "przelom_epsilon": self.epsilon,
            "przelom_target_mode": self.target_mode,
            "przelom_regularization": self.regularization,
            "przelom_inference_k": self.inference_k,
            "steering_strategy": self.steering_strategy,
        }
