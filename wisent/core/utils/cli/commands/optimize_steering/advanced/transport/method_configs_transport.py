"""Method configuration for PRZELOM (attention-transport) steering optimization."""
from dataclasses import dataclass
from typing import Dict, Any, Optional

from wisent.core.utils.cli.optimize_steering.method_configs import MethodConfig


@dataclass
class PrzelomConfig(MethodConfig):
    """PRZELOM attention-transport parameters for Optuna optimization."""
    layer: Optional[int] = None
    epsilon: Optional[float] = None
    target_mode: Optional[str] = None
    regularization: Optional[float] = None
    inference_k: Optional[int] = None

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
