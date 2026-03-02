"""Method configuration for PRZELOM (attention-transport) steering optimization."""
from dataclasses import dataclass
from typing import Dict, Any, Optional

from wisent.core.utils.cli.optimize_steering.method_configs import MethodConfig
from wisent.core.utils.config_tools.constants import PARSER_DEFAULT_LAYER_START, PRZELOM_EPSILON, SZLAK_INFERENCE_K, TIKHONOV_REG


@dataclass
class PrzelomConfig(MethodConfig):
    """PRZELOM attention-transport parameters for Optuna optimization."""
    layer: int = PARSER_DEFAULT_LAYER_START
    epsilon: float = PRZELOM_EPSILON
    target_mode: Optional[str] = None
    regularization: float = TIKHONOV_REG
    inference_k: int = SZLAK_INFERENCE_K

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
