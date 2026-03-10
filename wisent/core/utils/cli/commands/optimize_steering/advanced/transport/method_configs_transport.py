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
        base = {
            "method": "przelom",
            "layer": self.layer,
            "steering_strategy": self.steering_strategy,
        }
        base.update(self._extra_args)
        return base
