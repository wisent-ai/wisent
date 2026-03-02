"""
Steering optimization module for improving benchmark performance.

This module handles training and optimizing different steering methods that can
improve model performance on benchmarks by steering internal activations.
"""

import logging
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from wisent.core.optuna.classifier import CacheConfig
from wisent.core.steering_methods import SteeringMethodRegistry

logger = logging.getLogger(__name__)


@dataclass
class SteeringMethodConfig(ABC):
    """Base configuration for steering methods - uses centralized registry."""

    method_name: str = "base"
    layers: List[int] = None
    strengths: List[float] = None

    def __post_init__(self):
        if self.layers is None:
            self.layers = []
        if self.strengths is None:
            self.strengths = [1.0]

    @classmethod
    def from_registry(cls, method_name: str, layers: List[int] = None, strengths: List[float] = None):
        """Create config from centralized registry."""
        definition = SteeringMethodRegistry.get(method_name)
        return cls(
            method_name=method_name,
            layers=layers or [],
            strengths=strengths or [definition.default_strength],
        )


@dataclass
class SteeringResult:
    """Results from training and evaluating a steering method configuration."""

    method_name: str
    layer: int
    hyperparameters: Dict[str, Any]
    benchmark_metrics: Dict[str, float]
    training_success: bool
    training_stats: Dict[str, Any] = None
    baseline_metrics: Dict[str, float] = None
    comparative_metrics: Dict[str, Any] = None


# Import split parts - SteeringMethodTrainer ABC and SteeringTrainer live in _part2
from wisent.core.optuna.steering._steering_opt_parts._part2 import (  # noqa: E402
    SteeringMethodTrainer,
    SteeringTrainer,
)
from wisent.core.optuna.steering._steering_opt_parts._part3 import (  # noqa: E402
    _SteeringOptimizerCore,
)
from wisent.core.optuna.steering._steering_opt_parts._part4 import (  # noqa: E402
    _SteeringOptimizerEval,
)
from wisent.core.optuna.steering._steering_opt_parts._part5 import (  # noqa: E402
    _SteeringOptimizerClassifier,
)


class SteeringOptimizer(_SteeringOptimizerCore, _SteeringOptimizerEval, _SteeringOptimizerClassifier):
    """
    Optimizes steering methods for improving benchmark performance.

    The steering optimization process:
    1. Train steering methods on training data
    2. Evaluate steering performance on validation data using benchmark metrics
    3. Select best configuration based on benchmark performance
    4. Test final steering method on test data
    """

    def __init__(self, cache_config: Optional[CacheConfig] = None):
        _SteeringOptimizerCore.__init__(self, cache_config=cache_config)


__all__ = [
    "SteeringMethodConfig",
    "SteeringResult",
    "SteeringMethodTrainer",
    "SteeringTrainer",
    "SteeringOptimizer",
]
