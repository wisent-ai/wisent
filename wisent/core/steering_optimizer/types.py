"""
Type definitions for steering optimization.

Contains enums, dataclasses, and default configuration functions for steering optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from wisent.core.activations import ExtractionStrategy


class SteeringApplicationStrategy(Enum):
    """How steering is applied during generation."""
    CONSTANT = "constant"           # Same strength throughout generation
    INITIAL_ONLY = "initial_only"   # Only apply at first N tokens
    DIMINISHING = "diminishing"     # Strength decreases over tokens
    INCREASING = "increasing"       # Strength increases over tokens
    GAUSSIAN = "gaussian"           # Gaussian curve centered at specific token position


@dataclass
class SteeringApplicationConfig:
    """Configuration for how steering is applied during generation."""
    strategy: SteeringApplicationStrategy = SteeringApplicationStrategy.CONSTANT
    # For INITIAL_ONLY: number of tokens to apply steering
    initial_tokens: int = 10
    # For DIMINISHING/INCREASING: decay/growth rate
    rate: float = 0.1
    # For GAUSSIAN: center position (as fraction of sequence)
    gaussian_center: float = 0.5
    gaussian_width: float = 0.2


@dataclass
class SteeringMethodConfig:
    """Configuration for a specific steering method with parameter variations."""
    name: str  # Display name like "CAA_L2"
    method: Any  # SteeringMethodType enum
    params: Dict[str, Any]  # Method-specific parameters

    def __post_init__(self):
        """Ensure method is SteeringMethodType enum."""
        from wisent.core.steering_methods import SteeringMethodType
        if isinstance(self.method, str):
            self.method = SteeringMethodType(self.method.lower())


@dataclass
class SteeringOptimizationResult:
    """Results from optimizing steering parameters for a single task."""
    task_name: str
    best_steering_layer: int
    best_steering_method: str
    best_steering_strength: float
    optimal_parameters: Dict[str, Any]  # Method-specific parameters
    steering_effectiveness_score: float  # How well steering changes outputs
    classification_accuracy_impact: float  # Impact on classification performance
    optimization_time_seconds: float
    total_configurations_tested: int
    # New fields for extended optimization
    best_token_aggregation: Optional[str] = None
    best_prompt_construction: Optional[str] = None
    best_steering_application: Optional[str] = None
    steering_application_params: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class SteeringOptimizationSummary:
    """Summary of steering optimization across tasks/methods."""
    model_name: str
    optimization_type: str  # "single_task", "multi_task", "method_comparison", "comprehensive"
    total_configurations_tested: int
    optimization_time_minutes: float
    best_overall_method: str
    best_overall_layer: int
    best_overall_strength: float
    method_performance_ranking: Dict[str, float]  # method -> effectiveness score
    layer_effectiveness_analysis: Dict[int, float]  # layer -> avg effectiveness
    task_results: List[SteeringOptimizationResult]
    optimization_date: str
    # New fields for extended optimization
    best_token_aggregation: Optional[str] = None
    best_prompt_construction: Optional[str] = None
    best_steering_application: Optional[str] = None
    token_aggregation_ranking: Optional[Dict[str, float]] = None
    prompt_construction_ranking: Optional[Dict[str, float]] = None
    steering_application_ranking: Optional[Dict[str, float]] = None


def get_default_token_aggregation_strategies() -> List[ExtractionStrategy]:
    """Get token aggregation strategies to test."""
    return [
        ExtractionStrategy.CHAT_LAST,
        ExtractionStrategy.CHAT_MEAN,
        ExtractionStrategy.CHAT_FIRST,
        ExtractionStrategy.CHAT_MAX_NORM,
    ]


def get_default_prompt_construction_strategies() -> List[ExtractionStrategy]:
    """Get prompt construction strategies to test."""
    return [
        ExtractionStrategy.CHAT_LAST,
        ExtractionStrategy.CHAT_LAST,
        ExtractionStrategy.CHAT_LAST,
    ]


def get_default_steering_application_configs() -> List[SteeringApplicationConfig]:
    """Get steering application configurations to test."""
    return [
        SteeringApplicationConfig(strategy=SteeringApplicationStrategy.CONSTANT),
        SteeringApplicationConfig(strategy=SteeringApplicationStrategy.INITIAL_ONLY, initial_tokens=5),
        SteeringApplicationConfig(strategy=SteeringApplicationStrategy.INITIAL_ONLY, initial_tokens=20),
        SteeringApplicationConfig(strategy=SteeringApplicationStrategy.DIMINISHING, rate=0.05),
        SteeringApplicationConfig(strategy=SteeringApplicationStrategy.DIMINISHING, rate=0.1),
    ]


def get_default_steering_configs() -> List[SteeringMethodConfig]:
    """Get default steering method configurations from centralized registry."""
    from wisent.core.steering_methods import SteeringMethodRegistry

    configs = []
    for method_name in SteeringMethodRegistry.list_methods():
        definition = SteeringMethodRegistry.get(method_name)
        method_type = definition.method_type

        # Base config with default params
        configs.append(SteeringMethodConfig(
            name=method_name.upper(),
            method=method_type,
            params=definition.get_default_params()
        ))

        # Add L2 normalized variant
        configs.append(SteeringMethodConfig(
            name=f"{method_name.upper()}_L2",
            method=method_type,
            params={**definition.get_default_params(), "normalization_method": "l2_unit"}
        ))

    return configs
