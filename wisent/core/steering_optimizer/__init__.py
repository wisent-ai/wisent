"""
Steering optimization package.

Provides comprehensive steering parameter optimization including:
- Automatic method selection via repscan geometry analysis
- Method comparison optimization
- Layer optimization
- Strength optimization
- Full pipeline optimization across all dimensions
"""

from __future__ import annotations

# Types and configuration
from .types import (
    SteeringApplicationStrategy,
    SteeringApplicationConfig,
    SteeringMethodConfig,
    SteeringOptimizationResult,
    SteeringOptimizationSummary,
    get_default_token_aggregation_strategies,
    get_default_prompt_construction_strategies,
    get_default_steering_application_configs,
    get_default_steering_configs,
)

# Main optimizer class
from .optimizer import SteeringOptimizer

# Auto optimization
from .auto import run_auto_steering_optimization

# CLI convenience functions
from .cli import (
    run_steering_optimization,
    get_optimal_steering_params,
)


__all__ = [
    # Types and enums
    "SteeringApplicationStrategy",
    "SteeringApplicationConfig",
    "SteeringMethodConfig",
    "SteeringOptimizationResult",
    "SteeringOptimizationSummary",
    # Default configuration functions
    "get_default_token_aggregation_strategies",
    "get_default_prompt_construction_strategies",
    "get_default_steering_application_configs",
    "get_default_steering_configs",
    # Main optimizer class
    "SteeringOptimizer",
    # Auto optimization
    "run_auto_steering_optimization",
    # CLI functions
    "run_steering_optimization",
    "get_optimal_steering_params",
]
