"""
Steering Optimizer class assembled from component modules.

This package provides the SteeringOptimizer class for optimizing steering parameters.
"""

from __future__ import annotations

from .base import SteeringOptimizerBase
from .method_comparison import MethodComparisonMixin
from .full_pipeline import FullPipelineMixin
from .strength import StrengthOptimizationMixin


class SteeringOptimizer(
    MethodComparisonMixin,
    FullPipelineMixin,
    StrengthOptimizationMixin,
    SteeringOptimizerBase
):
    """
    Framework for optimizing steering parameters.

    This class provides methods for:
    - Comparing different steering methods
    - Optimizing steering layer selection
    - Optimizing steering strength
    - Full pipeline optimization across all dimensions
    - Comprehensive multi-task optimization

    Assembled from component mixins for modularity.
    """
    pass


__all__ = ["SteeringOptimizer"]
