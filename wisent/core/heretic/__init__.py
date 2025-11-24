"""
Heretic-inspired features for Wisent.

This module implements features from the Heretic abliteration tool,
adapted for Wisent's activation steering paradigm:

1. Multi-objective optimization (KL divergence + task performance)
2. Refusal detection and counting
3. Direction interpolation between layers
4. Geometry analysis for steering vectors
5. Per-component steering parameters
"""

from wisent.core.heretic.refusal_detector import RefusalDetector
from wisent.core.heretic.kl_divergence import compute_kl_divergence, KLDivergenceEvaluator
from wisent.core.heretic.direction_interpolation import (
    interpolate_steering_vectors,
    interpolate_steering_vector,
    get_global_steering_direction,
)
from wisent.core.heretic.geometry_analyzer import (
    GeometryAnalyzer,
    GeometryMetrics,
    analyze_steering_geometry,
)
from wisent.core.heretic.multi_objective_optimizer import (
    MultiObjectiveOptimizer,
    SteeringParameters,
    OptimizationResult,
    quick_optimize_steering,
)

__all__ = [
    "RefusalDetector",
    "compute_kl_divergence",
    "KLDivergenceEvaluator",
    "interpolate_steering_vectors",
    "interpolate_steering_vector",
    "get_global_steering_direction",
    "GeometryAnalyzer",
    "GeometryMetrics",
    "analyze_steering_geometry",
    "MultiObjectiveOptimizer",
    "SteeringParameters",
    "OptimizationResult",
    "quick_optimize_steering",
]
