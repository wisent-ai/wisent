"""
CLI convenience functions for steering optimization.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

from .optimizer import SteeringOptimizer
from .auto import run_auto_steering_optimization
from .types import SteeringOptimizationSummary

logger = logging.getLogger(__name__)


def run_steering_optimization(
    model_name: str,
    task_name: str,
    optimization_type: str = "auto",
    methods_to_test: Optional[List[str]] = None,
    layer_range: Optional[str] = None,
    strength_range: Optional[List[float]] = None,
    limit: int = 100,
    device: str = None,
    verbose: bool = False,
    max_time_minutes: float = 60.0
) -> Dict[str, Any]:
    """
    Run steering optimization with the specified type.

    Args:
        model_name: Model to optimize steering for
        task_name: Task to optimize steering for
        optimization_type: Type of optimization to run:
            - "auto": Use repscan to select method and optimize
            - "method_comparison": Compare different steering methods
            - "layer": Optimize steering layer
            - "strength": Optimize steering strength
            - "comprehensive": Full pipeline optimization
        methods_to_test: Steering methods to test (for method_comparison)
        layer_range: Layer range to test (e.g., "10-20" or "10,12,14")
        strength_range: Strength values to test
        limit: Maximum samples for testing
        device: Device to use
        verbose: Enable verbose logging
        max_time_minutes: Maximum optimization time

    Returns:
        Dictionary with optimization results
    """
    if optimization_type == "auto":
        return run_auto_steering_optimization(
            model_name=model_name,
            task_name=task_name,
            limit=limit,
            device=device,
            verbose=verbose,
            max_time_minutes=max_time_minutes,
            methods_to_test=methods_to_test,
            strength_range=strength_range,
            layer_range=layer_range,
        )

    optimizer = SteeringOptimizer(model_name=model_name, device=device, verbose=verbose)

    if optimization_type == "method_comparison":
        summary = optimizer.optimize_steering_method_comparison(
            task_name=task_name,
            methods_to_test=methods_to_test,
            layer_range=layer_range,
            strength_range=strength_range,
            limit=limit,
            max_time_minutes=max_time_minutes,
        )
        return _summary_to_dict(summary)

    elif optimization_type == "layer":
        from wisent.core.steering_methods import SteeringMethodType
        method = SteeringMethodType.CAA
        if methods_to_test and len(methods_to_test) > 0:
            method = SteeringMethodType(methods_to_test[0].lower())

        result = optimizer.optimize_steering_layer(
            task_name=task_name,
            steering_method=method,
            layer_search_range=_parse_layer_tuple(layer_range) if layer_range else None,
            strength=strength_range[0] if strength_range else 1.0,
            limit=limit,
        )
        return {
            'task_name': result.task_name,
            'best_layer': result.best_steering_layer,
            'best_method': result.best_steering_method,
            'best_strength': result.best_steering_strength,
            'score': result.steering_effectiveness_score,
        }

    elif optimization_type == "strength":
        from wisent.core.steering_methods import SteeringMethodType
        method = SteeringMethodType.CAA
        if methods_to_test and len(methods_to_test) > 0:
            method = SteeringMethodType(methods_to_test[0].lower())

        layer = None
        if layer_range:
            layers = _parse_layer_range(layer_range)
            layer = layers[0] if layers else None

        result = optimizer.optimize_steering_strength(
            task_name=task_name,
            steering_method=method,
            layer=layer,
            strength_range=_parse_strength_tuple(strength_range) if strength_range else None,
            limit=limit,
        )
        return {
            'task_name': result.task_name,
            'best_layer': result.best_steering_layer,
            'best_method': result.best_steering_method,
            'best_strength': result.best_steering_strength,
            'score': result.steering_effectiveness_score,
        }

    elif optimization_type == "comprehensive":
        summary = optimizer.optimize_full_steering_pipeline(
            task_name=task_name,
            methods_to_test=methods_to_test,
            layer_range=layer_range,
            strength_range=strength_range,
            limit=limit,
            max_time_minutes=max_time_minutes,
        )
        return _summary_to_dict(summary)

    else:
        return {"error": f"Unknown optimization type: {optimization_type}"}


def get_optimal_steering_params(
    model_name: str,
    task_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Load optimal steering configuration for a model/task."""
    optimizer = SteeringOptimizer(model_name=model_name)
    return optimizer.load_optimal_steering_config(task_name=task_name)


def _generate_pairs_for_repscan(task_name: str, limit: int = 100) -> List:
    """Generate contrastive pairs for repscan analysis."""
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import build_contrastive_pairs
    return build_contrastive_pairs(task_name=task_name, limit=limit)


def _summary_to_dict(summary: SteeringOptimizationSummary) -> Dict[str, Any]:
    """Convert SteeringOptimizationSummary to dictionary."""
    return {
        'model_name': summary.model_name,
        'optimization_type': summary.optimization_type,
        'best_method': summary.best_overall_method,
        'best_layer': summary.best_overall_layer,
        'best_strength': summary.best_overall_strength,
        'method_ranking': summary.method_performance_ranking,
        'layer_effectiveness': summary.layer_effectiveness_analysis,
        'total_configurations_tested': summary.total_configurations_tested,
        'optimization_time_minutes': summary.optimization_time_minutes,
        'optimization_date': summary.optimization_date,
    }


def _parse_layer_range(layer_range: str) -> List[int]:
    """Parse layer range string."""
    if '-' in layer_range:
        start, end = map(int, layer_range.split('-'))
        return list(range(start, end + 1))
    elif ',' in layer_range:
        return [int(x.strip()) for x in layer_range.split(',')]
    return [int(layer_range)]


def _parse_layer_tuple(layer_range: str) -> tuple:
    """Parse layer range to tuple for layer optimization."""
    layers = _parse_layer_range(layer_range)
    return (min(layers), max(layers))


def _parse_strength_tuple(strength_range: List[float]) -> tuple:
    """Parse strength range to tuple."""
    if strength_range:
        return (min(strength_range), max(strength_range))
    return (0.1, 2.0)
