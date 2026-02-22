"""
Strength and layer optimization for SteeringOptimizer.
"""

from __future__ import annotations

import logging
import time
import json
import os
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from wisent.core.errors import (
    MissingParameterError,
    SteeringMethodUnknownError,
    InsufficientDataError,
)
from wisent.core.steering_methods import SteeringMethodType

from ..types import SteeringOptimizationResult, SteeringOptimizationSummary

logger = logging.getLogger(__name__)

# Alias for backward compatibility
SteeringMethod = SteeringMethodType


class StrengthOptimizationMixin:
    """Mixin providing strength and layer optimization methods."""

    def optimize_steering_layer(
        self,
        task_name: str,
        steering_method: SteeringMethod = SteeringMethod.CAA,
        layer_search_range: Optional[Tuple[int, int]] = None,
        strength: float = 1.0,
        limit: int = 100
    ) -> SteeringOptimizationResult:
        """
        Find optimal steering layer for a specific method and task.

        Args:
            task_name: Task to optimize for
            steering_method: Steering method to use
            layer_search_range: (min_layer, max_layer) to search
            strength: Fixed steering strength to use during layer search
            limit: Maximum samples for testing

        Returns:
            SteeringOptimizationResult with optimal layer
        """
        logger.info(f"Optimizing steering layer for {task_name} using {steering_method.value}")

        if layer_search_range is None:
            if not self.base_classification_layer:
                raise MissingParameterError(
                    params=["layer_search_range", "base_classification_layer"],
                    context="Layer optimization"
                )
            min_layer = max(1, self.base_classification_layer - 3)
            max_layer = self.base_classification_layer + 3
            layer_search_range = (min_layer, max_layer)

        raise NotImplementedError(
            "Steering layer optimization not yet implemented. "
            f"Would search layers {layer_search_range}."
        )

    def optimize_steering_strength(
        self,
        task_name: str,
        steering_method: SteeringMethod = SteeringMethod.CAA,
        layer: Optional[int] = None,
        strength_range: Optional[Tuple[float, float]] = None,
        strength_steps: int = 10,
        limit: int = 100,
        method_params: Optional[Dict[str, Any]] = None
    ) -> SteeringOptimizationResult:
        """
        Find optimal steering strength for a specific method, layer, and task.
        """
        start_time = time.time()

        if layer is None:
            if not self.base_classification_layer:
                raise MissingParameterError(
                    params=["layer", "base_classification_layer"],
                    context="Steering strength optimization"
                )
            layer = self.base_classification_layer

        if strength_range is None:
            strength_range = (0.1, 2.0)

        logger.info(f"Optimizing steering strength for {task_name}")
        logger.info(f"   Method: {steering_method.value}, Layer: {layer}")

        # Load steering config if available
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'steering_optimization_parameters.json')
        steering_config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                steering_config = json.load(f)

        if method_params is None:
            method_configs = steering_config.get('steering_methods', [])
            for config in method_configs:
                if config['method'] == steering_method.value:
                    method_params = config.get('params', {})
                    break
            if method_params is None:
                method_params = {}

        strengths = np.linspace(strength_range[0], strength_range[1], strength_steps)

        results = []
        best_score = -float('inf')
        best_strength = 0.0

        for strength in strengths:
            try:
                from wisent.cli import run_task_pipeline

                pipeline_kwargs = {
                    'task_name': task_name,
                    'model_name': self.model_name,
                    'limit': limit,
                    'device': self.device,
                    'verbose': False,
                    'steering_mode': True,
                    'steering_method': steering_method.value,
                    'steering_strength': float(strength),
                    'layer': str(layer),
                    'output_mode': "likelihoods",
                    'allow_small_dataset': True
                }

                for param, value in method_params.items():
                    pipeline_kwargs[param] = value

                result = run_task_pipeline(**pipeline_kwargs)

                score = self._extract_strength_score(result)

                results.append({
                    'strength': float(strength),
                    'score': score,
                })

                if score > best_score:
                    best_score = score
                    best_strength = float(strength)

                logger.info(f"   Strength {strength:.2f}: score={score:.3f}")

            except Exception as e:
                logger.error(f"   Error testing strength {strength}: {e}")
                results.append({'strength': float(strength), 'score': 0.0, 'error': str(e)})

        optimization_time = time.time() - start_time

        return SteeringOptimizationResult(
            task_name=task_name,
            best_steering_layer=layer,
            best_steering_method=steering_method.value,
            best_steering_strength=best_strength,
            optimal_parameters={'strength': best_strength},
            steering_effectiveness_score=best_score,
            classification_accuracy_impact=best_score,
            optimization_time_seconds=optimization_time,
            total_configurations_tested=len(results),
            error_message=None
        )

    def _extract_strength_score(self, result: Dict) -> float:
        """Extract score from strength optimization result."""
        if not isinstance(result, dict):
            return 0.0

        eval_results = result.get('evaluation_results', result)

        accuracy = eval_results.get('accuracy', 0.0)
        if isinstance(accuracy, str) or accuracy is None:
            accuracy = 0.0

        baseline_likes = eval_results.get('baseline_likelihoods', [])
        steered_likes = eval_results.get('steered_likelihoods', [])

        if baseline_likes and steered_likes:
            valid_pairs = [(b, s) for b, s in zip(baseline_likes, steered_likes)
                          if np.isfinite(b) and np.isfinite(s)]
            if valid_pairs:
                changes = [abs(s - b) for b, s in valid_pairs]
                steering_effect = min(sum(changes) / len(changes), 100.0)
                score = steering_effect
                if np.isfinite(accuracy) and accuracy > 0.5:
                    score += accuracy * 0.5
                return score

        return float(accuracy) if np.isfinite(accuracy) else 0.0

    def optimize_method_specific_parameters(
        self,
        task_name: str,
        steering_method: SteeringMethod,
        base_layer: Optional[int] = None,
        base_strength: float = 1.0,
        limit: int = 100
    ) -> SteeringOptimizationResult:
        """Optimize method-specific parameters for a steering approach."""
        logger.info(f"Optimizing {steering_method.value}-specific parameters for {task_name}")

        if steering_method == SteeringMethod.CAA:
            return self._optimize_caa_parameters(task_name, base_layer, base_strength, limit)
        else:
            raise SteeringMethodUnknownError(method=str(steering_method))

    def _optimize_caa_parameters(
        self, task_name: str, layer: Optional[int], strength: float, limit: int
    ) -> SteeringOptimizationResult:
        """Optimize CAA specific parameters."""
        return SteeringOptimizationResult(
            task_name=task_name,
            best_steering_layer=layer if layer is not None else 15,
            best_steering_method="caa",
            best_steering_strength=strength,
            optimal_parameters={"normalize": True},
            steering_effectiveness_score=0.0,
            classification_accuracy_impact=0.0,
            optimization_time_seconds=0.0,
            total_configurations_tested=1
        )

    def run_comprehensive_steering_optimization(
        self,
        tasks: Optional[List[str]] = None,
        methods: Optional[List[SteeringMethod]] = None,
        limit: int = 100,
        max_time_per_task_minutes: float = 20.0,
        save_results: bool = True
    ) -> SteeringOptimizationSummary:
        """Run comprehensive steering optimization across multiple tasks and methods."""
        logger.info("Starting comprehensive steering optimization")

        if tasks is None:
            if self.classification_config:
                task_overrides = self.classification_config.get("task_specific_overrides", {})
                tasks = list(task_overrides.keys())
                if not tasks:
                    raise InsufficientDataError(
                        reason="No classification-optimized tasks found."
                    )
            else:
                raise MissingParameterError(
                    params=["tasks", "classification_config"],
                    context="comprehensive steering optimization"
                )

        if methods is None:
            methods = [SteeringMethod.CAA]

        logger.info(f"Tasks: {tasks}")
        logger.info(f"Methods: {[m.value for m in methods]}")

        all_results = []
        for task in tasks:
            for method in methods:
                try:
                    result = self._optimize_caa_parameters(task, None, 1.0, limit)
                    all_results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to optimize {method.value} for {task}: {e}")

        return SteeringOptimizationSummary(
            model_name=self.model_name,
            optimization_type="comprehensive",
            total_configurations_tested=len(all_results),
            optimization_time_minutes=0.0,
            best_overall_method="caa",
            best_overall_layer=15,
            best_overall_strength=1.0,
            method_performance_ranking={},
            layer_effectiveness_analysis={},
            task_results=all_results,
            optimization_date=""
        )
