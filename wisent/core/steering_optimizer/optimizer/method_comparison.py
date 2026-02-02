"""
Method comparison optimization for SteeringOptimizer.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from wisent.core.steering_methods import SteeringMethodRegistry, SteeringMethodType

from ..types import (
    SteeringMethodConfig,
    SteeringOptimizationResult,
    SteeringOptimizationSummary,
    get_default_steering_configs,
)

logger = logging.getLogger(__name__)

# Alias for backward compatibility
SteeringMethod = SteeringMethodType


class MethodComparisonMixin:
    """Mixin providing method comparison optimization."""

    def optimize_steering_method_comparison(
        self,
        task_name: str,
        methods_to_test: Optional[Union[List[SteeringMethod], List[SteeringMethodConfig]]] = None,
        layer_range: Optional[str] = None,
        strength_range: Optional[List[float]] = None,
        limit: int = 100,
        max_time_minutes: float = 30.0,
        split_ratio: float = 0.8
    ) -> SteeringOptimizationSummary:
        """
        Compare different steering methods to find the best one for a task.

        Args:
            task_name: Task to optimize steering for
            methods_to_test: List of steering methods to compare
            layer_range: Range of layers to test for steering
            strength_range: Range of steering strengths to test
            limit: Maximum samples for testing
            max_time_minutes: Maximum optimization time
            split_ratio: Train/test split ratio

        Returns:
            SteeringOptimizationSummary with method comparison results
        """
        # Handle both old-style method list and new config list
        if methods_to_test is None:
            method_configs = get_default_steering_configs()
        else:
            method_configs = []
            for item in methods_to_test:
                if isinstance(item, SteeringMethodConfig):
                    method_configs.append(item)
                elif isinstance(item, SteeringMethod):
                    method_configs.append(SteeringMethodConfig(
                        name=item.value, method=item, params={}
                    ))
                elif isinstance(item, str):
                    try:
                        method = SteeringMethod(item)
                        method_configs.append(SteeringMethodConfig(
                            name=method.value, method=method, params={}
                        ))
                    except ValueError:
                        logger.warning(f"Unknown steering method: {item}")
                else:
                    logger.warning(f"Unknown method type: {type(item)}, value: {item}")

        if strength_range is None:
            strength_range = [0.5, 1.0, 1.5, 2.0]

        logger.info(f"Comparing {len(method_configs)} steering method configs for: {task_name}")

        start_time = time.time()
        task_results = []
        all_results = {}

        # Determine layer search range
        layers_to_test = self._get_layers_to_test(layer_range)

        configurations_tested = 0
        best_overall_score = 0.0
        best_overall_config = None

        # Test each method configuration
        for method_config in method_configs:
            method_results = []

            for layer in layers_to_test:
                for strength in strength_range:
                    if time.time() - start_time > max_time_minutes * 60:
                        logger.warning("Time limit reached, stopping optimization")
                        break

                    try:
                        score = self._evaluate_steering_configuration(
                            task_name=task_name,
                            method=method_config.method,
                            layer=layer,
                            strength=strength,
                            limit=limit,
                            split_ratio=split_ratio,
                            method_params=method_config.params
                        )

                        configurations_tested += 1
                        config_result = {
                            'method': method_config.name,
                            'method_type': method_config.method.value,
                            'layer': layer,
                            'strength': strength,
                            'score': score,
                            'params': method_config.params
                        }
                        method_results.append(config_result)

                        if score > best_overall_score:
                            best_overall_score = score
                            best_overall_config = config_result

                        if self.verbose:
                            logger.info(f"   {method_config.name} L{layer} S{strength}: {score:.3f}")

                    except Exception as e:
                        logger.error(f"   Error testing {method_config.name} L{layer} S{strength}: {e}")

            all_results[method_config.name] = method_results

        # Analyze results
        method_performance, layer_effectiveness = self._analyze_results(all_results)

        # Create optimization result
        optimization_time = time.time() - start_time

        if best_overall_config:
            result = SteeringOptimizationResult(
                task_name=task_name,
                best_steering_layer=best_overall_config['layer'],
                best_steering_method=best_overall_config['method'],
                best_steering_strength=best_overall_config['strength'],
                optimal_parameters={
                    'split_ratio': split_ratio,
                    'limit': limit,
                    **best_overall_config.get('params', {})
                },
                steering_effectiveness_score=best_overall_config['score'],
                classification_accuracy_impact=0.0,
                optimization_time_seconds=optimization_time,
                total_configurations_tested=configurations_tested
            )
            task_results.append(result)

        summary = SteeringOptimizationSummary(
            model_name=self.model_name,
            optimization_type="method_comparison",
            total_configurations_tested=configurations_tested,
            optimization_time_minutes=optimization_time / 60,
            best_overall_method=best_overall_config['method'] if best_overall_config else "none",
            best_overall_layer=best_overall_config['layer'] if best_overall_config else 0,
            best_overall_strength=best_overall_config['strength'] if best_overall_config else 0.0,
            method_performance_ranking=method_performance,
            layer_effectiveness_analysis=layer_effectiveness,
            task_results=task_results,
            optimization_date=datetime.now().isoformat()
        )

        self._save_steering_optimization_results(summary)
        return summary

    def _get_layers_to_test(self, layer_range: Optional[str]) -> List[int]:
        """Determine layers to test based on layer_range or model config."""
        if layer_range:
            return self._parse_layer_range(layer_range)
        elif self.base_classification_layer:
            min_layer = max(1, self.base_classification_layer - 2)
            max_layer = min(32, self.base_classification_layer + 2)
            return list(range(min_layer, max_layer + 1))
        else:
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
                num_layers = getattr(config, 'num_hidden_layers', None) or \
                             getattr(config, 'n_layer', None) or \
                             getattr(config, 'num_layers', None) or 32
                return list(range(num_layers))
            except Exception:
                return list(range(32))

    def _analyze_results(self, all_results: Dict) -> tuple:
        """Analyze optimization results to compute rankings."""
        method_performance = {}
        layer_effectiveness = {}

        for method, results in all_results.items():
            if results:
                scores = [r['score'] for r in results]
                method_performance[method] = max(scores)

                for result in results:
                    layer = result['layer']
                    if layer not in layer_effectiveness:
                        layer_effectiveness[layer] = []
                    layer_effectiveness[layer].append(result['score'])

        # Average layer effectiveness
        for layer in layer_effectiveness:
            layer_effectiveness[layer] = sum(layer_effectiveness[layer]) / len(layer_effectiveness[layer])

        return method_performance, layer_effectiveness
