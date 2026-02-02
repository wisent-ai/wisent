"""
Full pipeline optimization for SteeringOptimizer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Any

from wisent.core.activations import ExtractionStrategy
from wisent.core.steering_methods import SteeringMethodType

from ..types import (
    SteeringApplicationStrategy,
    SteeringApplicationConfig,
    SteeringOptimizationResult,
    SteeringOptimizationSummary,
    get_default_token_aggregation_strategies,
    get_default_prompt_construction_strategies,
    get_default_steering_application_configs,
)

logger = logging.getLogger(__name__)

# Alias for backward compatibility
SteeringMethod = SteeringMethodType


class FullPipelineMixin:
    """Mixin providing full pipeline optimization across all dimensions."""

    def optimize_full_steering_pipeline(
        self,
        task_name: str,
        methods_to_test: Optional[List[SteeringMethod]] = None,
        layer_range: Optional[str] = None,
        strength_range: Optional[List[float]] = None,
        token_aggregation_strategies: Optional[List[ExtractionStrategy]] = None,
        prompt_construction_strategies: Optional[List[ExtractionStrategy]] = None,
        steering_application_configs: Optional[List[SteeringApplicationConfig]] = None,
        limit: int = 100,
        max_time_minutes: float = 60.0,
        split_ratio: float = 0.8
    ) -> SteeringOptimizationSummary:
        """
        Full optimization across all steering dimensions.

        Args:
            task_name: Task to optimize steering for
            methods_to_test: Steering methods to test
            layer_range: Range of layers to test
            strength_range: Steering strengths to test
            token_aggregation_strategies: Token aggregation strategies to test
            prompt_construction_strategies: Prompt construction strategies to test
            steering_application_configs: Steering application configs to test
            limit: Maximum samples for testing
            max_time_minutes: Maximum optimization time
            split_ratio: Train/test split ratio

        Returns:
            SteeringOptimizationSummary with comprehensive results
        """
        # Set defaults
        if methods_to_test is None:
            methods_to_test = [SteeringMethod.CAA]
        if strength_range is None:
            strength_range = [0.5, 1.0, 1.5, 2.0]
        if token_aggregation_strategies is None:
            token_aggregation_strategies = get_default_token_aggregation_strategies()
        if prompt_construction_strategies is None:
            prompt_construction_strategies = get_default_prompt_construction_strategies()
        if steering_application_configs is None:
            steering_application_configs = get_default_steering_application_configs()

        # Determine layer range
        layers_to_test = self._get_layers_for_full_pipeline(layer_range)

        logger.info(f"Full steering optimization for task: {task_name}")
        logger.info(f"   Methods: {[m.value for m in methods_to_test]}")
        logger.info(f"   Layers: {layers_to_test}")

        start_time = time.time()
        all_results = []
        best_score = -float('inf')
        best_config = None
        configurations_tested = 0

        # Track rankings by dimension
        scores_by_dim = {
            'method': {}, 'layer': {}, 'strength': {},
            'token_agg': {}, 'prompt_const': {}, 'steering_app': {}
        }

        # Iterate through all combinations
        for method in methods_to_test:
            for layer in layers_to_test:
                for strength in strength_range:
                    for token_agg in token_aggregation_strategies:
                        for prompt_const in prompt_construction_strategies:
                            for steering_app in steering_application_configs:
                                if time.time() - start_time > max_time_minutes * 60:
                                    logger.warning("Time limit reached")
                                    break

                                try:
                                    score = self._evaluate_full_configuration(
                                        task_name, method, layer, strength,
                                        token_agg, prompt_const, steering_app,
                                        limit, split_ratio
                                    )

                                    configurations_tested += 1
                                    config = {
                                        'method': method.value,
                                        'layer': layer,
                                        'strength': strength,
                                        'token_aggregation': token_agg.value,
                                        'prompt_construction': prompt_const.value,
                                        'steering_application': steering_app.strategy.value,
                                        'steering_application_params': asdict(steering_app),
                                        'score': score
                                    }
                                    all_results.append(config)

                                    if score > best_score:
                                        best_score = score
                                        best_config = config

                                    # Track scores
                                    self._track_score(scores_by_dim, config, score)

                                except Exception as e:
                                    logger.error(f"   Error: {e}")

        # Compute rankings
        rankings = self._compute_rankings(scores_by_dim)
        optimization_time = time.time() - start_time

        # Create result and summary
        task_result = self._create_task_result(
            task_name, best_config, optimization_time,
            configurations_tested, split_ratio, limit
        )

        summary = self._create_summary(
            best_config, optimization_time, configurations_tested,
            rankings, task_result
        )

        self._save_steering_optimization_results(summary)
        logger.info(f"Full optimization complete! Best score: {best_score:.3f}")

        return summary

    def _get_layers_for_full_pipeline(self, layer_range: Optional[str]) -> List[int]:
        """Get layers to test for full pipeline optimization."""
        if layer_range:
            return self._parse_layer_range(layer_range)
        elif self.base_classification_layer:
            min_layer = max(1, self.base_classification_layer - 2)
            max_layer = min(32, self.base_classification_layer + 2)
            return list(range(min_layer, max_layer + 1))
        else:
            return [4, 6, 8, 10, 12]

    def _evaluate_full_configuration(
        self, task_name: str, method: SteeringMethod, layer: int, strength: float,
        token_aggregation: ExtractionStrategy, prompt_construction: ExtractionStrategy,
        steering_application: SteeringApplicationConfig, limit: int, split_ratio: float
    ) -> float:
        """Evaluate a full steering configuration with all parameters."""
        try:
            from wisent.cli import run_task_pipeline

            kwargs = {
                'task_name': task_name,
                'model_name': self.model_name,
                'layer': str(layer),
                'limit': limit,
                'steering_mode': True,
                'steering_method': method.value,
                'steering_strength': strength,
                'split_ratio': split_ratio,
                'device': self.device,
                'verbose': False,
                'allow_small_dataset': True,
                'token_aggregation': token_aggregation.value,
                'prompt_construction_strategy': prompt_construction.value,
                'steering_application_strategy': steering_application.strategy.value,
            }

            # Add steering application specific params
            if steering_application.strategy == SteeringApplicationStrategy.INITIAL_ONLY:
                kwargs['steering_initial_tokens'] = steering_application.initial_tokens
            elif steering_application.strategy in (SteeringApplicationStrategy.DIMINISHING,
                                                    SteeringApplicationStrategy.INCREASING):
                kwargs['steering_decay_rate'] = steering_application.rate

            result = run_task_pipeline(**kwargs)

            if isinstance(result, dict):
                if 'accuracy' in result and result['accuracy'] != 'N/A':
                    return float(result['accuracy'])
                elif 'evaluation_results' in result:
                    eval_results = result['evaluation_results']
                    if 'accuracy' in eval_results and eval_results['accuracy'] != 'N/A':
                        return float(eval_results['accuracy'])
            return 0.0

        except Exception as e:
            logger.error(f"Configuration evaluation failed: {e}")
            return 0.0

    def _track_score(self, scores_by_dim: Dict, config: Dict, score: float):
        """Track score by each dimension."""
        scores_by_dim['method'].setdefault(config['method'], []).append(score)
        scores_by_dim['layer'].setdefault(config['layer'], []).append(score)
        scores_by_dim['strength'].setdefault(config['strength'], []).append(score)
        scores_by_dim['token_agg'].setdefault(config['token_aggregation'], []).append(score)
        scores_by_dim['prompt_const'].setdefault(config['prompt_construction'], []).append(score)
        scores_by_dim['steering_app'].setdefault(config['steering_application'], []).append(score)

    def _compute_rankings(self, scores_by_dim: Dict) -> Dict:
        """Compute average rankings for each dimension."""
        return {
            key: {k: sum(v)/len(v) for k, v in dim_scores.items()}
            for key, dim_scores in scores_by_dim.items()
        }

    def _create_task_result(
        self, task_name: str, best_config: Optional[Dict],
        optimization_time: float, configurations_tested: int,
        split_ratio: float, limit: int
    ) -> Optional[SteeringOptimizationResult]:
        """Create task result from best config."""
        if not best_config:
            return None
        return SteeringOptimizationResult(
            task_name=task_name,
            best_steering_layer=best_config['layer'],
            best_steering_method=best_config['method'],
            best_steering_strength=best_config['strength'],
            optimal_parameters={'split_ratio': split_ratio, 'limit': limit},
            steering_effectiveness_score=best_config['score'],
            classification_accuracy_impact=0.0,
            optimization_time_seconds=optimization_time,
            total_configurations_tested=configurations_tested,
            best_token_aggregation=best_config['token_aggregation'],
            best_prompt_construction=best_config['prompt_construction'],
            best_steering_application=best_config['steering_application'],
            steering_application_params=best_config['steering_application_params']
        )

    def _create_summary(
        self, best_config: Optional[Dict], optimization_time: float,
        configurations_tested: int, rankings: Dict,
        task_result: Optional[SteeringOptimizationResult]
    ) -> SteeringOptimizationSummary:
        """Create optimization summary."""
        return SteeringOptimizationSummary(
            model_name=self.model_name,
            optimization_type="comprehensive",
            total_configurations_tested=configurations_tested,
            optimization_time_minutes=optimization_time / 60,
            best_overall_method=best_config['method'] if best_config else "none",
            best_overall_layer=best_config['layer'] if best_config else 0,
            best_overall_strength=best_config['strength'] if best_config else 0.0,
            method_performance_ranking=rankings['method'],
            layer_effectiveness_analysis=rankings['layer'],
            task_results=[task_result] if task_result else [],
            optimization_date=datetime.now().isoformat(),
            best_token_aggregation=best_config['token_aggregation'] if best_config else None,
            best_prompt_construction=best_config['prompt_construction'] if best_config else None,
            best_steering_application=best_config['steering_application'] if best_config else None,
            token_aggregation_ranking=rankings['token_agg'],
            prompt_construction_ranking=rankings['prompt_const'],
            steering_application_ranking=rankings['steering_app']
        )
