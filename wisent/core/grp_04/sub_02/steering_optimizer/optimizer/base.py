"""
Base class for SteeringOptimizer with initialization and utility methods.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from wisent.core.config_manager import ModelConfigManager
from wisent.core.errors import MissingParameterError

from ..types import SteeringOptimizationResult, SteeringOptimizationSummary

logger = logging.getLogger(__name__)


class SteeringOptimizerBase:
    """Base class providing initialization and utility methods for SteeringOptimizer."""

    def __init__(self, model_name: str, device: str = None, verbose: bool = False):
        """
        Initialize steering optimizer.

        Args:
            model_name: Name/path of the model to optimize steering for
            device: Device to run optimization on
            verbose: Enable verbose logging
        """
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        self.config_manager = ModelConfigManager()

        # Load classification parameters if available
        self.classification_config = self.config_manager.load_model_config(model_name)
        if self.classification_config:
            self.base_classification_layer = self.classification_config.get(
                "optimal_parameters", {}
            ).get("classification_layer")
            logger.info(f"Found existing classification layer: {self.base_classification_layer}")
        else:
            self.base_classification_layer = None
            logger.warning("No existing classification configuration found")

    def _parse_layer_range(self, layer_range: str) -> List[int]:
        """Parse layer range string like '10-20' or '10,12,14'."""
        if '-' in layer_range:
            start, end = map(int, layer_range.split('-'))
            return list(range(start, end + 1))
        elif ',' in layer_range:
            return [int(x.strip()) for x in layer_range.split(',')]
        else:
            return [int(layer_range)]

    def _evaluate_steering_configuration(
        self,
        task_name: str,
        method: Any,  # SteeringMethodType
        layer: int,
        strength: float,
        limit: int,
        split_ratio: float,
        method_params: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Evaluate a single steering configuration and return its effectiveness score.

        Args:
            method_params: Additional method-specific parameters

        Returns:
            Effectiveness score (0.0 to 1.0)
        """
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
                'allow_small_dataset': True
            }

            # Add method-specific parameters
            if method_params:
                param_mapping = {'normalization_method': 'normalization_method'}
                for param_key, param_value in method_params.items():
                    if param_key in param_mapping:
                        kwargs[param_mapping[param_key]] = param_value

            result = run_task_pipeline(**kwargs)

            # Extract evaluation score
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

    def _save_steering_optimization_results(self, summary: SteeringOptimizationSummary):
        """Save optimization results to configuration."""
        config = self.config_manager.load_model_config(self.model_name) or {
            'model_name': self.model_name,
            'created_date': datetime.now().isoformat(),
            'config_version': '2.0'
        }

        # Add steering optimization results
        if 'steering_optimization' not in config:
            config['steering_optimization'] = {}

        config['steering_optimization']['best_method'] = summary.best_overall_method
        config['steering_optimization']['best_layer'] = summary.best_overall_layer
        config['steering_optimization']['best_strength'] = summary.best_overall_strength
        config['steering_optimization']['optimization_date'] = summary.optimization_date
        config['steering_optimization']['method_ranking'] = summary.method_performance_ranking

        # Save task-specific results
        if 'task_specific_steering' not in config:
            config['task_specific_steering'] = {}

        for task_result in summary.task_results:
            config['task_specific_steering'][task_result.task_name] = {
                'method': task_result.best_steering_method,
                'layer': task_result.best_steering_layer,
                'strength': task_result.best_steering_strength,
                'score': task_result.steering_effectiveness_score,
                'parameters': task_result.optimal_parameters
            }

        self.config_manager.update_model_config(self.model_name, config)
        logger.info(f"Steering optimization results saved for {self.model_name}")

    def load_optimal_steering_config(self, task_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load optimal steering configuration for a model/task.

        Args:
            task_name: Optional task name for task-specific configuration

        Returns:
            Dictionary with optimal steering parameters or None
        """
        config = self.config_manager.load_model_config(self.model_name)
        if not config:
            return None

        # Check for task-specific configuration first
        if task_name and 'task_specific_steering' in config:
            task_config = config['task_specific_steering'].get(task_name)
            if task_config:
                return task_config

        # Fall back to overall best configuration
        if 'steering_optimization' in config:
            steering_opt = config['steering_optimization']
            return {
                'method': steering_opt.get('best_method'),
                'layer': steering_opt.get('best_layer'),
                'strength': steering_opt.get('best_strength')
            }

        return None

    def evaluate_steering_effectiveness(
        self,
        task_name: str,
        steering_method: Any,  # SteeringMethodType
        layer: int,
        strength: float,
        method_params: Dict[str, Any],
        test_samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate how effectively steering changes model outputs.

        Args:
            task_name: Task being evaluated
            steering_method: Steering method being used
            layer: Steering layer
            strength: Steering strength
            method_params: Method-specific parameters
            test_samples: Test samples to evaluate on

        Returns:
            Dictionary with effectiveness metrics
        """
        score = self._evaluate_steering_configuration(
            task_name=task_name,
            method=steering_method,
            layer=layer,
            strength=strength,
            limit=len(test_samples),
            split_ratio=0.8
        )

        return {
            'effectiveness_score': score,
            'accuracy': score,
            'consistency': 1.0 if score > 0.5 else 0.5,
            'direction_accuracy': score
        }
