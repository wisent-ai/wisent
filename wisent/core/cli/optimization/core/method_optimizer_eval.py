"""Training and evaluation methods for MethodOptimizer."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations import ExtractionStrategy
from wisent.core.activations.core.atoms import LayerActivations
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.models.core.atoms import SteeringPlan, SteeringVector
from wisent.core.steering_methods.registry import SteeringMethodRegistry
from wisent.core.cli.optimization.core.method_optimizer_config import (
    OptimizationConfig, OptimizationResult, OptimizationSummary,
)

logger = logging.getLogger(__name__)


    def train_method(
        self,
        pair_set: ContrastivePairSet,
        config: OptimizationConfig,
    ) -> Tuple[LayerActivations, Dict[str, Any]]:
        """
        Train a steering method using the universal interface.
        
        Args:
            pair_set: ContrastivePairSet with activations already collected
            config: Configuration with method parameters
            
        Returns:
            Tuple of (LayerActivations, metadata dict)
        """
        # Create method instance with parameters from config
        method_class = self.method_definition.get_method_class()
        
        # Merge default params with config params
        params = self.method_definition.get_default_params()
        params.update(config.method_params)
        
        # Resolve special parameter values
        params = self._resolve_params(params, config)
        
        # Create method instance
        method = method_class(**params)

        # Train using universal interface
        # Force CPU context to avoid MPS/CPU tensor mismatches on Mac
        import torch
        with torch.device("cpu"):
            steering_vectors = method.train(pair_set)
        
        # Extract any metadata from training
        metadata = {}
        if hasattr(method, "_training_logs"):
            metadata["training_logs"] = method._training_logs
        
        return steering_vectors, metadata
    
    def _resolve_params(
        self,
        params: Dict[str, Any],
        config: OptimizationConfig,
    ) -> Dict[str, Any]:
        """Resolve special parameter values like 'auto'."""
        resolved = dict(params)
        num_layers = self.model.num_layers
        
        # Resolve sensor_layer
        if resolved.get("sensor_layer") == "auto":
            # Use 75% through the network
            resolved["sensor_layer"] = int(num_layers * 0.75)
        
        # Resolve steering_layers from string config to actual layer list
        steering_config = resolved.get("steering_layers")
        if isinstance(steering_config, str):
            base_layer = int(config.layers[0]) if config.layers else 0
            if steering_config == "single":
                resolved["steering_layers"] = [base_layer]
            elif steering_config == "range_3":
                resolved["steering_layers"] = list(range(
                    max(0, base_layer - 1),
                    min(num_layers, base_layer + 2)
                ))
            elif steering_config == "range_5":
                resolved["steering_layers"] = list(range(
                    max(0, base_layer - 2),
                    min(num_layers, base_layer + 3)
                ))
            elif steering_config == "all_late":
                start = int(num_layers * 0.75)
                resolved["steering_layers"] = list(range(start, num_layers - 1))
        
        return resolved
    
    def create_steering_plan(
        self,
        steering_vectors: LayerActivations,
        config: OptimizationConfig,
    ) -> SteeringPlan:
        """Create a SteeringPlan from trained vectors and config."""
        raw_map = steering_vectors.to_dict()
        
        # Apply strength scaling
        scaled_map = {}
        for layer, vec in raw_map.items():
            steering_vec = SteeringVector(vector=vec, scale=config.strength)
            scaled_map[layer] = steering_vec
        
        return SteeringPlan(
            layers=scaled_map,
            layers_description=[
                f"{self.method_name.upper()} {config.to_dict()}"
            ],
        )
    
    def evaluate(
        self,
        steering_plan: SteeringPlan,
        test_pairs: ContrastivePairSet,
        evaluator,
        task_name: str,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a steering plan.
        
        Args:
            steering_plan: The steering plan to evaluate
            test_pairs: Test pairs for evaluation
            evaluator: Evaluator instance (from EvaluatorRotator)
            task_name: Name of the task for evaluator context
            
        Returns:
            Tuple of (primary_score, metrics_dict)
        """
        correct = 0
        total = 0
        
        for pair in test_pairs.pairs:
            try:
                # Generate response with steering applied
                self.model.apply_steering(steering_plan)
                generated_response = self.model.generate(
                    [[{"role": "user", "content": pair.prompt}]],
                    max_new_tokens=100,
                    use_steering=True,
                    steering_plan=steering_plan,
                )[0]
                self.model.clear_steering()
                
                choices = [
                    pair.negative_response.model_response,
                    pair.positive_response.model_response,
                ]
                expected = pair.positive_response.model_response
                metadata = pair.metadata or {}
                
                eval_result = evaluator.evaluate(
                    response=generated_response,
                    expected=expected,
                    model=self.model,
                    question=pair.prompt,
                    choices=choices,
                    steering_plan=steering_plan,
                    correct_answers=metadata.get("correct_answers", []),
                    incorrect_answers=metadata.get("incorrect_answers", []),
                    task_name=task_name,
                )
                
                if eval_result.ground_truth == "TRUTHFUL":
                    correct += 1
                total += 1
            except Exception as e:
                logger.warning(f"Evaluation failed for pair: {e}")
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return accuracy, {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
    
    def evaluate_baseline(
        self,
        test_pairs: ContrastivePairSet,
        evaluator,
        task_name: str,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate baseline (unsteered) performance.
        
        Args:
            test_pairs: Test pairs for evaluation
            evaluator: Evaluator instance
            task_name: Name of the task for evaluator context
            
        Returns:
            Tuple of (baseline_score, metrics_dict)
        """
        self._log("Evaluating baseline (unsteered)...")
        
        correct = 0
        total = 0
        
        for pair in test_pairs.pairs:
            try:
                # Generate response WITHOUT steering
                generated_response = self.model.generate(
                    [[{"role": "user", "content": pair.prompt}]],
                    max_new_tokens=100,
                )[0]
                
                expected = pair.positive_response.model_response
                metadata = pair.metadata or {}
                
                eval_result = evaluator.evaluate(
                    response=generated_response,
                    expected=expected,
                    model=self.model,
                    question=pair.prompt,
                    choices=[
                        pair.negative_response.model_response,
                        pair.positive_response.model_response,
                    ],
                    correct_answers=metadata.get("correct_answers", []),
                    incorrect_answers=metadata.get("incorrect_answers", []),
                    task_name=task_name,
                )
                
                if eval_result.ground_truth == "TRUTHFUL":
                    correct += 1
                total += 1
            except Exception as e:
                logger.warning(f"Baseline evaluation failed for pair: {e}")
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        self._log(f"Baseline score: {accuracy:.4f} ({correct}/{total})")
        
        return accuracy, {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
    
