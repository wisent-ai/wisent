"""
Universal Steering Method Optimizer.

This module provides a foolproof way to optimize ANY steering method's parameters.
It uses the universal `train(pair_set)` interface that all methods must implement,
ensuring compatibility with current and future steering methods.

Key design principles:
1. Uses `method.train(pair_set)` - the universal interface all methods implement
2. Automatically extracts method-specific parameters from the registry
3. Search spaces are defined per-method and automatically iterated
4. Evaluation is decoupled from training - any evaluator can be used
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type

import torch

from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations import ExtractionStrategy
from wisent.core.activations.core.atoms import LayerActivations
from wisent.core.utils import resolve_default_device
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.models.core.atoms import SteeringPlan, SteeringVector
from wisent.core.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.steering_methods.registry import SteeringMethodRegistry
from wisent.core.constants import SEPARATOR_WIDTH_STANDARD

from wisent.core.cli.optimization.core.method_optimizer_config import (
    OptimizationConfig, OptimizationResult, OptimizationSummary,
)
from wisent.core.cli.optimization.core.method_optimizer_search import (
    generate_search_space, _get_full_layers,
    _get_method_param_ranges, _generate_param_combinations,
    _determine_activation_layers, _log, _load_evidence_reductions,
)
from wisent.core.cli.optimization.core.method_optimizer_eval import (
    collect_activations, train_method, _resolve_params,
    create_steering_plan, evaluate, evaluate_baseline,
)

logger = logging.getLogger(__name__)


class MethodOptimizer:
    """
    Universal optimizer for any steering method.
    
    This optimizer works with ANY steering method by using the universal
    `train(pair_set)` interface. It handles:
    
    1. Collecting activations with different extraction parameters
    2. Training methods with different method-specific parameters
    3. Evaluating results with any provided evaluator
    4. Tracking and comparing results across configurations
    
    Example usage:
    
        optimizer = MethodOptimizer(model, method_name="grom")
        
        # Generate search space
        configs = optimizer.generate_search_space(num_layers=16)
        
        # Run optimization
        summary = optimizer.optimize(
            train_pairs=train_pairs,
            test_pairs=test_pairs,
            evaluator=evaluator,
            configs=configs,
        )
        
        print(f"Best config: {summary.best_result.config}")
        print(f"Best score: {summary.best_result.score}")
    """
    
    def __init__(
        self,
        model,
        method_name: str,
        device: str | None = None,
        verbose: bool = True,
    ):
        """
        Initialize the optimizer.
        
        Args:
            model: WisentModel instance
            method_name: Name of steering method to optimize
            device: Device for storing activations
            verbose: Whether to print progress
        """
        self.model = model
        self.method_name = method_name.lower()
        self.device = device or resolve_default_device()
        self.verbose = verbose
        
        # Validate method exists
        if not SteeringMethodRegistry.validate_method(self.method_name):
            available = ", ".join(SteeringMethodRegistry.list_methods())
            raise ValueError(f"Unknown method: {method_name}. Available: {available}")
        
        self.method_definition = SteeringMethodRegistry.get(self.method_name)
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    

    # Delegate methods to extracted modules
    _log = _log
    generate_search_space = generate_search_space
    _get_full_layers = _get_full_layers
    _get_method_param_ranges = _get_method_param_ranges
    _generate_param_combinations = _generate_param_combinations
    _determine_activation_layers = _determine_activation_layers
    _load_evidence_reductions = _load_evidence_reductions
    collect_activations = collect_activations
    train_method = train_method
    _resolve_params = _resolve_params
    create_steering_plan = create_steering_plan
    evaluate = evaluate
    evaluate_baseline = evaluate_baseline

    def optimize(
        self,
        train_pairs: ContrastivePairSet,
        test_pairs: ContrastivePairSet,
        evaluator,
        task_name: str,
        configs: Optional[List[OptimizationConfig]] = None,
        progress_callback: Optional[Callable[[int, int, OptimizationResult], None]] = None,
    ) -> OptimizationSummary:
        """
        Run optimization over configurations.

        Args:
            train_pairs: Training pairs for steering vector extraction
            test_pairs: Test pairs for evaluation
            evaluator: Evaluator instance
            task_name: Name of task/benchmark
            configs: Configurations to test (or generate automatically)
            progress_callback: Called after each config with (idx, total, result)

        Returns:
            OptimizationSummary with best result and all results
        """
        start_time = time.time()

        # Evaluate baseline (unsteered) first
        baseline_score, baseline_metrics = self.evaluate_baseline(
            test_pairs, evaluator, task_name
        )

        # Generate configs if not provided
        if configs is None:
            evidence_reductions = self._load_evidence_reductions(task_name)
            configs = self.generate_search_space(
                num_layers=self.model.num_layers,
                evidence_reductions=evidence_reductions,
            )
        
        self._log(f"\n{'='*SEPARATOR_WIDTH_STANDARD}")
        self._log(f"Optimizing {self.method_name.upper()} with {len(configs)} configurations")
        self._log(f"Baseline: {baseline_score:.4f}")
        self._log(f"{'='*SEPARATOR_WIDTH_STANDARD}\n")
        
        results: List[OptimizationResult] = []
        best_result: Optional[OptimizationResult] = None
        
        for idx, config in enumerate(configs):
            try:
                self._log(f"[{idx+1}/{len(configs)}] Testing: layers={config.layers}, "
                         f"strength={config.strength}, token_agg={config.token_aggregation.value}")
                
                # Collect activations
                t0 = time.time()
                train_with_acts = self.collect_activations(train_pairs, config)
                
                # Train method
                t1 = time.time()
                steering_vectors, metadata = self.train_method(train_with_acts, config)
                training_time = time.time() - t1
                
                # Create steering plan
                steering_plan = self.create_steering_plan(steering_vectors, config)
                
                # Evaluate
                t2 = time.time()
                score, metrics = self.evaluate(
                    steering_plan, test_pairs, evaluator, task_name
                )
                evaluation_time = time.time() - t2
                
                result = OptimizationResult(
                    config=config,
                    score=score,
                    metrics=metrics,
                    steering_vectors=steering_vectors,
                    training_time=training_time,
                    evaluation_time=evaluation_time,
                    metadata=metadata,
                )
                results.append(result)
                
                self._log(f"        Score: {score:.4f} (train: {training_time:.1f}s, eval: {evaluation_time:.1f}s)")
                
                # Track best
                if best_result is None or score > best_result.score:
                    best_result = result
                    self._log(f"        *** New best! ***")
                
                # Callback
                if progress_callback:
                    progress_callback(idx, len(configs), result)
                
            except Exception as e:
                logger.error(f"Config {idx} failed: {e}")
                self._log(f"        FAILED: {e}")
        
        total_time = time.time() - start_time
        
        improvement = (best_result.score - baseline_score) if best_result else 0.0
        
        self._log(f"\n{'='*SEPARATOR_WIDTH_STANDARD}")
        self._log(f"Optimization complete in {total_time:.1f}s")
        self._log(f"Baseline: {baseline_score:.4f}")
        self._log(f"Best score: {best_result.score:.4f}" if best_result else "No results")
        self._log(f"Improvement: {improvement:+.4f}" if best_result else "")
        self._log(f"{'='*SEPARATOR_WIDTH_STANDARD}\n")
        
        return OptimizationSummary(
            best_result=best_result,
            all_results=results,
            method_name=self.method_name,
            task_name=task_name,
            total_time=total_time,
            configs_tested=len(results),
            baseline_score=baseline_score,
            baseline_metrics=baseline_metrics,
        )




def optimize_steering_method(
    model,
    method_name: str,
    train_pairs: ContrastivePairSet,
    test_pairs: ContrastivePairSet,
    evaluator,
    task_name: str,
    verbose: bool = True,
) -> OptimizationSummary:
    """
    Convenience function to optimize a steering method.

    This is the main entry point for optimizing any steering method.
    It handles all the complexity of:
    - Generating appropriate search spaces
    - Collecting activations
    - Training methods
    - Evaluating results

    Args:
        model: WisentModel instance
        method_name: Name of method to optimize (caa, tecza, tetno, grom)
        train_pairs: Training pairs
        test_pairs: Test pairs
        evaluator: Evaluator instance
        task_name: Name of task/benchmark
        verbose: Print progress

    Returns:
        OptimizationSummary with results
    """
    optimizer = MethodOptimizer(
        model=model,
        method_name=method_name,
        verbose=verbose,
    )

    return optimizer.optimize(
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        evaluator=evaluator,
        task_name=task_name,
    )
