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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type

import torch

from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations.extraction_strategy import ExtractionStrategy
from wisent.core.activations.core.atoms import LayerActivations
from wisent.core.utils.device import resolve_default_device

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.models.core.atoms import SteeringPlan, SteeringVector
from wisent.core.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.steering_methods.registry import SteeringMethodRegistry

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for a single optimization trial."""
    
    method_name: str
    """Name of the steering method (caa, prism, pulse, titan)."""
    
    # Activation extraction parameters
    layers: List[str]
    """Layer indices to extract activations from."""
    
    token_aggregation: ExtractionStrategy
    """How to aggregate tokens within a sequence."""
    
    prompt_strategy: ExtractionStrategy
    """How to construct prompts for the model."""
    
    # Application parameters
    strength: float = 1.0
    """Steering strength multiplier."""
    
    strategy: str = "constant"
    """Steering application strategy."""
    
    # Method-specific parameters
    method_params: Dict[str, Any] = field(default_factory=dict)
    """Method-specific parameters (num_directions, sensor_layer, etc.)."""
    
    def __hash__(self):
        return hash((
            self.method_name,
            tuple(self.layers),
            self.token_aggregation.value,
            self.prompt_strategy.value,
            self.strength,
            self.strategy,
            tuple(sorted(self.method_params.items())),
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "layers": self.layers,
            "token_aggregation": self.token_aggregation.value,
            "prompt_strategy": self.prompt_strategy.value,
            "strength": self.strength,
            "strategy": self.strategy,
            "method_params": self.method_params,
        }


@dataclass
class OptimizationResult:
    """Result of a single optimization trial."""
    
    config: OptimizationConfig
    """The configuration that was tested."""
    
    score: float
    """Primary evaluation score."""
    
    metrics: Dict[str, float] = field(default_factory=dict)
    """Additional metrics (accuracy, f1, etc.)."""
    
    steering_vectors: Optional[LayerActivations] = None
    """Trained steering vectors (optional, for caching)."""
    
    training_time: float = 0.0
    """Time taken to train the method."""
    
    evaluation_time: float = 0.0
    """Time taken to evaluate."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata from training."""


@dataclass
class OptimizationSummary:
    """Summary of optimization run."""
    
    best_result: OptimizationResult
    """Best result found."""
    
    all_results: List[OptimizationResult]
    """All results from the optimization."""
    
    method_name: str
    """Method that was optimized."""
    
    task_name: str
    """Task/benchmark used for evaluation."""
    
    total_time: float = 0.0
    """Total optimization time."""
    
    configs_tested: int = 0
    """Number of configurations tested."""
    
    baseline_score: float = 0.0
    """Baseline (unsteered) accuracy for comparison."""
    
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    """Baseline metrics (accuracy, correct, total)."""


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
    
        optimizer = MethodOptimizer(model, method_name="titan")
        
        # Generate search space
        configs = optimizer.generate_search_space(num_layers=16, quick=True)
        
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
    
    def generate_search_space(
        self,
        num_layers: int,
        quick: bool = False,
        custom_layers: Optional[List[int]] = None,
        custom_strengths: Optional[List[float]] = None,
        custom_token_aggregations: Optional[List[str]] = None,
        custom_prompt_strategies: Optional[List[str]] = None,
        custom_method_params: Optional[Dict[str, List[Any]]] = None,
    ) -> List[OptimizationConfig]:
        """
        Generate search space for optimization.
        
        Args:
            num_layers: Number of layers in the model
            quick: Use reduced search space for faster testing
            custom_*: Override default search values
            
        Returns:
            List of OptimizationConfig to test
        """
        # Default extraction parameter ranges
        if quick:
            layers = custom_layers or self._get_quick_layers(num_layers)
            strengths = custom_strengths or [0.5, 1.0, 1.5]
            token_aggs = custom_token_aggregations or ["last_token"]
            prompt_strats = custom_prompt_strategies or ["chat_template"]
            steering_strategies = ["constant"]
        else:
            layers = custom_layers or self._get_full_layers(num_layers)
            strengths = custom_strengths or [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            token_aggs = custom_token_aggregations or ["last_token", "mean_pooling", "first_token", "max_pooling", "continuation_token"]
            prompt_strats = custom_prompt_strategies or ["chat_template", "direct_completion", "multiple_choice", "role_playing", "instruction_following"]
            steering_strategies = ["constant", "initial_only", "diminishing", "increasing", "gaussian"]
        
        # Get method-specific parameter ranges
        method_param_ranges = self._get_method_param_ranges(quick, custom_method_params)
        
        # Generate all configurations
        configs = []
        
        # Convert to enums
        token_agg_map = {
            "last_token": ExtractionStrategy.CHAT_LAST,
            "mean_pooling": ExtractionStrategy.CHAT_MEAN,
            "first_token": ExtractionStrategy.CHAT_FIRST,
            "max_pooling": ExtractionStrategy.CHAT_MAX_NORM,
            "continuation_token": ExtractionStrategy.CHAT_FIRST,  # First answer token
        }
        
        prompt_strat_map = {
            "chat_template": ExtractionStrategy.CHAT_LAST,
            "direct_completion": ExtractionStrategy.CHAT_LAST,
            "multiple_choice": ExtractionStrategy.MC_BALANCED,
            "role_playing": ExtractionStrategy.ROLE_PLAY,
            "instruction_following": ExtractionStrategy.CHAT_LAST,
        }
        
        # Generate method param combinations
        method_param_combos = self._generate_param_combinations(method_param_ranges)
        
        for layer in layers:
            for strength in strengths:
                for token_agg_name in token_aggs:
                    for prompt_strat_name in prompt_strats:
                        for steering_strat in steering_strategies:
                            for method_params in method_param_combos:
                                # Determine which layers to collect activations for
                                # For multi-layer methods, collect all needed layers
                                activation_layers = self._determine_activation_layers(
                                    layer, num_layers, method_params
                                )
                                
                                config = OptimizationConfig(
                                    method_name=self.method_name,
                                    layers=[str(l) for l in activation_layers],
                                    token_aggregation=token_agg_map.get(
                                        token_agg_name, ExtractionStrategy.CHAT_LAST
                                    ),
                                    prompt_strategy=prompt_strat_map.get(
                                        prompt_strat_name, ExtractionStrategy.CHAT_LAST
                                    ),
                                    strength=strength,
                                    strategy=steering_strat,
                                    method_params=method_params,
                                )
                                configs.append(config)
        
        return configs
    
    def _get_quick_layers(self, num_layers: int) -> List[int]:
        """Get reduced layer set for quick search."""
        if num_layers <= 12:
            return [num_layers // 3, num_layers // 2, 2 * num_layers // 3]
        elif num_layers <= 24:
            return [num_layers // 4, num_layers // 2, 3 * num_layers // 4]
        else:
            return [num_layers // 4, num_layers // 3, num_layers // 2, 2 * num_layers // 3]
    
    def _get_full_layers(self, num_layers: int) -> List[int]:
        """Get full layer set for comprehensive search."""
        # Test ALL layers from 0 to num_layers-1
        return list(range(num_layers))
    
    def _get_method_param_ranges(
        self,
        quick: bool,
        custom: Optional[Dict[str, List[Any]]] = None,
    ) -> Dict[str, List[Any]]:
        """Get method-specific parameter ranges."""
        custom = custom or {}
        
        if self.method_name == "caa":
            return {
                "normalize": custom.get("normalize", [True]),
            }
        
        elif self.method_name == "prism":
            if quick:
                return {
                    "num_directions": custom.get("num_directions", [2, 3]),
                    "optimization_steps": custom.get("optimization_steps", [50]),
                    "retain_weight": custom.get("retain_weight", [0.1]),
                    "learning_rate": custom.get("learning_rate", [0.01]),
                    "use_caa_init": custom.get("use_caa_init", [True]),
                }
            return {
                "num_directions": custom.get("num_directions", [1, 2, 3, 5]),
                "optimization_steps": custom.get("optimization_steps", [50, 100]),
                "retain_weight": custom.get("retain_weight", [0.0, 0.1, 0.3]),
                "learning_rate": custom.get("learning_rate", [0.01]),
                "independence_weight": custom.get("independence_weight", [0.05]),
                "use_caa_init": custom.get("use_caa_init", [True]),
            }
        
        elif self.method_name == "pulse":
            if quick:
                return {
                    "sensor_layer": custom.get("sensor_layer", ["auto"]),  # Will be resolved
                    "steering_layers": custom.get("steering_layers", ["range_3"]),
                    "condition_threshold": custom.get("condition_threshold", [0.5]),
                    "gate_temperature": custom.get("gate_temperature", [0.5]),
                    "use_entropy_scaling": custom.get("use_entropy_scaling", [False]),
                    "max_alpha": custom.get("max_alpha", [2.0]),
                }
            return {
                "sensor_layer": custom.get("sensor_layer", ["auto"]),
                "steering_layers": custom.get("steering_layers", ["single", "range_3", "range_5"]),
                "condition_threshold": custom.get("condition_threshold", [0.3, 0.5, 0.7]),
                "gate_temperature": custom.get("gate_temperature", [0.1, 0.5, 1.0]),
                "per_layer_scaling": custom.get("per_layer_scaling", [True, False]),
                "use_entropy_scaling": custom.get("use_entropy_scaling", [True, False]),
                "max_alpha": custom.get("max_alpha", [1.5, 2.0, 3.0]),
            }
        
        elif self.method_name == "titan":
            if quick:
                return {
                    "num_directions": custom.get("num_directions", [3]),
                    "sensor_layer": custom.get("sensor_layer", ["auto"]),
                    "steering_layers": custom.get("steering_layers", ["range_3"]),
                    "gate_hidden_dim": custom.get("gate_hidden_dim", [64]),
                    "intensity_hidden_dim": custom.get("intensity_hidden_dim", [32]),
                    "optimization_steps": custom.get("optimization_steps", [100]),
                    "behavior_weight": custom.get("behavior_weight", [1.0]),
                    "retain_weight": custom.get("retain_weight", [0.2]),
                    "max_alpha": custom.get("max_alpha", [2.0]),
                }
            return {
                "num_directions": custom.get("num_directions", [2, 3, 5]),
                "sensor_layer": custom.get("sensor_layer", ["auto"]),
                "steering_layers": custom.get("steering_layers", ["range_3", "range_5", "all_late"]),
                "gate_hidden_dim": custom.get("gate_hidden_dim", [32, 64, 128]),
                "intensity_hidden_dim": custom.get("intensity_hidden_dim", [16, 32, 64]),
                "optimization_steps": custom.get("optimization_steps", [100, 200]),
                "behavior_weight": custom.get("behavior_weight", [0.5, 1.0]),
                "retain_weight": custom.get("retain_weight", [0.1, 0.2, 0.5]),
                "sparse_weight": custom.get("sparse_weight", [0.0, 0.05]),
                "max_alpha": custom.get("max_alpha", [2.0, 3.0]),
            }
        
        # Default for unknown methods - empty params
        return {}
    
    def _generate_param_combinations(
        self,
        param_ranges: Dict[str, List[Any]],
    ) -> List[Dict[str, Any]]:
        """Generate all combinations of method parameters."""
        if not param_ranges:
            return [{}]
        
        import itertools
        
        keys = list(param_ranges.keys())
        values = [param_ranges[k] for k in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _determine_activation_layers(
        self,
        base_layer: int,
        num_layers: int,
        method_params: Dict[str, Any],
    ) -> List[int]:
        """Determine which layers to collect activations for."""
        # For methods that need multi-layer activations
        steering_layers_config = method_params.get("steering_layers", "single")
        
        if steering_layers_config == "single":
            return [base_layer]
        elif steering_layers_config == "range_3":
            return list(range(max(0, base_layer - 1), min(num_layers, base_layer + 2)))
        elif steering_layers_config == "range_5":
            return list(range(max(0, base_layer - 2), min(num_layers, base_layer + 3)))
        elif steering_layers_config == "all_late":
            start = int(num_layers * 0.75)
            return list(range(start, num_layers - 1))
        elif isinstance(steering_layers_config, list):
            return steering_layers_config
        else:
            return [base_layer]
    
    def collect_activations(
        self,
        pairs: ContrastivePairSet,
        config: OptimizationConfig,
    ) -> ContrastivePairSet:
        """
        Collect activations for a pair set using the given config.
        
        Args:
            pairs: ContrastivePairSet to collect activations for
            config: Configuration specifying layers, aggregation, etc.
            
        Returns:
            ContrastivePairSet with activations populated
        """
        collector = ActivationCollector(model=self.model, store_device=self.device)
        
        updated_pairs = []
        for pair in pairs.pairs:
            updated_pair = collector.collect_for_pair(
                pair,
                layers=config.layers,
                aggregation=config.token_aggregation,
                return_full_sequence=False,
                normalize_layers=False,
            )
            updated_pairs.append(updated_pair)
        
        return ContrastivePairSet(
            name=pairs.name if hasattr(pairs, 'name') else "collected",
            pairs=updated_pairs,
            task_type=pairs.task_type if hasattr(pairs, 'task_type') else None,
        )
    
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
    
    def optimize(
        self,
        train_pairs: ContrastivePairSet,
        test_pairs: ContrastivePairSet,
        evaluator,
        task_name: str,
        configs: Optional[List[OptimizationConfig]] = None,
        quick: bool = False,
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
            quick: Use quick search space if configs not provided
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
            configs = self.generate_search_space(
                num_layers=self.model.num_layers,
                quick=quick,
            )
        
        self._log(f"\n{'='*60}")
        self._log(f"Optimizing {self.method_name.upper()} with {len(configs)} configurations")
        self._log(f"Baseline: {baseline_score:.4f}")
        self._log(f"{'='*60}\n")
        
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
        
        self._log(f"\n{'='*60}")
        self._log(f"Optimization complete in {total_time:.1f}s")
        self._log(f"Baseline: {baseline_score:.4f}")
        self._log(f"Best score: {best_result.score:.4f}" if best_result else "No results")
        self._log(f"Improvement: {improvement:+.4f}" if best_result else "")
        self._log(f"{'='*60}\n")
        
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
    quick: bool = False,
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
        method_name: Name of method to optimize (caa, prism, pulse, titan)
        train_pairs: Training pairs
        test_pairs: Test pairs
        evaluator: Evaluator instance
        task_name: Name of task/benchmark
        quick: Use reduced search space
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
        quick=quick,
    )
