"""
Steering Parameter Optimizer for Wisent.

Optimizes steering-specific parameters including:
1. Optimal steering layer (may differ from classification layer)
2. Optimal steering strength and dynamics
3. Steering method selection and configuration
4. Task-specific steering parameter tuning
5. Token aggregation strategy
6. Prompt construction strategy
7. Steering application strategy (constant, diminishing, initial)

This module builds on top of classification optimization to find optimal
steering configurations for each model and task.
"""

import logging
import json
import time
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from pathlib import Path

from .config_manager import ModelConfigManager
from .activations.extraction_strategy import ExtractionStrategy

from wisent.core.errors import (
    MissingParameterError,
    SteeringMethodUnknownError,
    UnknownTypeError,
    InsufficientDataError,
)

logger = logging.getLogger(__name__)


class SteeringApplicationStrategy(Enum):
    """How steering is applied during generation."""
    CONSTANT = "constant"           # Same strength throughout generation
    INITIAL_ONLY = "initial_only"   # Only apply at first N tokens
    DIMINISHING = "diminishing"     # Strength decreases over tokens
    INCREASING = "increasing"       # Strength increases over tokens
    GAUSSIAN = "gaussian"           # Gaussian curve centered at specific token position


@dataclass
class SteeringApplicationConfig:
    """Configuration for how steering is applied during generation."""
    strategy: SteeringApplicationStrategy = SteeringApplicationStrategy.CONSTANT
    # For INITIAL_ONLY: number of tokens to apply steering
    initial_tokens: int = 10
    # For DIMINISHING/INCREASING: decay/growth rate
    rate: float = 0.1
    # For GAUSSIAN: center position (as fraction of sequence)
    gaussian_center: float = 0.5
    gaussian_width: float = 0.2


def get_default_token_aggregation_strategies() -> List[ExtractionStrategy]:
    """Get token aggregation strategies to test."""
    return [
        ExtractionStrategy.CHAT_LAST,
        ExtractionStrategy.CHAT_MEAN,
        ExtractionStrategy.CHAT_FIRST,
        ExtractionStrategy.CHAT_MAX_NORM,
    ]


def get_default_prompt_construction_strategies() -> List[ExtractionStrategy]:
    """Get prompt construction strategies to test."""
    return [
        ExtractionStrategy.CHAT_LAST,
        ExtractionStrategy.CHAT_LAST,
        ExtractionStrategy.CHAT_LAST,
    ]


def get_default_steering_application_configs() -> List[SteeringApplicationConfig]:
    """Get steering application configurations to test."""
    return [
        SteeringApplicationConfig(strategy=SteeringApplicationStrategy.CONSTANT),
        SteeringApplicationConfig(strategy=SteeringApplicationStrategy.INITIAL_ONLY, initial_tokens=5),
        SteeringApplicationConfig(strategy=SteeringApplicationStrategy.INITIAL_ONLY, initial_tokens=20),
        SteeringApplicationConfig(strategy=SteeringApplicationStrategy.DIMINISHING, rate=0.05),
        SteeringApplicationConfig(strategy=SteeringApplicationStrategy.DIMINISHING, rate=0.1),
    ]


from wisent.core.steering_methods import SteeringMethodRegistry, SteeringMethodType


def get_default_steering_configs() -> List['SteeringMethodConfig']:
    """Get default steering method configurations from centralized registry."""
    configs = []
    for method_name in SteeringMethodRegistry.list_methods():
        definition = SteeringMethodRegistry.get(method_name)
        method_type = definition.method_type
        
        # Base config with default params
        configs.append(SteeringMethodConfig(
            name=method_name.upper(),
            method=method_type,
            params=definition.get_default_params()
        ))
        
        # Add L2 normalized variant
        configs.append(SteeringMethodConfig(
            name=f"{method_name.upper()}_L2",
            method=method_type,
            params={**definition.get_default_params(), "normalization_method": "l2_unit"}
        ))
    
    return configs


# Use centralized registry for steering methods
SteeringMethod = SteeringMethodType  # Alias for backward compatibility


@dataclass
class SteeringMethodConfig:
    """Configuration for a specific steering method with parameter variations."""
    name: str  # Display name like "CAA_L2"
    method: SteeringMethodType
    params: Dict[str, Any]  # Method-specific parameters
    
    def __post_init__(self):
        """Ensure method is SteeringMethodType enum."""
        if isinstance(self.method, str):
            self.method = SteeringMethodType(self.method.lower())


@dataclass
class SteeringOptimizationResult:
    """Results from optimizing steering parameters for a single task."""
    task_name: str
    best_steering_layer: int
    best_steering_method: str
    best_steering_strength: float
    optimal_parameters: Dict[str, Any]  # Method-specific parameters
    steering_effectiveness_score: float  # How well steering changes outputs
    classification_accuracy_impact: float  # Impact on classification performance
    optimization_time_seconds: float
    total_configurations_tested: int
    # New fields for extended optimization
    best_token_aggregation: Optional[str] = None
    best_prompt_construction: Optional[str] = None
    best_steering_application: Optional[str] = None
    steering_application_params: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class SteeringOptimizationSummary:
    """Summary of steering optimization across tasks/methods."""
    model_name: str
    optimization_type: str  # "single_task", "multi_task", "method_comparison", "comprehensive"
    total_configurations_tested: int
    optimization_time_minutes: float
    best_overall_method: str
    best_overall_layer: int
    best_overall_strength: float
    method_performance_ranking: Dict[str, float]  # method -> effectiveness score
    layer_effectiveness_analysis: Dict[int, float]  # layer -> avg effectiveness
    task_results: List[SteeringOptimizationResult]
    optimization_date: str
    # New fields for extended optimization
    best_token_aggregation: Optional[str] = None
    best_prompt_construction: Optional[str] = None
    best_steering_application: Optional[str] = None
    token_aggregation_ranking: Optional[Dict[str, float]] = None
    prompt_construction_ranking: Optional[Dict[str, float]] = None
    steering_application_ranking: Optional[Dict[str, float]] = None


class SteeringOptimizer:
    """
    Framework for optimizing steering parameters.
    
    This class provides the structure for steering optimization but requires
    implementation of the actual optimization algorithms for each steering method.
    """
    
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
        
        # Load classification parameters if available (steering often builds on classification)
        self.classification_config = self.config_manager.load_model_config(model_name)
        if self.classification_config:
            self.base_classification_layer = self.classification_config.get("optimal_parameters", {}).get("classification_layer")
            logger.info(f"📊 Found existing classification layer: {self.base_classification_layer}")
        else:
            self.base_classification_layer = None
            logger.warning("⚠️ No existing classification configuration found")
    
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
            # Use all default configurations
            method_configs = get_default_steering_configs()
        else:
            method_configs = []
            for item in methods_to_test:
                if isinstance(item, SteeringMethodConfig):
                    method_configs.append(item)
                elif isinstance(item, SteeringMethod):
                    # Convert simple method to config with default params
                    method_configs.append(SteeringMethodConfig(
                        name=item.value,
                        method=item,
                        params={}
                    ))
                elif isinstance(item, str):
                    # Convert string to SteeringMethod enum
                    try:
                        method = SteeringMethod(item)
                        method_configs.append(SteeringMethodConfig(
                            name=method.value,
                            method=method,
                            params={}
                        ))
                    except ValueError:
                        logger.warning(f"Unknown steering method: {item}")
                else:
                    logger.warning(f"Unknown method type: {type(item)}, value: {item}")
        
        if strength_range is None:
            strength_range = [0.5, 1.0, 1.5, 2.0]
            
        logger.info(f"🎯 Comparing {len(method_configs)} steering method configurations for task: {task_name}")
        
        start_time = time.time()
        task_results = []
        all_results = {}
        
        # Determine layer search range
        if layer_range:
            layers_to_test = self._parse_layer_range(layer_range)
        elif self.base_classification_layer:
            # Search around classification layer
            min_layer = max(1, self.base_classification_layer - 2)
            max_layer = min(32, self.base_classification_layer + 2)  # Assume max 32 layers
            layers_to_test = list(range(min_layer, max_layer + 1))
        else:
            # Default: try to detect model layer count, otherwise use all common layers
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
                num_layers = getattr(config, 'num_hidden_layers', None) or \
                             getattr(config, 'n_layer', None) or \
                             getattr(config, 'num_layers', None) or 32
                layers_to_test = list(range(num_layers))
                logger.info(f"📊 Detected {num_layers} layers, testing all")
            except Exception:
                # Fallback: test common layer range for typical models
                layers_to_test = list(range(32))
                logger.warning("⚠️ Could not detect layer count, testing layers 0-31")
        
        configurations_tested = 0
        best_overall_score = 0.0
        best_overall_config = None
        
        # Test each method configuration
        for method_config in method_configs:
            method_results = []
            
            for layer in layers_to_test:
                for strength in strength_range:
                    if time.time() - start_time > max_time_minutes * 60:
                        logger.warning(f"⏰ Time limit reached, stopping optimization")
                        break
                        
                    try:
                        # Run evaluation for this configuration
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
                            'method': method_config.name,  # Use display name
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
        method_performance = {}
        layer_effectiveness = {}
        
        for method, results in all_results.items():
            if results:
                scores = [r['score'] for r in results]
                method_performance[method] = max(scores)
                
                # Aggregate by layer
                for result in results:
                    layer = result['layer']
                    if layer not in layer_effectiveness:
                        layer_effectiveness[layer] = []
                    layer_effectiveness[layer].append(result['score'])
        
        # Average layer effectiveness
        for layer in layer_effectiveness:
            layer_effectiveness[layer] = sum(layer_effectiveness[layer]) / len(layer_effectiveness[layer])
        
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
                classification_accuracy_impact=0.0,  # Not measured here
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
        
        # Save the results
        self._save_steering_optimization_results(summary)

        return summary

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
        Full optimization across all steering dimensions:
        - Steering method (CAA)
        - Layer
        - Strength
        - Token aggregation strategy
        - Prompt construction strategy
        - Steering application strategy

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
            methods_to_test = [SteeringMethod.CAA]  # Start with CAA only for speed
        if strength_range is None:
            strength_range = [0.5, 1.0, 1.5, 2.0]
        if token_aggregation_strategies is None:
            token_aggregation_strategies = get_default_token_aggregation_strategies()
        if prompt_construction_strategies is None:
            prompt_construction_strategies = get_default_prompt_construction_strategies()
        if steering_application_configs is None:
            steering_application_configs = get_default_steering_application_configs()

        # Determine layer range
        if layer_range:
            layers_to_test = self._parse_layer_range(layer_range)
        elif self.base_classification_layer:
            min_layer = max(1, self.base_classification_layer - 2)
            max_layer = min(32, self.base_classification_layer + 2)
            layers_to_test = list(range(min_layer, max_layer + 1))
        else:
            layers_to_test = [4, 6, 8, 10, 12]  # Default range for smaller models

        logger.info(f"🚀 Full steering optimization for task: {task_name}")
        logger.info(f"   Methods: {[m.value for m in methods_to_test]}")
        logger.info(f"   Layers: {layers_to_test}")
        logger.info(f"   Strengths: {strength_range}")
        logger.info(f"   Token aggregations: {[s.value for s in token_aggregation_strategies]}")
        logger.info(f"   Prompt constructions: {[s.value for s in prompt_construction_strategies]}")
        logger.info(f"   Steering applications: {[c.strategy.value for c in steering_application_configs]}")

        start_time = time.time()
        all_results = []
        best_score = -float('inf')
        best_config = None
        configurations_tested = 0

        # Track rankings by dimension
        method_scores = {}
        layer_scores = {}
        strength_scores = {}
        token_agg_scores = {}
        prompt_const_scores = {}
        steering_app_scores = {}

        # Iterate through all combinations
        for method in methods_to_test:
            for layer in layers_to_test:
                for strength in strength_range:
                    for token_agg in token_aggregation_strategies:
                        for prompt_const in prompt_construction_strategies:
                            for steering_app in steering_application_configs:
                                # Check time limit
                                if time.time() - start_time > max_time_minutes * 60:
                                    logger.warning("⏰ Time limit reached")
                                    break

                                try:
                                    score = self._evaluate_full_configuration(
                                        task_name=task_name,
                                        method=method,
                                        layer=layer,
                                        strength=strength,
                                        token_aggregation=token_agg,
                                        prompt_construction=prompt_const,
                                        steering_application=steering_app,
                                        limit=limit,
                                        split_ratio=split_ratio
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

                                    # Update best
                                    if score > best_score:
                                        best_score = score
                                        best_config = config

                                    # Track scores by dimension
                                    method_scores.setdefault(method.value, []).append(score)
                                    layer_scores.setdefault(layer, []).append(score)
                                    strength_scores.setdefault(strength, []).append(score)
                                    token_agg_scores.setdefault(token_agg.value, []).append(score)
                                    prompt_const_scores.setdefault(prompt_const.value, []).append(score)
                                    steering_app_scores.setdefault(steering_app.strategy.value, []).append(score)

                                    if self.verbose:
                                        logger.info(
                                            f"   {method.value} L{layer} S{strength} "
                                            f"T:{token_agg.value[:4]} P:{prompt_const.value[:4]} "
                                            f"A:{steering_app.strategy.value[:4]}: {score:.3f}"
                                        )

                                except Exception as e:
                                    logger.error(f"   Error: {e}")

        # Compute average scores by dimension
        method_ranking = {k: sum(v)/len(v) for k, v in method_scores.items()}
        layer_ranking = {k: sum(v)/len(v) for k, v in layer_scores.items()}
        token_agg_ranking = {k: sum(v)/len(v) for k, v in token_agg_scores.items()}
        prompt_const_ranking = {k: sum(v)/len(v) for k, v in prompt_const_scores.items()}
        steering_app_ranking = {k: sum(v)/len(v) for k, v in steering_app_scores.items()}

        optimization_time = time.time() - start_time

        # Create task result
        task_result = None
        if best_config:
            task_result = SteeringOptimizationResult(
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

        # Create summary
        summary = SteeringOptimizationSummary(
            model_name=self.model_name,
            optimization_type="comprehensive",
            total_configurations_tested=configurations_tested,
            optimization_time_minutes=optimization_time / 60,
            best_overall_method=best_config['method'] if best_config else "none",
            best_overall_layer=best_config['layer'] if best_config else 0,
            best_overall_strength=best_config['strength'] if best_config else 0.0,
            method_performance_ranking=method_ranking,
            layer_effectiveness_analysis=layer_ranking,
            task_results=[task_result] if task_result else [],
            optimization_date=datetime.now().isoformat(),
            best_token_aggregation=best_config['token_aggregation'] if best_config else None,
            best_prompt_construction=best_config['prompt_construction'] if best_config else None,
            best_steering_application=best_config['steering_application'] if best_config else None,
            token_aggregation_ranking=token_agg_ranking,
            prompt_construction_ranking=prompt_const_ranking,
            steering_application_ranking=steering_app_ranking
        )

        # Save results
        self._save_steering_optimization_results(summary)

        logger.info(f"\n✅ Full optimization complete!")
        logger.info(f"   Configurations tested: {configurations_tested}")
        logger.info(f"   Best score: {best_score:.3f}")
        if best_config:
            logger.info(f"   Best config: {best_config['method']} L{best_config['layer']} "
                       f"S{best_config['strength']} T:{best_config['token_aggregation']} "
                       f"P:{best_config['prompt_construction']} A:{best_config['steering_application']}")

        return summary

    def _evaluate_full_configuration(
        self,
        task_name: str,
        method: SteeringMethod,
        layer: int,
        strength: float,
        token_aggregation: ExtractionStrategy,
        prompt_construction: ExtractionStrategy,
        steering_application: SteeringApplicationConfig,
        limit: int,
        split_ratio: float
    ) -> float:
        """
        Evaluate a full steering configuration with all parameters.

        Returns effectiveness score (0.0 to 1.0).
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
                'allow_small_dataset': True,
                # New parameters
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

            # Extract score
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
        logger.info(f"🔍 Optimizing steering layer for {task_name} using {steering_method.value}")
        
        if layer_search_range is None:
            # Default: search around classification layer if available
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
            "This requires implementing steering vector training and "
            "effectiveness measurement across different layers. "
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

        Args:
            task_name: Task to optimize for
            steering_method: Steering method to use
            layer: Steering layer to use (defaults to classification layer)
            strength_range: (min_strength, max_strength) to search
            strength_steps: Number of strength values to test
            limit: Maximum samples for testing

        Returns:
            SteeringOptimizationResult with optimal strength
        """
        import time
        start_time = time.time()

        if layer is None:
            if not self.base_classification_layer:
                raise MissingParameterError(
                    params=["layer", "base_classification_layer"],
                    context="Steering strength optimization"
                )
            layer = self.base_classification_layer

        if strength_range is None:
            # Default strength range is reasonable for most steering methods
            strength_range = (0.1, 2.0)

        logger.info(f"⚡ Optimizing steering strength for {task_name}")
        logger.info(f"   Method: {steering_method.value}, Layer: {layer}")
        logger.info(f"   Strength range: {strength_range}, Steps: {strength_steps}")
        
        # Load steering parameters from config
        import json
        import os
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'steering_optimization_parameters.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                steering_config = json.load(f)
        else:
            steering_config = {}
        
        # Get default layer if not provided
        if layer is None:
            layer = self._get_classification_layer(task_name)
            logger.info(f"   Using classification layer: {layer}")
        
        # Default strength range from config or fallback
        if strength_range is None:
            default_strengths = steering_config.get('steering_strengths', {}).get('default', [0.5, 1.0, 1.5, 2.0])
            strength_range = (min(default_strengths), max(default_strengths))
        
        # Generate strength values to test
        import numpy as np
        strengths = np.linspace(strength_range[0], strength_range[1], strength_steps)
        
        # Get method-specific parameters from config if not provided
        if method_params is None:
            method_configs = steering_config.get('steering_methods', [])
            for config in method_configs:
                if config['method'] == steering_method.value and config['name'] == steering_method.value:
                    method_params = config.get('params', {})
                    break
            if method_params is None:
                method_params = {}
        
        results = []
        best_score = -float('inf')
        best_strength = 0.0
        
        logger.info(f"   Testing {len(strengths)} strength values...")
        
        # Test each strength value
        for strength in strengths:
            try:
                # Run evaluation with this strength
                from wisent.cli import run_task_pipeline

                # Build kwargs for run_task_pipeline
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
                    'output_mode': "likelihoods",  # Get likelihoods for evaluation
                    'allow_small_dataset': True
                }
                
                # Add method-specific parameters
                for param, value in method_params.items():
                    pipeline_kwargs[param] = value
                
                result = run_task_pipeline(**pipeline_kwargs)
                
                # Extract score from evaluation results
                score = 0.0
                steering_effect = 0.0
                accuracy = 0.0
                
                if isinstance(result, dict):
                    # Get evaluation results - try both nested and direct access
                    eval_results = None
                    
                    # First try: result[task_name]['evaluation_results']
                    task_result = result.get(task_name, {})
                    if 'evaluation_results' in task_result:
                        eval_results = task_result['evaluation_results']
                    # Second try: result['evaluation_results'] (direct from run_task_pipeline)
                    elif 'evaluation_results' in result:
                        eval_results = result['evaluation_results']
                    else:
                        eval_results = {}
                    
                    # Get accuracy score (but don't use it as the primary metric)
                    accuracy = eval_results.get('accuracy', 0.0)
                    if isinstance(accuracy, str) or accuracy is None:
                        accuracy = 0.0
                    
                    # Calculate steering effect from likelihood changes
                    baseline_likes = eval_results.get('baseline_likelihoods', [])
                    steered_likes = eval_results.get('steered_likelihoods', [])
                    
                    if self.verbose:
                        logger.debug(f"   Found {len(baseline_likes)} baseline and {len(steered_likes)} steered likelihoods")
                        if baseline_likes and len(baseline_likes) > 0:
                            logger.debug(f"   First few baseline likes: {baseline_likes[:3]}")
                            logger.debug(f"   First few steered likes: {steered_likes[:3]}")
                        logger.debug(f"   Full eval_results keys: {list(eval_results.keys())}")
                        logger.debug(f"   Accuracy value: {eval_results.get('accuracy')}")
                        
                        # Check if we're getting the right data structure
                        if isinstance(result, dict):
                            logger.debug(f"   Result keys: {list(result.keys())}")
                            if 'evaluation_results' in result:
                                logger.debug(f"   Direct evaluation_results found")
                    
                    if baseline_likes and steered_likes:
                        # Filter out inf and nan values
                        valid_pairs = []
                        for b, s in zip(baseline_likes, steered_likes):
                            if np.isfinite(b) and np.isfinite(s):
                                valid_pairs.append((b, s))
                        
                        if valid_pairs:
                            changes = [abs(s - b) for b, s in valid_pairs]
                            steering_effect = sum(changes) / len(changes) if changes else 0.0
                            
                            # Cap steering effect to prevent infinity
                            steering_effect = min(steering_effect, 100.0)
                            
                            # Also calculate how many preferences changed
                            preference_changes = 0
                            for i in range(0, len(baseline_likes), 2):  # Assuming binary choices
                                if i+1 < len(baseline_likes):
                                    if np.isfinite(baseline_likes[i]) and np.isfinite(baseline_likes[i+1]) and \
                                       np.isfinite(steered_likes[i]) and np.isfinite(steered_likes[i+1]):
                                        baseline_pref = 0 if baseline_likes[i] > baseline_likes[i+1] else 1
                                        steered_pref = 0 if steered_likes[i] > steered_likes[i+1] else 1
                                        if baseline_pref != steered_pref:
                                            preference_changes += 1
                            
                            # Use steering effect as the primary score
                            score = steering_effect
                            
                            # Add bonus if accuracy is valid and good
                            if np.isfinite(accuracy) and accuracy > 0.5:
                                score += accuracy * 0.5
                        else:
                            # No valid likelihood pairs
                            score = 0.0
                            steering_effect = 0.0
                    else:
                        # Fallback to accuracy if no likelihood data
                        score = accuracy if np.isfinite(accuracy) else 0.0
                
                results.append({
                    'strength': float(strength),
                    'score': score,
                    'steering_effect': steering_effect,
                    'evaluation_results': eval_results if isinstance(result, dict) else {}
                })
                
                if score > best_score:
                    best_score = score
                    best_strength = float(strength)
                
                logger.info(f"   Strength {strength:.2f}: score={score:.3f}, effect={steering_effect:.3f}, accuracy={accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"   Error testing strength {strength}: {e}")
                results.append({
                    'strength': float(strength),
                    'score': 0.0,
                    'error': str(e)
                })

        # Calculate optimization time
        optimization_time = time.time() - start_time

        return SteeringOptimizationResult(
            task_name=task_name,
            best_steering_layer=layer,
            best_steering_method=steering_method.value,
            best_steering_strength=best_strength,
            optimal_parameters={'strength': best_strength},
            steering_effectiveness_score=best_score,
            classification_accuracy_impact=best_score,  # Using same score for now
            optimization_time_seconds=optimization_time,
            total_configurations_tested=len(results),
            error_message=None
        )
    
    def optimize_method_specific_parameters(
        self,
        task_name: str,
        steering_method: SteeringMethod,
        base_layer: Optional[int] = None,
        base_strength: float = 1.0,
        limit: int = 100
    ) -> SteeringOptimizationResult:
        """
        Optimize method-specific parameters for a steering approach.
        
        Args:
            task_name: Task to optimize for
            steering_method: Specific steering method to optimize
            base_layer: Base steering layer to use
            base_strength: Base steering strength to use
            limit: Maximum samples for testing
            
        Returns:
            SteeringOptimizationResult with optimized method parameters
        """
        logger.info(f"🔧 Optimizing {steering_method.value}-specific parameters for {task_name}")
        
        if steering_method == SteeringMethod.CAA:
            return self._optimize_caa_parameters(task_name, base_layer, base_strength, limit)
        else:
            raise SteeringMethodUnknownError(method=str(steering_method))
    
    def _optimize_caa_parameters(
        self, 
        task_name: str, 
        layer: Optional[int], 
        strength: float, 
        limit: int
    ) -> SteeringOptimizationResult:
        """Optimize CAA (Concept Activation Analysis) specific parameters."""
        # CAA typically doesn't have many hyperparameters beyond layer/strength
        # but may include normalization options, vector aggregation methods, etc.
        # For now, return default parameters as CAA is relatively simple
        return SteeringOptimizationResult(
            method=SteeringMethod.CAA,
            layer=layer if layer is not None else 15,
            strength=strength,
            method_specific_params={"normalize": True},
            performance_metrics={"baseline": True}
        )
    
    def run_comprehensive_steering_optimization(
        self,
        tasks: Optional[List[str]] = None,
        methods: Optional[List[SteeringMethod]] = None,
        limit: int = 100,
        max_time_per_task_minutes: float = 20.0,
        save_results: bool = True
    ) -> SteeringOptimizationSummary:
        """
        Run comprehensive steering optimization across multiple tasks and methods.
        
        Args:
            tasks: List of tasks to optimize (if None, uses classification-optimized tasks)
            methods: List of steering methods to test
            limit: Sample limit per task
            max_time_per_task_minutes: Time limit per task
            save_results: Whether to save results to config
            
        Returns:
            SteeringOptimizationSummary with comprehensive results
        """
        logger.info(f"🚀 Starting comprehensive steering optimization")
        
        if tasks is None:
            # Use tasks that were successfully optimized for classification
            if self.classification_config:
                task_overrides = self.classification_config.get("task_specific_overrides", {})
                tasks = list(task_overrides.keys())
                if not tasks:
                    raise InsufficientDataError(
                        reason="No classification-optimized tasks found in classification_config. "
                        "Run classification optimization first or provide explicit tasks."
                    )
            else:
                raise MissingParameterError(
                    params=["tasks", "classification_config"],
                    context="comprehensive steering optimization"
                )
        
        if methods is None:
            methods = [SteeringMethod.CAA]  # CAA is the only supported method
        
        logger.info(f"📊 Tasks: {tasks}")
        logger.info(f"🔧 Methods: {[method.value for method in methods]}")
        
        # Run optimization for each task
        all_results = []
        for task in tasks:
            for method in methods:
                try:
                    result = self._optimize_caa_parameters(task, None, 1.0, limit)
                    all_results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to optimize {method.value} for {task}: {e}")
        
        return SteeringOptimizationSummary(
            task_results={task: all_results for task in tasks},
            best_overall_config=all_results[0] if all_results else None,
            optimization_metadata={"tasks": tasks, "methods": [m.value for m in methods]}
        )
    
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
        method: SteeringMethod,
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
            # Import CLI runner to test configuration
            from wisent.cli import run_task_pipeline
            
            # Prepare kwargs with method-specific parameters
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
                # Map parameter names to CLI argument names
                param_mapping = {
                    'normalization_method': 'normalization_method',
                }
                
                for param_key, param_value in method_params.items():
                    if param_key in param_mapping:
                        kwargs[param_mapping[param_key]] = param_value
            
            # Run steering evaluation
            result = run_task_pipeline(**kwargs)
            
            # Extract evaluation score
            # Priority: accuracy > likelihood change > 0.0
            if 'accuracy' in result and result['accuracy'] != 'N/A':
                return float(result['accuracy'])
            elif 'evaluation_results' in result:
                eval_results = result['evaluation_results']
                if 'accuracy' in eval_results and eval_results['accuracy'] != 'N/A':
                    return float(eval_results['accuracy'])
                # Could also use likelihood changes as a metric
            
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
        
        # Save overall best configuration
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
        
        # Update configuration
        self.config_manager.update_model_config(self.model_name, config)
        logger.info(f"✅ Steering optimization results saved for {self.model_name}")
    
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
        steering_method: SteeringMethod,
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
        # Use the internal evaluation method
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
            'accuracy': score,  # For now, use the same score
            'consistency': 1.0 if score > 0.5 else 0.5,
            'direction_accuracy': score
        }


# Convenience functions for CLI integration
def run_steering_optimization(
    model_name: str,
    optimization_type: str = "auto",
    task_name: str = None,
    limit: int = 100,
    device: str = None,
    verbose: bool = False,
    use_classification_config: bool = True,
    **kwargs
) -> Union[SteeringOptimizationResult, SteeringOptimizationSummary, Dict[str, Any]]:
    """
    Convenience function to run steering optimization.
    
    Args:
        model_name: Model to optimize steering for
        optimization_type: Type of optimization ("auto", "method_comparison", "layer", "strength", "comprehensive")
        task_name: Task to optimize for (if None and optimization_type="auto", uses all classification-optimized tasks)
        limit: Sample limit
        device: Device to use
        verbose: Enable verbose logging
        use_classification_config: Whether to use existing classification config as starting point
        **kwargs: Additional arguments for specific optimization types
        
    Returns:
        SteeringOptimizationResult, SteeringOptimizationSummary, or auto-optimization results
    """
    optimizer = SteeringOptimizer(
        model_name=model_name,
        device=device,
        verbose=verbose
    )
    
    if optimization_type == "auto":
        # Automatic optimization based on classification config
        return run_auto_steering_optimization(
            model_name=model_name,
            task_name=task_name,
            limit=limit,
            device=device,
            verbose=verbose,
            use_classification_config=use_classification_config,
            **kwargs
        )
    elif optimization_type == "method_comparison":
        if not task_name:
            raise MissingParameterError(params=["task_name"], context="method comparison")
        return optimizer.optimize_steering_method_comparison(
            task_name=task_name,
            limit=limit,
            **kwargs
        )
    elif optimization_type == "layer":
        if not task_name:
            raise MissingParameterError(params=["task_name"], context="layer optimization")
        
        # Convert string steering_method to enum if needed
        if 'steering_method' in kwargs and isinstance(kwargs['steering_method'], str):
            kwargs['steering_method'] = SteeringMethod[kwargs['steering_method']]
        
        return optimizer.optimize_steering_layer(
            task_name=task_name,
            limit=limit,
            **kwargs
        )
    elif optimization_type == "strength":
        if not task_name:
            raise MissingParameterError(params=["task_name"], context="strength optimization")
        
        # Convert string steering_method to enum if needed
        if 'steering_method' in kwargs and isinstance(kwargs['steering_method'], str):
            kwargs['steering_method'] = SteeringMethod[kwargs['steering_method']]
        
        return optimizer.optimize_steering_strength(
            task_name=task_name,
            limit=limit,
            **kwargs
        )
    elif optimization_type == "comprehensive":
        return optimizer.run_comprehensive_steering_optimization(
            limit=limit,
            **kwargs
        )
    else:
        raise UnknownTypeError(entity_type="optimization_type", value=optimization_type, valid_values=["method_comparison", "layer", "strength", "comprehensive"])


def run_auto_steering_optimization(
    model_name: str,
    task_name: Optional[str] = None,
    limit: int = 100,
    device: str = None,
    verbose: bool = False,
    use_classification_config: bool = True,
    max_time_minutes: float = 60.0,
    methods_to_test: Optional[List[str]] = None,
    strength_range: Optional[List[float]] = None,
    layer_range: Optional[str] = None
) -> Dict[str, Any]:
    """
    Automatically optimize steering configuration using repscan geometry analysis.

    This function uses repscan metrics (linear probe accuracy, signal strength, ICD)
    to automatically select the best steering method (CAA, PRISM, or TITAN).

    Args:
        model_name: Model to optimize
        task_name: Specific task to optimize (required)
        limit: Sample limit per evaluation
        device: Device to use
        verbose: Enable verbose logging
        use_classification_config: Use classification layer as starting point
        max_time_minutes: Maximum time for optimization (unused - repscan is fast)
        methods_to_test: Ignored - method is auto-selected via repscan
        strength_range: Ignored - strength is determined by method
        layer_range: Explicit layer range to search (e.g. "0-5" or "0,2,4")

    Returns:
        Dictionary with optimization results including recommended method and configuration
    """
    import torch
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.extraction_strategy import ExtractionStrategy
    from wisent.core.geometry import (
        compute_geometry_metrics,
        compute_recommendation,
        compute_concept_coherence,
    )

    if not task_name:
        return {"error": "Task name is required for auto steering optimization"}

    if verbose:
        print("\n" + "=" * 70)
        print("🔍 AUTO STEERING OPTIMIZATION (repscan)")
        print("=" * 70)
        print(f"   Model: {model_name}")
        print(f"   Task: {task_name}")
        print("   Method: Geometry-based selection (not grid search)")
        print("=" * 70 + "\n")

    # Step 1: Load model
    if verbose:
        print("Loading model...", flush=True)

    wisent_model = WisentModel(model_name, device=device)

    if verbose:
        print(f"✓ Model loaded with {wisent_model.num_layers} layers\n")

    # Step 2: Generate contrastive pairs
    if verbose:
        print(f"Generating contrastive pairs for {task_name}...", flush=True)

    pairs = _generate_pairs_for_repscan(task_name, limit)

    if not pairs or len(pairs) < 10:
        return {"error": f"Could not generate enough pairs for {task_name} (got {len(pairs) if pairs else 0})"}

    if verbose:
        print(f"✓ Generated {len(pairs)} contrastive pairs\n")

    # Step 3: Collect activations at analysis layer (75% depth)
    if verbose:
        print("Collecting activations for geometry analysis...", flush=True)

    num_layers = wisent_model.num_layers
    analysis_layer = str(int(num_layers * 0.75))

    collector = ActivationCollector(model=wisent_model)
    sample_pairs = pairs[:min(50, len(pairs))]

    pos_activations = []
    neg_activations = []

    for pair in sample_pairs:
        enriched = collector.collect(
            pair,
            strategy=ExtractionStrategy.CHAT_LAST,
            layers=[analysis_layer]
        )

        if enriched.positive_response.layers_activations.get(analysis_layer) is not None:
            pos_activations.append(enriched.positive_response.layers_activations[analysis_layer])
        if enriched.negative_response.layers_activations.get(analysis_layer) is not None:
            neg_activations.append(enriched.negative_response.layers_activations[analysis_layer])

    if len(pos_activations) < 10 or len(neg_activations) < 10:
        if verbose:
            print("⚠️  Insufficient activations for analysis, defaulting to TITAN")
        return {
            'model_name': model_name,
            'task_name': task_name,
            'recommended_method': 'TITAN',
            'confidence': 0.5,
            'reasoning': 'Insufficient activations for geometry analysis',
            'optimization_date': datetime.now().isoformat(),
        }

    if verbose:
        print(f"✓ Collected {len(pos_activations)} positive and {len(neg_activations)} negative activations\n")

    # Step 4: Run repscan geometry analysis
    if verbose:
        print("Running repscan geometry analysis...", flush=True)

    pos_tensor = torch.stack(pos_activations)
    neg_tensor = torch.stack(neg_activations)

    metrics = compute_geometry_metrics(
        pos_tensor, neg_tensor,
        include_expensive=False,
        n_folds=3,
    )

    # Get recommendation from repscan
    recommendation = compute_recommendation(metrics)
    recommended_method = recommendation.get("recommended_method", "TITAN").upper()
    confidence = recommendation.get("confidence", 0.5)
    reasoning = recommendation.get("reasoning", "")

    # Also compute coherence for more detail
    coherence = compute_concept_coherence(pos_tensor, neg_tensor)

    if verbose:
        print(f"\n   Repscan Analysis Results:")
        print(f"   ├─ Linear probe accuracy: {metrics.get('linear_probe_accuracy', 0):.3f}")
        print(f"   ├─ Signal strength:       {metrics.get('signal_strength', 0):.3f}")
        print(f"   ├─ Concept coherence:     {coherence:.3f}")
        print(f"   ├─ Steerability score:    {metrics.get('steer_steerability_score', 0):.3f}")
        print(f"   ├─ ICD:                   {metrics.get('icd_icd', 0):.1f}")
        print(f"   └─ Recommendation:        {recommended_method} (confidence={confidence:.2f})")
        print(f"       Reasoning: {reasoning}")

    # Step 5: Determine layer and strength search space
    if layer_range:
        # Parse provided layer range
        if '-' in layer_range:
            start, end = map(int, layer_range.split('-'))
            layers_to_test = list(range(start, end + 1))
        elif ',' in layer_range:
            layers_to_test = [int(x.strip()) for x in layer_range.split(',')]
        else:
            layers_to_test = [int(layer_range)]
    else:
        # Default: test layers in upper half of model (where steering is most effective)
        mid_layer = num_layers // 2
        layers_to_test = list(range(mid_layer, num_layers))

    # Strength search space
    if strength_range is None:
        strength_range = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    if verbose:
        print(f"\n🔍 GRID SEARCH for {recommended_method}")
        print(f"   Layers to test: {layers_to_test}")
        print(f"   Strengths to test: {strength_range}")
        total_combos = len(layers_to_test) * len(strength_range)
        print(f"   Total combinations: {total_combos}")

    # Step 6: Train steering vectors for all layers
    if verbose:
        print(f"\n🎯 Training {recommended_method} steering vectors...")

    steering_result = _train_recommended_method(
        wisent_model=wisent_model,
        pairs=pairs,
        method=recommended_method,
        layer=layers_to_test[0],  # Layer param not used for multi-layer training
        verbose=verbose,
    )

    # Step 7: Grid search over layers and strengths
    from wisent.core.evaluators.rotator import EvaluatorRotator
    from wisent.core.models.inference_config import get_generate_kwargs

    if verbose:
        print(f"\n📊 Evaluating {len(layers_to_test) * len(strength_range)} configurations...")

    # Split pairs into train/eval
    train_pairs = pairs[:len(pairs)//2]
    eval_pairs = pairs[len(pairs)//2:]

    if len(eval_pairs) < 10:
        eval_pairs = pairs  # Use all if not enough

    # Initialize evaluator
    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.oracles")
    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
    evaluator = EvaluatorRotator(evaluator=None, task_name=task_name, autoload=False)

    grid_results = []
    best_score = -1
    best_layer = layers_to_test[0]
    best_strength = 1.0

    combo_idx = 0
    total_combos = len(layers_to_test) * len(strength_range)

    for layer in layers_to_test:
        # TITAN/PRISM use string layer numbers, CAA uses layer_N format
        layer_key_simple = str(layer)
        layer_key_prefixed = f"layer_{layer}"

        # Get steering vector for this layer
        if recommended_method == "CAA":
            if hasattr(steering_result.get('result'), 'directions'):
                directions = steering_result['result'].directions
                # Try both key formats
                if layer_key_prefixed in directions:
                    steering_vector = directions[layer_key_prefixed]
                elif layer_key_simple in directions:
                    steering_vector = directions[layer_key_simple]
                else:
                    continue
            else:
                continue
        elif recommended_method == "TITAN":
            result = steering_result.get('result')
            if hasattr(result, 'directions'):
                # TITAN uses simple layer numbers as keys
                if layer_key_simple in result.directions:
                    dirs = result.directions[layer_key_simple]
                    weights = result.direction_weights[layer_key_simple]
                    weights_norm = weights / (weights.sum() + 1e-8)
                    steering_vector = (dirs * weights_norm.unsqueeze(-1)).sum(dim=0)
                elif layer_key_prefixed in result.directions:
                    dirs = result.directions[layer_key_prefixed]
                    weights = result.direction_weights[layer_key_prefixed]
                    weights_norm = weights / (weights.sum() + 1e-8)
                    steering_vector = (dirs * weights_norm.unsqueeze(-1)).sum(dim=0)
                else:
                    continue
            else:
                continue
        elif recommended_method == "PRISM":
            result = steering_result.get('result')
            if hasattr(result, 'directions'):
                if layer_key_simple in result.directions:
                    steering_vector = result.directions[layer_key_simple][0]
                elif layer_key_prefixed in result.directions:
                    steering_vector = result.directions[layer_key_prefixed][0]
                else:
                    continue
            else:
                continue
        else:
            continue

        steering_vector = steering_vector / (steering_vector.norm() + 1e-8)

        for strength in strength_range:
            combo_idx += 1

            # Apply steering
            wisent_model.set_steering_from_raw(
                {str(layer): steering_vector},
                scale=strength,
                normalize=False
            )

            # Evaluate on subset
            correct = 0
            total = 0
            eval_subset = eval_pairs[:min(30, len(eval_pairs))]  # Limit for speed

            for pair in eval_subset:
                messages = [{"role": "user", "content": pair.prompt}]
                response = wisent_model.generate(
                    [messages],
                    **get_generate_kwargs(max_new_tokens=256),
                )[0]

                # Evaluate
                eval_kwargs = {
                    'response': response,
                    'expected': pair.positive_response.model_response,
                    'question': pair.prompt,
                    'choices': [pair.negative_response.model_response, pair.positive_response.model_response],
                    'task_name': task_name,
                }
                if hasattr(pair, 'metadata') and pair.metadata:
                    for key, value in pair.metadata.items():
                        if value is not None and key not in eval_kwargs:
                            eval_kwargs[key] = value

                result = evaluator.evaluate(**eval_kwargs)
                if result.ground_truth == "TRUTHFUL":
                    correct += 1
                total += 1

            # Clear steering
            wisent_model.clear_steering()

            score = correct / total if total > 0 else 0
            grid_results.append({
                'layer': layer,
                'strength': strength,
                'score': score,
                'correct': correct,
                'total': total,
            })

            if verbose:
                bar = "█" * int(score * 20)
                print(f"   [{combo_idx:3d}/{total_combos}] L{layer:2d} S{strength:.2f}: {score:.3f} {bar}")

            if score > best_score:
                best_score = score
                best_layer = layer
                best_strength = strength

    if verbose:
        print(f"\n   ✓ Best: Layer {best_layer}, Strength {best_strength:.2f}, Score {best_score:.3f}")

    optimal_layer = best_layer
    optimal_strength = best_strength

    # Step 8: Save configuration
    config_manager = ModelConfigManager()
    config = config_manager.load_model_config(model_name) or {
        'model_name': model_name,
        'created_date': datetime.now().isoformat(),
        'config_version': '2.0'
    }

    # Add steering optimization results
    if 'steering_optimization' not in config:
        config['steering_optimization'] = {}

    config['steering_optimization']['best_method'] = recommended_method
    config['steering_optimization']['best_layer'] = optimal_layer
    config['steering_optimization']['best_strength'] = optimal_strength
    config['steering_optimization']['best_score'] = best_score
    config['steering_optimization']['optimization_date'] = datetime.now().isoformat()
    config['steering_optimization']['repscan_metrics'] = {
        'linear_probe_accuracy': metrics.get('linear_probe_accuracy', 0),
        'signal_strength': metrics.get('signal_strength', 0),
        'steerability_score': metrics.get('steer_steerability_score', 0),
        'icd': metrics.get('icd_icd', 0),
        'concept_coherence': coherence,
    }
    config['steering_optimization']['confidence'] = confidence
    config['steering_optimization']['reasoning'] = reasoning
    config['steering_optimization']['grid_search'] = {
        'layers_tested': layers_to_test,
        'strengths_tested': strength_range,
        'total_combinations': len(grid_results),
        'all_results': grid_results,
    }

    # Save task-specific results
    if 'task_specific_steering' not in config:
        config['task_specific_steering'] = {}

    config['task_specific_steering'][task_name] = {
        'method': recommended_method,
        'layer': optimal_layer,
        'strength': optimal_strength,
        'score': best_score,
        'confidence': confidence,
        'repscan_metrics': config['steering_optimization']['repscan_metrics'],
    }

    config_manager.save_model_config(model_name, **config)

    if verbose:
        print(f"\n✅ Steering optimization complete!")
        print(f"   Method: {recommended_method} (selected by repscan)")
        print(f"   Best Layer: {optimal_layer}")
        print(f"   Best Strength: {optimal_strength}")
        print(f"   Best Score: {best_score:.3f}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Config saved to: {config_manager._get_config_path(model_name)}")

    return {
        'model_name': model_name,
        'task_name': task_name,
        'recommended_method': recommended_method,
        'optimal_layer': optimal_layer,
        'optimal_strength': optimal_strength,
        'best_score': best_score,
        'confidence': confidence,
        'reasoning': reasoning,
        'repscan_metrics': {
            'linear_probe_accuracy': metrics.get('linear_probe_accuracy', 0),
            'signal_strength': metrics.get('signal_strength', 0),
            'steerability_score': metrics.get('steer_steerability_score', 0),
            'icd': metrics.get('icd_icd', 0),
            'concept_coherence': coherence,
        },
        'grid_search_results': grid_results,
        'steering_result': steering_result,
        'optimization_date': datetime.now().isoformat(),
        'config_saved': True,
    }


def _generate_pairs_for_repscan(task_name: str, limit: int):
    """Generate contrastive pairs for repscan analysis."""
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import build_contrastive_pairs

    try:
        pairs = build_contrastive_pairs(
            task_name=task_name,
            limit=limit,
        )
        return pairs
    except Exception as e:
        logger.error(f"Failed to generate pairs for {task_name}: {e}")
        return []


def _train_recommended_method(wisent_model, pairs, method: str, layer: int, verbose: bool = False):
    """Train the recommended steering method."""
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.extraction_strategy import ExtractionStrategy

    # Get all layers for training
    num_layers = wisent_model.num_layers
    all_layers = [str(i) for i in range(1, num_layers + 1)]

    # Collect activations for all pairs
    if verbose:
        print(f"   Collecting activations for {len(pairs)} pairs...")

    collector = ActivationCollector(model=wisent_model)
    enriched_pairs = []

    for i, pair in enumerate(pairs):
        enriched = collector.collect(pair, strategy=ExtractionStrategy.CHAT_LAST, layers=all_layers)
        enriched_pairs.append(enriched)
        if verbose and (i + 1) % 25 == 0:
            print(f"     {i + 1}/{len(pairs)} pairs processed")

    pair_set = ContrastivePairSet(pairs=enriched_pairs, name=f"{method.lower()}_training")

    if verbose:
        print(f"   ✓ Collected activations for {len(enriched_pairs)} pairs")

    # Train based on method
    if method == "CAA":
        from wisent.core.steering_methods.methods.caa import CAAMethod

        caa_method = CAAMethod()
        result = caa_method.train(pair_set)

        if verbose:
            print(f"   ✓ CAA trained on {len(pairs)} pairs")
            print(f"     Layers: {len(result.directions)}")

        return {"method": "CAA", "layers": len(result.directions), "result": result}

    elif method == "TITAN":
        from wisent.core.steering_methods.methods.titan import TITANMethod

        layer_indices = [int(l) for l in all_layers]
        titan_method = TITANMethod(
            model=wisent_model,
            num_directions=8,
            manifold_method="pca",
            steering_layers=layer_indices,
            sensor_layer=layer_indices[0],
        )

        result = titan_method.train_titan(pair_set)

        if verbose:
            print(f"   ✓ TITAN trained on {len(pairs)} pairs")
            print(f"     Layers: {len(result.layer_order)}")
            print(f"     Directions per layer: {result.directions[result.layer_order[0]].shape[0]}")

        return {"method": "TITAN", "layers": len(result.layer_order), "result": result}

    elif method == "PRISM":
        from wisent.core.steering_methods.methods.prism import PRISMMethod

        prism_method = PRISMMethod(
            model=wisent_model.hf_model,
            num_directions=3,
        )

        result = prism_method.train(pair_set)

        if verbose:
            num_dirs = next(iter(result.directions.values())).shape[0]
            print(f"   ✓ PRISM trained on {len(pairs)} pairs")
            print(f"     Layers: {len(result.directions)}")
            print(f"     Directions per layer: {num_dirs}")

        return {"method": "PRISM", "layers": len(result.directions), "result": result}

    elif method == "PULSE":
        from wisent.core.steering_methods.methods.pulse import PULSEMethod

        layer_indices = [int(l) for l in all_layers]
        pulse_method = PULSEMethod(
            model=wisent_model.hf_model,
            steering_layers=layer_indices,
            sensor_layer=layer_indices[0],
        )

        result = pulse_method.train_pulse(pair_set)

        if verbose:
            print(f"   ✓ PULSE trained on {len(pairs)} pairs")
            print(f"     Layers: {len(result.behavior_vectors)}")
            print(f"     Optimal threshold: {result.optimal_threshold:.3f}")

        return {"method": "PULSE", "layers": len(result.behavior_vectors), "result": result}

    else:
        logger.warning(f"Unknown method {method}, falling back to CAA")
        from wisent.core.steering_methods.methods.caa import CAAMethod

        caa_method = CAAMethod()
        result = caa_method.train(pair_set)

        return {"method": "CAA", "layers": len(result.directions), "result": result}


def get_optimal_steering_params(
    model_name: str,
    task_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get optimal steering parameters for a model/task.
    
    Args:
        model_name: Model name
        task_name: Optional task name for task-specific params
        
    Returns:
        Dictionary with steering parameters or None
    """
    optimizer = SteeringOptimizer(model_name)
    return optimizer.load_optimal_steering_config(task_name)