import logging
import itertools
import copy
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import numpy as np

from .contrastive_pairs import ContrastivePairSet
from .steering import SteeringMethod, SteeringType
from .activations.activations_collector import ActivationCollector
from .activations.extraction_strategy import ExtractionStrategy

from wisent.core.utils.infra_tools.errors import OptimizationError, NoActivationDataError, InsufficientDataError
from wisent.core.utils.config_tools.constants import PROGRESS_LOG_INTERVAL_20
from wisent.core.utils.services.optimization._hyperparameter_evaluate import (
    HyperparameterEvaluateMixin,
    detect_model_layers,
    get_default_layer_range,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    aggregation_methods: List[str]
    token_targeting_strategies: List[str]
    threshold_range: List[float]
    layer_range: List[int] = None
    prompt_construction_strategies: List[str] = field(default_factory=lambda: [
        "multiple_choice", "role_playing", "direct_completion", "instruction_following"
    ])

    # Classifier types to try
    classifier_types: List[str] = field(default_factory=lambda: ["logistic"])

    # Performance metric to optimize
    metric: Optional[str] = None  # Options: "accuracy", "f1", "precision", "recall", "auc"

    # Cross-validation folds (if 0, uses simple train/val split)
    cv_folds: int = 0

    # Validation split ratio (used when cv_folds is zero) - required
    val_split: float = None

    # (max_combinations removed — all combinations are tested)

    # Random seed for reproducibility
    seed: Optional[int] = None


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""

    best_layer: int
    best_aggregation: str
    best_threshold: float
    best_classifier_type: str
    best_prompt_construction_strategy: str
    best_token_targeting_strategy: str
    best_score: float
    best_metrics: Dict[str, float]

    # All tested combinations and their scores
    all_results: List[Dict[str, Any]] = field(default_factory=list)

    # Configuration used for optimization
    config: OptimizationConfig = None


class HyperparameterOptimizer(HyperparameterEvaluateMixin):
    """Optimizes hyperparameters for the guard system."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        from wisent.core.utils.config_tools.constants import DEFAULT_RANDOM_SEED
        np.random.seed(self.config.seed if self.config.seed is not None else DEFAULT_RANDOM_SEED)
        
    def optimize(
        self, 
        model,
        train_pair_set: ContrastivePairSet,
        test_pair_set: ContrastivePairSet,
        device: str = None,
        verbose: bool = False
    ) -> OptimizationResult:
        """
        Optimize hyperparameters for the guard system.
        
        Args:
            model: The model to use for training
            train_pair_set: Training contrastive pairs
            test_pair_set: Test contrastive pairs for evaluation
            device: Device to run on
            verbose: Whether to print progress
            
        Returns:
            OptimizationResult with best hyperparameters and performance
        """
        
        # Auto-detect layer range if not provided
        layer_range = self.config.layer_range
        if layer_range is None:
            total_layers = detect_model_layers(model)
            layer_range = get_default_layer_range(total_layers, use_all=True)
            if verbose:
                print(f"   • Auto-detected {total_layers} model layers")
                print(f"   • Using all layers for optimization: {layer_range[0]}-{layer_range[-1]}")
        
        if verbose:
            print(f"\n🔍 Starting hyperparameter optimization...")
            print(f"   • Layers to test: {len(layer_range)} (range: {layer_range[0]}-{layer_range[-1]})")
            print(f"   • Aggregation methods: {len(self.config.aggregation_methods)}")
            print(f"   • Prompt construction strategies: {len(self.config.prompt_construction_strategies)}")
            print(f"   • Token targeting strategies: {len(self.config.token_targeting_strategies)}")
            print(f"   • Thresholds: {len(self.config.threshold_range)}")
            print(f"   • Classifier types: {len(self.config.classifier_types)}")
            print(f"   • Optimization metric: {self.config.metric}")

        # Generate all combinations of hyperparameters
        combinations = list(itertools.product(
            layer_range,
            self.config.aggregation_methods,
            self.config.prompt_construction_strategies,
            self.config.token_targeting_strategies,
            self.config.threshold_range,
            self.config.classifier_types
        ))
        
        if verbose:
            print(f"   • Testing {len(combinations)} combinations...")
        
        best_score = -np.inf
        best_result = None
        all_results = []
        
        for i, (layer, aggregation, prompt_strategy, token_strategy, threshold, classifier_type) in enumerate(combinations):
            try:
                if verbose and (i + 1) % PROGRESS_LOG_INTERVAL_20 == 0:
                    print(f"   • Progress: {i + 1}/{len(combinations)} combinations tested")

                # Train and evaluate this combination
                result = self._evaluate_combination(
                    model=model,
                    train_pair_set=train_pair_set,
                    test_pair_set=test_pair_set,
                    layer=layer,
                    aggregation=aggregation,
                    prompt_construction_strategy=prompt_strategy,
                    token_targeting_strategy=token_strategy,
                    threshold=threshold,
                    classifier_type=classifier_type,
                    device=device
                )

                all_results.append(result)

                # Check if this is the best so far
                score = result[self.config.metric]
                if score > best_score:
                    best_score = score
                    best_result = result

                    if verbose:
                        print(f"   • New best: layer={layer}, agg={aggregation}, prompt={prompt_strategy}, token={token_strategy}, thresh={threshold:.2f}, {self.config.metric}={score:.3f}")

            except Exception as e:
                logger.warning(f"Failed to evaluate combination (layer={layer}, agg={aggregation}, prompt={prompt_strategy}, token={token_strategy}, thresh={threshold}, type={classifier_type}): {e}")
                continue
        
        if best_result is None:
            raise OptimizationError(reason="No valid combinations found during optimization")
        
        # Create optimization result
        optimization_result = OptimizationResult(
            best_layer=best_result['layer'],
            best_aggregation=best_result['aggregation'],
            best_threshold=best_result['threshold'],
            best_classifier_type=best_result['classifier_type'],
            best_prompt_construction_strategy=best_result['prompt_construction_strategy'],
            best_token_targeting_strategy=best_result['token_targeting_strategy'],
            best_score=best_result[self.config.metric],
            best_metrics={
                'accuracy': best_result['accuracy'],
                'f1': best_result['f1'],
                'precision': best_result['precision'],
                'recall': best_result['recall'],
                'auc': best_result.get('auc', 0.0)
            },
            all_results=all_results,
            config=self.config
        )

        if verbose:
            print(f"\n✅ Optimization complete!")
            print(f"   • Best layer: {optimization_result.best_layer}")
            print(f"   • Best aggregation: {optimization_result.best_aggregation}")
            print(f"   • Best prompt strategy: {optimization_result.best_prompt_construction_strategy}")
            print(f"   • Best token strategy: {optimization_result.best_token_targeting_strategy}")
            print(f"   • Best threshold: {optimization_result.best_threshold:.2f}")
            print(f"   • Best classifier: {optimization_result.best_classifier_type}")
            print(f"   • Best {self.config.metric}: {optimization_result.best_score:.3f}")
            print(f"   • Tested {len(all_results)} valid combinations")
        
        return optimization_result
    
