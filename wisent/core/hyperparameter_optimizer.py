import logging
import itertools
import copy
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .contrastive_pairs import ContrastivePairSet
from .steering import SteeringMethod, SteeringType
from .activations.activations_collector import ActivationCollector
from .activations.extraction_strategy import ExtractionStrategy

from wisent.core.errors import OptimizationError, NoActivationDataError, InsufficientDataError

logger = logging.getLogger(__name__)


def detect_model_layers(model) -> int:
    """
    Detect the number of layers in a model.
    
    Args:
        model: The model object to inspect
        
    Returns:
        Number of layers in the model
    """
    try:
        # Try different ways to get layer count based on model architecture
        if hasattr(model, 'hf_model'):
            hf_model = model.hf_model
        else:
            hf_model = model
        
        # Method 1: Check config for common layer count attributes
        if hasattr(hf_model, 'config'):
            config = hf_model.config
            
            # Different models use different names for layer count
            layer_attrs = ['num_hidden_layers', 'n_layer', 'num_layers', 'n_layers']
            for attr in layer_attrs:
                if hasattr(config, attr):
                    layer_count = getattr(config, attr)
                    if isinstance(layer_count, int) and layer_count > 0:
                        logger.info(f"Detected {layer_count} layers from config.{attr}")
                        return layer_count
        
        # Method 2: Count actual layer modules
        if hasattr(hf_model, 'model') and hasattr(hf_model.model, 'layers'):
            # Llama/Mistral style: model.layers
            layer_count = len(hf_model.model.layers)
            logger.info(f"Detected {layer_count} layers from model.layers")
            return layer_count
        elif hasattr(hf_model, 'transformer') and hasattr(hf_model.transformer, 'h'):
            # GPT style: transformer.h
            layer_count = len(hf_model.transformer.h)
            logger.info(f"Detected {layer_count} layers from transformer.h")
            return layer_count
        elif hasattr(hf_model, 'encoder') and hasattr(hf_model.encoder, 'layer'):
            # BERT style: encoder.layer
            layer_count = len(hf_model.encoder.layer)
            logger.info(f"Detected {layer_count} layers from encoder.layer")
            return layer_count
        
        # Method 3: Try to count by iterating through named modules
        layer_count = 0
        for name, _ in hf_model.named_modules():
            # Look for patterns like "layers.0", "h.0", "layer.0", etc.
            if any(pattern in name for pattern in ['.layers.', '.h.', '.layer.']):
                # Extract layer number
                for part in name.split('.'):
                    if part.isdigit():
                        layer_num = int(part)
                        layer_count = max(layer_count, layer_num + 1)
        
        if layer_count > 0:
            logger.info(f"Detected {layer_count} layers from module names")
            return layer_count
        
        # Fallback: Conservative default
        logger.warning("Could not detect layer count, using default of 32")
        return 32
        
    except Exception as e:
        logger.warning(f"Error detecting layer count: {e}, using default of 32")
        return 32


def get_default_layer_range(total_layers: int, use_all: bool = True) -> List[int]:
    """
    Get a reasonable default layer range for optimization.
    
    Args:
        total_layers: Total number of layers in the model
        use_all: If True, use all layers; if False, use middle layers only
        
    Returns:
        List of layer indices to optimize over
    """
    if use_all:
        # Use all layers (0-indexed)
        return list(range(total_layers))
    else:
        # Use middle layers (skip first and last quarter)
        start_layer = max(0, total_layers // 4)
        end_layer = min(total_layers, (3 * total_layers) // 4)
        return list(range(start_layer, end_layer))


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""

    # Layer range to search (will be auto-detected if None)
    layer_range: List[int] = None

    # Token aggregation methods to try
    aggregation_methods: List[str] = field(default_factory=lambda: ["average", "final", "first", "max", "min"])

    # Prompt construction strategies to try
    prompt_construction_strategies: List[str] = field(default_factory=lambda: [
        "multiple_choice", "role_playing", "direct_completion", "instruction_following"
    ])

    # Token targeting strategies to try
    token_targeting_strategies: List[str] = field(default_factory=lambda: [
        "choice_token", "continuation_token", "last_token", "first_token", "mean_pooling"
    ])

    # Threshold range to search (for classification)
    threshold_range: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    # Classifier types to try
    classifier_types: List[str] = field(default_factory=lambda: ["logistic"])

    # Performance metric to optimize
    metric: str = "f1"  # Options: "accuracy", "f1", "precision", "recall", "auc"

    # Cross-validation folds (if 0, uses simple train/val split)
    cv_folds: int = 0

    # Validation split ratio (used when cv_folds=0)
    val_split: float = 0.2

    # Maximum number of combinations to try (for performance)
    max_combinations: int = 1000

    # Random seed for reproducibility
    seed: int = 42


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


class HyperparameterOptimizer:
    """Optimizes hyperparameters for the guard system."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        np.random.seed(self.config.seed)
        
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
        
        # Limit combinations if too many
        if len(combinations) > self.config.max_combinations:
            if verbose:
                print(f"   • Too many combinations ({len(combinations)}), sampling {self.config.max_combinations}")
            # Sample indices since np.random.choice doesn't work on list of tuples
            indices = np.random.choice(
                len(combinations),
                size=self.config.max_combinations,
                replace=False
            )
            combinations = [combinations[i] for i in indices]
        
        if verbose:
            print(f"   • Testing {len(combinations)} combinations...")
        
        best_score = -np.inf
        best_result = None
        all_results = []
        
        for i, (layer, aggregation, prompt_strategy, token_strategy, threshold, classifier_type) in enumerate(combinations):
            try:
                if verbose and (i + 1) % 20 == 0:
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
    
    def _evaluate_combination(
        self,
        model,
        train_pair_set: ContrastivePairSet,
        test_pair_set: ContrastivePairSet,
        layer: int,
        aggregation: str,
        prompt_construction_strategy: str,
        token_targeting_strategy: str,
        threshold: float,
        classifier_type: str,
        device: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single hyperparameter combination.

        Args:
            model: The model to use
            train_pair_set: Training data
            test_pair_set: Test data
            layer: Layer index to use
            aggregation: Token aggregation method
            prompt_construction_strategy: Strategy for constructing prompts
            token_targeting_strategy: Strategy for targeting tokens
            threshold: Classification threshold
            classifier_type: Type of classifier
            device: Device to run on

        Returns:
            Dictionary with evaluation metrics
        """
        import torch

        # Map aggregation string to enum
        aggregation_map = {
            'average': ExtractionStrategy.CHAT_MEAN,
            'first': ExtractionStrategy.CHAT_FIRST,
            'last': ExtractionStrategy.CHAT_LAST,
            'max': ExtractionStrategy.CHAT_MAX_NORM,
        }
        agg_strategy = aggregation_map.get(aggregation, ExtractionStrategy.CHAT_MEAN)

        # Map prompt strategy string to enum
        prompt_strategy_map = {
            'multiple_choice': ExtractionStrategy.MC_BALANCED,
            'role_playing': ExtractionStrategy.ROLE_PLAY,
            'direct_completion': ExtractionStrategy.CHAT_LAST,
            'instruction_following': ExtractionStrategy.CHAT_LAST,
            'chat_template': ExtractionStrategy.CHAT_LAST,
        }
        prompt_strategy = prompt_strategy_map.get(prompt_construction_strategy, ExtractionStrategy.CHAT_LAST)

        # Create activation collector
        collector = ActivationCollector(model=model)
        layer_str = str(layer)

        # Collect activations for training pairs
        train_pos_acts = []
        train_neg_acts = []

        for pair in train_pair_set.pairs:
            updated_pair = collector.collect_for_pair(
                pair,
                layers=[layer_str],
                aggregation=agg_strategy,
                return_full_sequence=False,
                normalize_layers=False,
                prompt_strategy=prompt_strategy
            )

            if updated_pair.positive_response.layers_activations and layer_str in updated_pair.positive_response.layers_activations:
                act = updated_pair.positive_response.layers_activations[layer_str]
                if act is not None:
                    if isinstance(act, torch.Tensor):
                        train_pos_acts.append(act.detach().cpu().numpy().flatten())
                    else:
                        train_pos_acts.append(np.array(act).flatten())

            if updated_pair.negative_response.layers_activations and layer_str in updated_pair.negative_response.layers_activations:
                act = updated_pair.negative_response.layers_activations[layer_str]
                if act is not None:
                    if isinstance(act, torch.Tensor):
                        train_neg_acts.append(act.detach().cpu().numpy().flatten())
                    else:
                        train_neg_acts.append(np.array(act).flatten())

        if not train_pos_acts or not train_neg_acts:
            raise NoActivationDataError()

        # Create training data
        X_train = np.array(train_pos_acts + train_neg_acts)
        y_train = np.array([0] * len(train_pos_acts) + [1] * len(train_neg_acts))

        # Train classifier
        from .classifiers.classifiers.models.logistic import LogisticClassifier
        from .classifiers.classifiers.models.mlp import MLPClassifier

        if classifier_type == 'mlp':
            classifier = MLPClassifier(threshold=threshold, device=device)
        else:
            classifier = LogisticClassifier(threshold=threshold, device=device)

        classifier.fit(X_train, y_train)

        # Collect activations for test pairs and evaluate
        predictions = []
        true_labels = []
        prob_scores = []

        for pair in test_pair_set.pairs:
            updated_pair = collector.collect_for_pair(
                pair,
                layers=[layer_str],
                aggregation=agg_strategy,
                return_full_sequence=False,
                normalize_layers=False,
                prompt_strategy=prompt_strategy
            )

            pos_act = None
            neg_act = None

            if updated_pair.positive_response.layers_activations and layer_str in updated_pair.positive_response.layers_activations:
                act = updated_pair.positive_response.layers_activations[layer_str]
                if act is not None:
                    if isinstance(act, torch.Tensor):
                        pos_act = act.detach().cpu().numpy().flatten()
                    else:
                        pos_act = np.array(act).flatten()

            if updated_pair.negative_response.layers_activations and layer_str in updated_pair.negative_response.layers_activations:
                act = updated_pair.negative_response.layers_activations[layer_str]
                if act is not None:
                    if isinstance(act, torch.Tensor):
                        neg_act = act.detach().cpu().numpy().flatten()
                    else:
                        neg_act = np.array(act).flatten()

            if pos_act is not None:
                pos_prob = classifier.predict_proba([pos_act])
                # predict_proba returns float for single sample, list for multiple
                if isinstance(pos_prob, list):
                    pos_prob = pos_prob[0]
                pos_pred = 1 if pos_prob > threshold else 0
                predictions.append(pos_pred)
                true_labels.append(0)  # Positive = 0 (truthful)
                prob_scores.append(pos_prob)

            if neg_act is not None:
                neg_prob = classifier.predict_proba([neg_act])
                # predict_proba returns float for single sample, list for multiple
                if isinstance(neg_prob, list):
                    neg_prob = neg_prob[0]
                neg_pred = 1 if neg_prob > threshold else 0
                predictions.append(neg_pred)
                true_labels.append(1)  # Negative = 1 (untruthful)
                prob_scores.append(neg_prob)

        if len(predictions) == 0:
            raise InsufficientDataError(reason="No valid predictions generated")

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)

        # Calculate AUC if possible
        try:
            auc = roc_auc_score(true_labels, prob_scores) if len(set(true_labels)) > 1 else 0.0
        except:
            auc = 0.0

        return {
            'layer': layer,
            'aggregation': aggregation,
            'prompt_construction_strategy': prompt_construction_strategy,
            'token_targeting_strategy': token_targeting_strategy,
            'threshold': threshold,
            'classifier_type': classifier_type,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'num_train_samples': len(train_pos_acts) + len(train_neg_acts),
            'num_test_samples': len(predictions)
        }
    
    @staticmethod
    def from_config_dict(config_dict: Dict[str, Any]) -> 'HyperparameterOptimizer':
        """Create optimizer from configuration dictionary."""
        config = OptimizationConfig(**config_dict)
        return HyperparameterOptimizer(config)
    
    def save_results(self, result: OptimizationResult, filepath: str):
        """Save optimization results to file."""
        import json

        # Convert result to serializable format
        result_dict = {
            'best_hyperparameters': {
                'layer': result.best_layer,
                'aggregation': result.best_aggregation,
                'prompt_construction_strategy': result.best_prompt_construction_strategy,
                'token_targeting_strategy': result.best_token_targeting_strategy,
                'threshold': result.best_threshold,
                'classifier_type': result.best_classifier_type
            },
            'best_score': result.best_score,
            'best_metrics': result.best_metrics,
            'optimization_config': {
                'layer_range': self.config.layer_range,
                'aggregation_methods': self.config.aggregation_methods,
                'prompt_construction_strategies': self.config.prompt_construction_strategies,
                'token_targeting_strategies': self.config.token_targeting_strategies,
                'threshold_range': self.config.threshold_range,
                'classifier_types': self.config.classifier_types,
                'metric': self.config.metric,
                'max_combinations': self.config.max_combinations
            },
            'all_results': result.all_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Optimization results saved to {filepath}") 