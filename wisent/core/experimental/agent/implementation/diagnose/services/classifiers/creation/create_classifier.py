"""
On-the-Fly Classifier Creation System for Autonomous Agent

This module handles:
- Dynamic training of new classifiers for specific issue types
- Automatic training data generation for different problem domains
- Classifier optimization and validation
- Integration with the autonomous agent system
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from wisent.core.reading.classifiers.core.atoms import ActivationClassifier, Classifier
from wisent.core.utils.config_tools.constants import CLASSIFIER_TRAINING_SAMPLES, CLASSIFIER_TEST_SIZE, CLASSIFIER_THRESHOLD, CLASSIFIER_MIN_TARGET_SCORE, LAYER_STRIDE_DEFAULT
from wisent.core.utils.infra_tools.errors import ClassifierCreationError, InsufficientDataError

from ...activations import Activations
from ...layer import Layer
from ...model import Model
from ...model_persistence import ModelPersistence, create_classifier_metadata

from ._create_helpers import BenchmarkMixin, DataGenerationMixin, ScoringMixin, TrainingMixin


@dataclass
class TrainingConfig:
    """Configuration for classifier training."""

    issue_type: str
    layer: int
    classifier_type: Optional[str] = None
    threshold: float = CLASSIFIER_THRESHOLD
    model_name: str = ""
    training_samples: int = CLASSIFIER_TRAINING_SAMPLES
    test_split: float = CLASSIFIER_TEST_SIZE
    optimization_metric: Optional[str] = None
    save_path: Optional[str] = None


@dataclass
class TrainingResult:
    """Result of classifier training."""

    classifier: Classifier
    config: TrainingConfig
    performance_metrics: Dict[str, float]
    training_time: float
    save_path: Optional[str] = None


class ClassifierCreator(BenchmarkMixin, DataGenerationMixin, ScoringMixin, TrainingMixin):
    """Creates new classifiers on demand for the autonomous agent."""

    def __init__(self, model: Model):
        """
        Initialize the classifier creator.

        Args:
            model: The language model to use for training
        """
        self.model = model

    def create_classifier_for_issue_type(
        self, issue_type: str, layer: int, config: Optional[TrainingConfig] = None
    ) -> TrainingResult:
        """
        Create a new classifier for a specific issue type.

        Args:
            issue_type: Type of issue to detect (e.g., "hallucination", "quality")
            layer: Model layer to use for activation extraction
            config: Optional training configuration

        Returns:
            TrainingResult with the trained classifier and metrics
        """
        print(f"Creating classifier for {issue_type} at layer {layer}...")

        # Use provided config or create default
        if config is None:
            config = TrainingConfig(issue_type=issue_type, layer=layer, model_name=self.model.name)

        start_time = time.time()

        # Generate training data
        print("   Generating training data...")
        training_data = self._generate_training_data(issue_type, config.training_samples)

        # Extract activations
        print("   Extracting activations...")
        harmful_activations, harmless_activations = self._extract_activations_from_data(training_data, layer)

        # Train classifier
        print("   Training classifier...")
        classifier = self._train_classifier(harmful_activations, harmless_activations, config)

        # Evaluate performance
        print("   Evaluating performance...")
        metrics = self._evaluate_classifier(classifier, harmful_activations, harmless_activations)

        training_time = time.time() - start_time

        # Save classifier if path provided
        save_path = None
        if config.save_path:
            print("   Saving classifier...")
            save_path = self._save_classifier(classifier, config, metrics)

        result = TrainingResult(
            classifier=classifier.classifier,  # Return the base classifier
            config=config,
            performance_metrics=metrics,
            training_time=training_time,
            save_path=save_path,
        )

        print(
            f"   Classifier created in {training_time:.2f}s "
            f"(F1: {metrics.get('f1', 0):.3f}, Accuracy: {metrics.get('accuracy', 0):.3f})"
        )

        return result

    def create_multi_layer_classifiers(
        self, issue_type: str, layers: List[int], save_base_path: Optional[str] = None
    ) -> Dict[int, TrainingResult]:
        """
        Create classifiers for multiple layers for the same issue type.

        Args:
            issue_type: Type of issue to detect
            layers: List of layers to create classifiers for
            save_base_path: Base path for saving classifiers

        Returns:
            Dictionary mapping layer indices to training results
        """
        print(f"Creating multi-layer classifiers for {issue_type}...")

        results = {}

        for layer in layers:
            config = TrainingConfig(
                issue_type=issue_type,
                layer=layer,
                model_name=self.model.name,
                save_path=f"{save_base_path}_layer_{layer}.pkl" if save_base_path else None,
            )

            result = self.create_classifier_for_issue_type(issue_type, layer, config)
            results[layer] = result

        print(f"   Created {len(results)} classifiers across layers {layers}")
        return results

    def optimize_classifier_for_performance(
        self,
        issue_type: str,
        target_metric: str,
        layer_range: Tuple[int, int] = None,
        classifier_types: List[str] = None,
        min_target_score: float = CLASSIFIER_MIN_TARGET_SCORE,
    ) -> TrainingResult:
        """
        Optimize classifier by testing different configurations.

        Args:
            issue_type: Type of issue to detect
            layer_range: Range of layers to test (start, end). If None, auto-detect
            classifier_types: Types of classifiers to test
            target_metric: Metric to optimize for
            min_target_score: Minimum acceptable score

        Returns:
            Best performing classifier configuration
        """
        print(f"Optimizing classifier for {issue_type}...")

        if classifier_types is None:
            classifier_types = ["logistic", "mlp"]

        # Auto-detect layer range if not provided
        if layer_range is None:
            from ..hyperparameter_optimizer import detect_model_layers

            total_layers = detect_model_layers(self.model)
            layer_range = (0, total_layers - 1)
            print(
                f"   Auto-detected {total_layers} layers, "
                f"testing range {layer_range[0]}-{layer_range[1]}"
            )

        best_result = None
        best_score = 0.0

        layers_to_test = range(layer_range[0], layer_range[1] + 1, LAYER_STRIDE_DEFAULT)

        for layer in layers_to_test:
            for classifier_type in classifier_types:
                config = TrainingConfig(
                    issue_type=issue_type,
                    layer=layer,
                    classifier_type=classifier_type,
                    model_name=self.model.name,
                )

                try:
                    result = self.create_classifier_for_issue_type(issue_type, layer, config)
                    score = result.performance_metrics.get(target_metric, 0.0)

                    print(f"      Layer {layer}, {classifier_type}: {target_metric}={score:.3f}")

                    if score > best_score:
                        best_score = score
                        best_result = result

                        # Early stopping if we hit the target
                        if score >= min_target_score:
                            print(f"      Target score reached: {score:.3f}")
                            break

                except Exception as e:
                    print(f"      Failed layer {layer}, {classifier_type}: {e}")
                    continue

            # Break outer loop if target reached
            if best_score >= min_target_score:
                break

        if best_result is None:
            raise ClassifierCreationError(issue_type=issue_type)

        print(
            f"   Best configuration: Layer {best_result.config.layer}, "
            f"{best_result.config.classifier_type}, {target_metric}={best_score:.3f}"
        )

        return best_result


def create_classifier_on_demand(
    model: Model,
    issue_type: str,
    layer: int = None,
    save_path: str = None,
    optimize: bool = False,
) -> TrainingResult:
    """
    Convenience function to create a classifier on demand.

    Args:
        model: Language model to use
        issue_type: Type of issue to detect
        layer: Specific layer to use (auto-optimized if None)
        save_path: Path to save the classifier
        optimize: Whether to optimize for best performance

    Returns:
        TrainingResult with the created classifier
    """
    creator = ClassifierCreator(model)

    if optimize or layer is None:
        # Optimize to find best configuration
        result = creator.optimize_classifier_for_performance(issue_type, target_metric="f1")

        # Save if path provided
        if save_path:
            result.config.save_path = save_path
            result.save_path = creator._save_classifier(
                ActivationClassifier(device=model.device),
                result.config,
                result.performance_metrics,
            )

        return result
    # Use specified layer
    config = TrainingConfig(
        issue_type=issue_type,
        layer=layer,
        save_path=save_path,
        model_name=model.name,
    )

    return creator.create_classifier_for_issue_type(issue_type, layer, config)
