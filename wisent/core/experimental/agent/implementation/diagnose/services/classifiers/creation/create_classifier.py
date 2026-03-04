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
from wisent.core.utils.config_tools.constants import COMBO_OFFSET
from wisent.core.utils.infra_tools.errors import ClassifierCreationError, InsufficientDataError

from ...activations import Activations
from ...layer import Layer
from ...model import Model
from ...model_persistence import ModelPersistence, create_classifier_metadata

from ._create_helpers import BenchmarkMixin, DataGenerationMixin, ScoringMixin, TrainingMixin
from ._create_helpers.create_classifier_scoring import ScoringConfig


@dataclass
class TrainingConfig:
    """Configuration for classifier training."""

    issue_type: str
    layer: int
    threshold: float
    training_samples: int
    test_split: float
    classifier_type: Optional[str] = None
    model_name: str = ""
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

    def __init__(self, model: Model, max_tasks_to_process: int, scoring_config: ScoringConfig):
        """Initialize the classifier creator."""
        self.model = model
        self.max_tasks_to_process = max_tasks_to_process
        self.scoring_config = scoring_config

    def create_classifier_for_issue_type(
        self, issue_type: str, layer: int, time_budget_minutes: float,
        task_search_limit: int, *, data_oversample_multiplier: int, config: Optional[TrainingConfig] = None,
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

        if config is None:
            raise ValueError("TrainingConfig is required (threshold and training_samples must be specified)")

        start_time = time.time()

        # Generate training data
        print("   Generating training data...")
        training_data = self._generate_training_data(issue_type, config.training_samples, time_budget_minutes, task_search_limit=task_search_limit, data_oversample_multiplier=data_oversample_multiplier)

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
        self, issue_type: str, layers: List[int], time_budget_minutes: float,
        task_search_limit: int, *, data_oversample_multiplier: int, save_base_path: Optional[str] = None,
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

            result = self.create_classifier_for_issue_type(issue_type, layer, time_budget_minutes, task_search_limit=task_search_limit, data_oversample_multiplier=data_oversample_multiplier, config=config)
            results[layer] = result

        print(f"   Created {len(results)} classifiers across layers {layers}")
        return results

    def optimize_classifier_for_performance(
        self,
        issue_type: str,
        target_metric: str,
        time_budget_minutes: float,
        layer_stride: int,
        task_search_limit: int,
        *,
        data_oversample_multiplier: int,
        layer_range: Tuple[int, int] = None,
        classifier_types: List[str] = None,
        min_target_score: float = None,
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
        if min_target_score is None:
            raise ValueError("min_target_score is required for optimize_classifier_for_performance")
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

        start_layer, end_layer = layer_range
        layers_to_test = range(start_layer, end_layer + COMBO_OFFSET, layer_stride)

        for layer in layers_to_test:
            for classifier_type in classifier_types:
                config = TrainingConfig(
                    issue_type=issue_type,
                    layer=layer,
                    classifier_type=classifier_type,
                    model_name=self.model.name,
                )

                try:
                    result = self.create_classifier_for_issue_type(issue_type, layer, time_budget_minutes, task_search_limit=task_search_limit, data_oversample_multiplier=data_oversample_multiplier, config=config)
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
    model: Model, issue_type: str, time_budget_minutes: float,
    max_tasks_to_process: int, task_search_limit: int, scoring_config: ScoringConfig,
    *, data_oversample_multiplier: int,
    layer: int = None, save_path: str = None, optimize: bool = False,
) -> TrainingResult:
    """Convenience function to create a classifier on demand."""
    creator = ClassifierCreator(model, max_tasks_to_process=max_tasks_to_process, scoring_config=scoring_config)

    if optimize or layer is None:
        # Optimize to find best configuration
        result = creator.optimize_classifier_for_performance(issue_type, target_metric="f1", time_budget_minutes=time_budget_minutes, task_search_limit=task_search_limit, data_oversample_multiplier=data_oversample_multiplier)

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

    return creator.create_classifier_for_issue_type(issue_type, layer, time_budget_minutes, task_search_limit=task_search_limit, data_oversample_multiplier=data_oversample_multiplier, config=config)
