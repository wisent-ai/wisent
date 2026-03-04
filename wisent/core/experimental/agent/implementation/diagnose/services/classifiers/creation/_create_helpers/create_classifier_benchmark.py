"""Benchmark-related methods for ClassifierCreator."""

import random
import time
from typing import Any, Dict, List, Optional

from wisent.core.utils.infra_tools.errors import InsufficientDataError

from ....activations import Activations
from ....model import Model


class BenchmarkMixin:
    """Mixin providing benchmark-based classifier creation methods."""

    model: Model

    async def create_classifier_for_issue_with_benchmarks(
        self,
        issue_type: str,
        relevant_benchmarks: List[str],
        layer: int,
        num_samples: int = None,
        config: Optional["TrainingConfig"] = None,
    ) -> "TrainingResult":
        """
        Create a classifier using specific benchmarks for better contrastive pairs.

        Args:
            issue_type: Type of issue to detect (e.g., "hallucination", "quality")
            relevant_benchmarks: List of benchmark names to use for training data
            layer: Model layer to use for activation extraction (required)
            num_samples: Number of training samples to generate
            config: Optional training configuration

        Returns:
            TrainingResult with the trained classifier and metrics
        """
        from ..create_classifier import TrainingConfig, TrainingResult

        if num_samples is None:
            raise ValueError("num_samples is required for create_classifier_for_issue_with_benchmarks")
        print(f"Creating {issue_type} classifier using benchmarks: {relevant_benchmarks}")

        # Use provided config or create default
        if config is None:
            config = TrainingConfig(
                issue_type=issue_type, layer=layer, model_name=self.model.name, training_samples=num_samples
            )

        start_time = time.time()

        # Generate training data using the provided benchmarks
        print("   Loading benchmark-specific training data...")
        training_data = []

        try:
            # Load data from the relevant benchmarks
            benchmark_data = self._load_benchmark_data(relevant_benchmarks, num_samples)
            training_data.extend(benchmark_data)
            print(f"      Loaded {len(benchmark_data)} examples from benchmarks")
        except Exception as e:
            print(f"      Failed to load benchmark data: {e}")

        # If we don't have enough data from benchmarks, supplement with synthetic data
        if len(training_data) < num_samples // 2:
            print("   Supplementing with synthetic training data...")
            try:
                synthetic_data = self._generate_synthetic_training_data(
                    issue_type, num_samples - len(training_data)
                )
                training_data.extend(synthetic_data)
                print(f"      Added {len(synthetic_data)} synthetic examples")
            except Exception as e:
                print(f"      Failed to generate synthetic data: {e}")

        if not training_data:
            raise InsufficientDataError(reason=f"No training data available for {issue_type}")

        print(f"   Total training examples: {len(training_data)}")

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
            f"   Benchmark-based classifier created in {training_time:.2f}s "
            f"(F1: {metrics.get('f1', 0):.3f}, Accuracy: {metrics.get('accuracy', 0):.3f})"
        )
        print(f"      Used benchmarks: {relevant_benchmarks}")

        return result

    async def create_combined_benchmark_classifier(
        self, benchmark_names: List[str], classifier_params: "ClassifierParams",
        config: Optional["TrainingConfig"] = None,
    ) -> "TrainingResult":
        """
        Create a classifier trained on combined data from multiple benchmarks.

        Args:
            benchmark_names: List of benchmark names to combine training data from
            classifier_params: Model-determined classifier parameters
            config: Optional training configuration

        Returns:
            TrainingResult with the trained combined classifier
        """
        from ..create_classifier import TrainingConfig, TrainingResult

        print(f"Creating combined classifier from {len(benchmark_names)} benchmarks...")
        print(f"   Benchmarks: {benchmark_names}")
        print(f"   Using layer {classifier_params.optimal_layer}, {classifier_params.training_samples} samples")

        # Create config from classifier_params
        if config is None:
            config = TrainingConfig(
                issue_type=f"quality_combined_{'_'.join(sorted(benchmark_names))}",
                layer=classifier_params.optimal_layer,
                classifier_type=classifier_params.classifier_type,
                threshold=classifier_params.classification_threshold,
                training_samples=classifier_params.training_samples,
                model_name=self.model.name,
            )

        start_time = time.time()

        # Generate combined training data from all benchmarks
        print("   Loading and combining benchmark training data...")
        combined_training_data = await self._load_combined_benchmark_data(
            benchmark_names, classifier_params.training_samples
        )

        print(f"   Loaded {len(combined_training_data)} combined training examples")

        # Extract activations
        print("   Extracting activations...")
        harmful_activations, harmless_activations = self._extract_activations_from_data(
            combined_training_data, classifier_params.optimal_layer
        )

        # Train classifier
        print("   Training combined classifier...")
        classifier = self._train_classifier(harmful_activations, harmless_activations, config)

        # Evaluate performance
        print("   Evaluating performance...")
        metrics = self._evaluate_classifier(classifier, harmful_activations, harmless_activations)

        training_time = time.time() - start_time

        # Save classifier if path provided
        save_path = None
        if config.save_path:
            print("   Saving combined classifier...")
            save_path = self._save_classifier(classifier, config, metrics)

        result = TrainingResult(
            classifier=classifier.classifier,
            config=config,
            performance_metrics=metrics,
            training_time=training_time,
            save_path=save_path,
        )

        print(
            f"   Combined classifier created in {training_time:.2f}s "
            f"(F1: {metrics.get('f1', 0):.3f}, Accuracy: {metrics.get('accuracy', 0):.3f})"
        )

        return result

    async def _load_combined_benchmark_data(
        self, benchmark_names: List[str], total_samples: int
    ) -> List[Dict[str, Any]]:
        """
        Load and combine training data from multiple benchmarks.

        Args:
            benchmark_names: List of benchmark names to load data from
            total_samples: Total number of training samples to create

        Returns:
            Combined list of training examples with balanced sampling
        """
        combined_data = []
        samples_per_benchmark = max(1, total_samples // len(benchmark_names))

        print(f"      Loading ~{samples_per_benchmark} samples per benchmark")

        for benchmark_name in benchmark_names:
            try:
                print(f"      Loading data from {benchmark_name}...")
                benchmark_data = self._load_benchmark_data([benchmark_name], samples_per_benchmark)
                combined_data.extend(benchmark_data)
                print(f"         Loaded {len(benchmark_data)} samples from {benchmark_name}")

            except Exception as e:
                print(f"         Failed to load {benchmark_name}: {e}")
                # Continue with other benchmarks
                continue

        # If we don't have enough samples, pad with synthetic data
        if len(combined_data) < total_samples:
            remaining_samples = total_samples - len(combined_data)
            print(f"      Generating {remaining_samples} synthetic samples to reach target")
            synthetic_data = self._generate_synthetic_training_data("quality", remaining_samples)
            combined_data.extend(synthetic_data)

        # Shuffle the combined data to ensure good mixing
        random.shuffle(combined_data)

        # Trim to exact target if we have too many
        combined_data = combined_data[:total_samples]

        print(f"      Final combined dataset: {len(combined_data)} samples")
        return combined_data

    async def create_classifier_for_issue(self, issue_type: str, time_budget_minutes: float, task_search_limit: int, layer: int = None) -> "TrainingResult":
        """
        Create a classifier for an issue type (async version for compatibility).

        Args:
            issue_type: Type of issue to detect
            time_budget_minutes: Time budget in minutes for benchmark discovery
            layer: Model layer to use for activation extraction

        Returns:
            TrainingResult with the trained classifier
        """
        return self.create_classifier_for_issue_type(issue_type, layer, time_budget_minutes, task_search_limit=task_search_limit)
