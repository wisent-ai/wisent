"""Training pipeline methods for ClassifierCreator."""

from typing import Any, Dict, List, Tuple

from wisent.core.reading.classifiers.core.atoms import ActivationClassifier
from wisent.core.utils.infra_tools.errors import InsufficientDataError
from wisent.core.utils.config_tools.constants import DATA_OVERSAMPLE_MULTIPLIER, SPLIT_RATIO_FULL

from ....activations import Activations
from ....layer import Layer
from ....model import Model
from ....model_persistence import ModelPersistence, create_classifier_metadata


class TrainingMixin:
    """Mixin providing training pipeline methods for classifier creation."""

    model: Model

    def _load_benchmark_data(self, benchmarks: List[str], num_samples: int) -> List[Dict[str, Any]]:
        """Load training data from multiple relevant benchmarks."""
        from ..tasks import TaskManager

        training_data = []
        samples_per_benchmark = max(1, num_samples // len(benchmarks))

        # Create task manager instance
        task_manager = TaskManager()

        for benchmark in benchmarks:
            try:
                print(f"     Loading from {benchmark}...")

                # Load benchmark task using TaskManager
                task_data = task_manager.load_task(benchmark, limit=samples_per_benchmark * DATA_OVERSAMPLE_MULTIPLIER)
                docs = task_manager.split_task_data(task_data, split_ratio=SPLIT_RATIO_FULL)[0]

                # Extract QA pairs using existing system
                from ....contrastive_pairs.contrastive_pair_set import ContrastivePairSet

                qa_pairs = ContrastivePairSet.extract_qa_pairs_from_task_docs(benchmark, task_data, docs)

                # Convert to training format
                for pair in qa_pairs[:samples_per_benchmark]:
                    if self._is_valid_pair(pair):
                        training_data.append(
                            {
                                "prompt": pair.get("question", f"Context from {benchmark}"),
                                "harmful_response": pair.get("incorrect_answer", ""),
                                "harmless_response": pair.get("correct_answer", ""),
                                "source": benchmark,
                            }
                        )

                print(f"     Loaded {len(qa_pairs[:samples_per_benchmark])} examples from {benchmark}")

                # Stop if we have enough data
                if len(training_data) >= num_samples:
                    break

            except Exception as e:
                print(f"     Failed to load {benchmark}: {e}")
                continue

        if not training_data:
            print("   Failed to load from any benchmarks, using synthetic data...")
            return self._generate_synthetic_training_data("unknown", num_samples)

        print(
            f"   Loaded {len(training_data)} examples from "
            f"{len([b for b in benchmarks if any(b in item.get('source', '') for item in training_data)])} benchmarks"
        )
        return training_data[:num_samples]

    def _is_valid_pair(self, pair: Dict[str, Any]) -> bool:
        """Check if a QA pair is valid for training."""
        return (
            pair.get("correct_answer")
            and pair.get("incorrect_answer")
            and len(pair.get("correct_answer", "")) > 5
            and len(pair.get("incorrect_answer", "")) > 5
            and pair.get("correct_answer") != pair.get("incorrect_answer")
        )

    def _generate_synthetic_training_data(self, issue_type: str, num_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic training data for issue types."""
        from ....contrastive_pairs.generate_synthetically import SyntheticContrastivePairGenerator

        try:
            generator = SyntheticContrastivePairGenerator(self.model)

            # Create trait description based on issue type
            trait_descriptions = {
                "quality": "providing clear, accurate, and well-structured responses",
                "harmful": "giving safe, helpful, and constructive advice",
                "bias": "responding with fairness and avoiding stereotypes",
                "coherence": "maintaining logical flow and clear communication",
                "hallucination": "providing factually accurate information",
            }

            trait_description = trait_descriptions.get(issue_type, f"avoiding {issue_type} issues in responses")

            # Generate synthetic pairs
            synthetic_pairs = generator.generate_contrastive_pair_set(
                trait_description=trait_description, num_pairs=num_samples, name=f"synthetic_{issue_type}"
            )

            # Convert to training format
            training_data = []
            for pair in synthetic_pairs.pairs[:num_samples]:
                training_data.append(
                    {
                        "prompt": pair.prompt or f"Context for {issue_type} detection",
                        "harmful_response": pair.negative_response,
                        "harmless_response": pair.positive_response,
                    }
                )

            print(f"   Generated {len(training_data)} synthetic examples for {issue_type}")
            return training_data

        except Exception as e:
            print(f"   Failed to generate synthetic data: {e}")
            raise InsufficientDataError(reason=f"Cannot generate training data for issue type: {issue_type}")

    def _extract_activations_from_data(
        self, training_data: List[Dict[str, Any]], layer: int
    ) -> Tuple[List[Activations], List[Activations]]:
        """
        Extract activations from training data.

        Args:
            training_data: List of training examples
            layer: Layer to extract activations from

        Returns:
            Tuple of (harmful_activations, harmless_activations)
        """
        harmful_activations = []
        harmless_activations = []

        layer_obj = Layer(index=layer, type="transformer")

        for example in training_data:
            # Extract harmful activation
            harmful_tensor = self.model.extract_activations(example["harmful_response"], layer_obj)
            harmful_activation = Activations(tensor=harmful_tensor, layer=layer_obj)
            harmful_activations.append(harmful_activation)

            # Extract harmless activation
            harmless_tensor = self.model.extract_activations(example["harmless_response"], layer_obj)
            harmless_activation = Activations(tensor=harmless_tensor, layer=layer_obj)
            harmless_activations.append(harmless_activation)

        return harmful_activations, harmless_activations

    def _train_classifier(
        self, harmful_activations: List[Activations], harmless_activations: List[Activations],
        config: "TrainingConfig",
    ) -> ActivationClassifier:
        """
        Train a classifier on the activation data.

        Args:
            harmful_activations: List of harmful activations
            harmless_activations: List of harmless activations
            config: Training configuration

        Returns:
            Trained ActivationClassifier
        """
        classifier = ActivationClassifier(
            model_type=config.classifier_type, threshold=config.threshold, device=self.model.device
        )

        classifier.train_on_activations(harmful_activations, harmless_activations)

        return classifier

    def _evaluate_classifier(
        self,
        classifier: ActivationClassifier,
        harmful_activations: List[Activations],
        harmless_activations: List[Activations],
    ) -> Dict[str, float]:
        """
        Evaluate classifier performance.

        Args:
            classifier: Trained classifier
            harmful_activations: Test harmful activations
            harmless_activations: Test harmless activations

        Returns:
            Dictionary of performance metrics
        """
        # Use a portion of data for testing
        test_size = min(10, len(harmful_activations) // 5)  # 20% or at least 10

        test_harmful = harmful_activations[-test_size:]
        test_harmless = harmless_activations[-test_size:]

        return classifier.evaluate_on_activations(test_harmful, test_harmless)

    def _save_classifier(
        self, classifier: ActivationClassifier, config: "TrainingConfig", metrics: Dict[str, float]
    ) -> str:
        """
        Save classifier with metadata.

        Args:
            classifier: Trained classifier
            config: Training configuration
            metrics: Performance metrics

        Returns:
            Path where classifier was saved
        """
        # Create metadata
        metadata = create_classifier_metadata(
            model_name=config.model_name,
            task_name=config.issue_type,
            layer=config.layer,
            classifier_type=config.classifier_type,
            training_accuracy=metrics.get("accuracy", 0.0),
            training_samples=config.training_samples,
            token_aggregation="final",  # Default for our system
            detection_threshold=config.threshold,
            f1=metrics.get("f1", 0.0),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            auc=metrics.get("auc", 0.0),
        )

        # Save using ModelPersistence
        save_path = ModelPersistence.save_classifier(classifier.classifier, config.layer, config.save_path, metadata)

        return save_path
