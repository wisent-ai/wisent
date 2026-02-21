"""Evaluation and IO mixin for SteeringMethod."""
from __future__ import annotations

import torch
from typing import Any, Dict, Optional, TYPE_CHECKING
from wisent.core.contrastive_pairs import ContrastivePairSet
from wisent.core.errors import NoTrainedVectorsError, InsufficientDataError

if TYPE_CHECKING:
    from wisent.core.steering.steering import SteeringType


class SteeringEvalIOMixin:
    """Mixin providing evaluation and IO methods."""

    def evaluate(self, contrastive_pair_set: ContrastivePairSet) -> Dict[str, Any]:
        """
        Evaluate the steering method on a ContrastivePairSet.

        Args:
            contrastive_pair_set: Set of contrastive pairs for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        if self.classifier is None:
            raise NoTrainedVectorsError()

        # Get positive and negative activations
        pos_activations, neg_activations = contrastive_pair_set.get_activation_pairs()

        # Predict on positive activations (should be low scores)
        pos_predictions = []
        for activation in pos_activations:
            pred = self.predict_proba(activation)
            pos_predictions.append(pred)

        # Predict on negative activations (should be high scores)
        neg_predictions = []
        for activation in neg_activations:
            pred = self.predict_proba(activation)
            neg_predictions.append(pred)

        # Calculate metrics
        # True Positives: negative activations correctly identified as harmful (pred >= threshold)
        true_positives = sum(1 for pred in neg_predictions if pred >= self.threshold)

        # False Positives: positive activations incorrectly identified as harmful (pred >= threshold)
        false_positives = sum(1 for pred in pos_predictions if pred >= self.threshold)

        # True Negatives: positive activations correctly identified as harmless (pred < threshold)
        true_negatives = sum(1 for pred in pos_predictions if pred < self.threshold)

        # False Negatives: negative activations incorrectly identified as harmless (pred < threshold)
        false_negatives = sum(1 for pred in neg_predictions if pred < self.threshold)

        # Calculate metrics
        detection_rate = true_positives / len(neg_predictions) if neg_predictions else 0
        false_positive_rate = false_positives / len(pos_predictions) if pos_predictions else 0

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        accuracy = (
            (true_positives + true_negatives) / (len(pos_predictions) + len(neg_predictions))
            if (pos_predictions or neg_predictions)
            else 0
        )

        return {
            "detection_rate": detection_rate,
            "false_positive_rate": false_positive_rate,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "num_positive_samples": len(pos_predictions),
            "num_negative_samples": len(neg_predictions),
            "threshold": self.threshold,
        }

    def save_model(self, save_path: str) -> bool:
        """
        Save the steering method to disk.

        Args:
            save_path: Path to save the model

        Returns:
            Success flag
        """
        if self.classifier is None:
            return False

        try:
            self.classifier.save_model(save_path)
            return True
        except Exception:
            return False

    def load_model(self, model_path: str) -> bool:
        """
        Load a steering method from disk.

        Args:
            model_path: Path to the saved model

        Returns:
            Success flag
        """
        try:
            self.classifier = Classifier(
                model_type=self.method_type.value, device=self.device, threshold=self.threshold, model_path=model_path
            )
            return True
        except Exception:
            return False

    @classmethod
    def create_and_train(
        cls,
        method_type: SteeringType,
        contrastive_pair_set: ContrastivePairSet,
        device: Optional[str] = None,
        threshold: float = 0.5,
        **training_kwargs,
    ) -> "SteeringMethod":
        """
        Create and train a SteeringMethod in one step.

        Args:
            method_type: Type of steering method
            contrastive_pair_set: Training data
            device: Device to use
            threshold: Classification threshold
            **training_kwargs: Additional training parameters

        Returns:
            Trained SteeringMethod
        """
        steering = cls(method_type=method_type, device=device, threshold=threshold)
        steering.train(contrastive_pair_set, **training_kwargs)
        return steering
