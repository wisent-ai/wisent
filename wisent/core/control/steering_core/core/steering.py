import datetime
import json
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

# Note: Activations and Classifier imports removed - these classes don't exist in the current package structure
# from wisent.core.activations import Activations
# from wisent.core.classifier.classifier import Classifier

from wisent.core.contrastive_pairs import ContrastivePairSet
from wisent.core.constants import DEFAULT_STRENGTH, BLEND_DEFAULT, DISPLAY_TRUNCATION_COMPACT
from .steering_method import CAA
from wisent.core.errors import (
    MissingParameterError,
    InsufficientDataError,
    SteeringMethodUnknownError,
    NoTrainedVectorsError,
    LayerNotFoundError,
)

from wisent.core.steering_core._steering_logging import SteeringLoggingMixin
from wisent.core.steering_core._steering_optimization import SteeringOptimizationMixin
from wisent.core.steering_core._steering_eval_io import SteeringEvalIOMixin

class SteeringType(Enum):
    LOGISTIC = "logistic"
    MLP = "mlp"
    CUSTOM = "custom"
    CAA = "caa"  # New vector-based steering


class SteeringMethod(SteeringLoggingMixin, SteeringOptimizationMixin, SteeringEvalIOMixin):
    """
    Legacy classifier-based steering method for backward compatibility.
    For new vector-based steering, use steering_method.CAA directly.
    """

    def __init__(self, method_type: SteeringType, device=None, threshold=BLEND_DEFAULT):
        self.method_type = method_type
        self.device = device
        self.threshold = threshold
        self.classifier = None

        # For vector-based steering
        self.vector_steering = None
        self.is_vector_based = method_type == SteeringType.CAA

        if self.is_vector_based:
            self.vector_steering = CAA(device=device)

        # Response logging settings
        self.enable_logging = False
        self.log_file_path = "./harmful_responses.json"

        # Parameter optimization tracking
        self.original_parameters = {}
        self.optimization_history = []

    def train(
        self, contrastive_pair_set: ContrastivePairSet, layer_index: Optional[int] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Train the steering method on a ContrastivePairSet.

        Args:
            contrastive_pair_set: Set of contrastive pairs with activations
            layer_index: Layer index for vector-based steering (required for CAA)
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training metrics
        """
        if self.is_vector_based:
            if layer_index is None:
                raise MissingParameterError(params=["layer_index"], context="vector-based steering methods")
            return self.vector_steering.train(contrastive_pair_set, layer_index)

        # Legacy classifier-based training
        X, y = contrastive_pair_set.prepare_classifier_data()

        if len(X) < 4:
            raise InsufficientDataError(reason="Need at least 4 training examples", required=4, actual=len(X))

        # Create classifier
        self.classifier = Classifier(model_type=self.method_type.value, device=self.device, threshold=self.threshold)

        # Train classifier
        results = self.classifier.fit(X, y, **kwargs)

        return results

    def apply_steering(self, activations: torch.Tensor, strength: float = DEFAULT_STRENGTH) -> torch.Tensor:
        """
        Apply steering to activations (vector-based methods only).

        Args:
            activations: Input activations
            strength: Steering strength

        Returns:
            Steered activations
        """
        if not self.is_vector_based:
            raise SteeringMethodUnknownError(method="apply_steering requires vector-based methods")

        return self.vector_steering.apply_steering(activations, strength)

    def get_steering_vector(self) -> Optional[torch.Tensor]:
        """Get steering vector (vector-based methods only)."""
        if not self.is_vector_based:
            return None
        return self.vector_steering.get_steering_vector()

    def predict(self, activations) -> float:
        """
        Predict if activations represent harmful behavior (classifier-based only).

        Args:
            activations: Activation tensor or Activations object

        Returns:
            Prediction score (0 = harmless, 1 = harmful)
        """
        if self.is_vector_based:
            raise SteeringMethodUnknownError(method="predict not available for vector-based methods")

        if self.classifier is None:
            raise NoTrainedVectorsError()

        return self.classifier.predict(activations)

    def predict_proba(self, activations) -> float:
        """
        Get prediction probability for activations (classifier-based only).

        Args:
            activations: Activation tensor or Activations object

        Returns:
            Probability score (0.0-1.0)
        """
        if self.is_vector_based:
            raise SteeringMethodUnknownError(method="predict_proba not available for vector-based methods")

        if self.classifier is None:
            raise NoTrainedVectorsError()

        return self.classifier.predict_proba(activations)

    def is_harmful(self, activations, detailed=False) -> Union[bool, Dict[str, Any]]:
        """
        Check if activations represent harmful content (classifier-based only).

        Args:
            activations: Activation tensor or Activations object
            detailed: Whether to return detailed results

        Returns:
            Boolean or detailed dictionary
        """
        if self.is_vector_based:
            raise SteeringMethodUnknownError(method="is_harmful not available for vector-based methods")

        if self.classifier is None:
            raise NoTrainedVectorsError()

        # Get probability score
        probability = self.predict_proba(activations)
        is_harmful = probability >= self.threshold

        if detailed:
            return {
                "is_harmful": is_harmful,
                "probability": probability,
                "threshold": self.threshold,
                "method_type": self.method_type.value,
            }
        return is_harmful

    def check_safety(self, text: str, model, layer) -> Dict[str, Any]:
        """
        Comprehensive safety check for text using the model.

        Args:
            text: Text to check
            model: Model object for activation extraction
            layer: Layer object for activation extraction

        Returns:
            Safety check results
        """
        try:
            # Extract activations from text
            activations_tensor = model.extract_activations(text, layer)

            # Create Activations object
            activations = Activations(tensor=activations_tensor, layer=layer)

            # Get detailed prediction
            result = self.is_harmful(activations, detailed=True)

            # Add text information
            result.update(
                {
                    "text": text[:DISPLAY_TRUNCATION_COMPACT] + "..." if len(text) > DISPLAY_TRUNCATION_COMPACT else text,
                    "text_length": len(text),
                    "layer_index": layer.index,
                }
            )

            return result

        except Exception as e:
            return {
                "is_harmful": False,
                "probability": 0.0,
                "error": str(e),
                "text": text[:DISPLAY_TRUNCATION_COMPACT] + "..." if len(text) > DISPLAY_TRUNCATION_COMPACT else text,
            }

