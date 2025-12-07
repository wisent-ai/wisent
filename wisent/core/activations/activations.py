"""Activation wrapper for classifier feature extraction."""

from typing import Any
import torch
from wisent.core.activations.core.atoms import ActivationAggregationStrategy
from wisent.core.errors import InvalidValueError


class Activations:
    """Wrapper for activation tensors with aggregation strategy.

    This class wraps activation tensors and provides methods to extract
    features for classifier input based on the specified aggregation strategy.
    """

    def __init__(self, tensor: torch.Tensor, layer: Any, aggregation_strategy):
        """Initialize Activations wrapper.

        Args:
            tensor: Activation tensor (typically shape [batch, seq_len, hidden_dim])
            layer: Layer object containing layer metadata
            aggregation_strategy: Strategy for aggregating tokens (string or ActivationAggregationStrategy enum)
        """
        self.tensor = tensor
        self.layer = layer

        # Convert string to enum if needed
        if isinstance(aggregation_strategy, str):
            # Map common string values to enum
            strategy_map = {
                "average": ActivationAggregationStrategy.MEAN_POOLING,
                "mean": ActivationAggregationStrategy.MEAN_POOLING,
                "final": ActivationAggregationStrategy.LAST_TOKEN,
                "last": ActivationAggregationStrategy.LAST_TOKEN,
                "first": ActivationAggregationStrategy.FIRST_TOKEN,
                "max": ActivationAggregationStrategy.MAX_POOLING,
                "mean_pooling": ActivationAggregationStrategy.MEAN_POOLING,
                "last_token": ActivationAggregationStrategy.LAST_TOKEN,
                "first_token": ActivationAggregationStrategy.FIRST_TOKEN,
                "max_pooling": ActivationAggregationStrategy.MAX_POOLING,
            }
            self.aggregation_strategy = strategy_map.get(
                aggregation_strategy.lower(),
                ActivationAggregationStrategy.MEAN_POOLING
            )
        else:
            self.aggregation_strategy = aggregation_strategy

    def extract_features_for_classifier(self) -> torch.Tensor:
        """Extract features from activations for classifier input.

        Aggregates the activation tensor based on the specified strategy
        to produce a single feature vector suitable for classifier input.

        Returns:
            torch.Tensor: Aggregated features (typically shape [hidden_dim])
        """
        if self.tensor is None:
            raise InvalidValueError(param="tensor", reason="Cannot extract features from None tensor")

        # Ensure tensor is 3D: [batch, seq_len, hidden_dim]
        if len(self.tensor.shape) == 2:
            # If [seq_len, hidden_dim], add batch dimension
            tensor = self.tensor.unsqueeze(0)
        else:
            tensor = self.tensor

        # Apply aggregation strategy
        if self.aggregation_strategy == ActivationAggregationStrategy.MEAN_POOLING:
            # Average over sequence length dimension
            features = tensor.mean(dim=1).squeeze(0)
        elif self.aggregation_strategy == ActivationAggregationStrategy.LAST_TOKEN:
            # Take last token
            features = tensor[:, -1, :].squeeze(0)
        elif self.aggregation_strategy == ActivationAggregationStrategy.FIRST_TOKEN:
            # Take first token
            features = tensor[:, 0, :].squeeze(0)
        elif self.aggregation_strategy == ActivationAggregationStrategy.MAX_POOLING:
            # Max over sequence length dimension
            features = tensor.max(dim=1)[0].squeeze(0)
        else:
            # Default to mean pooling
            features = tensor.mean(dim=1).squeeze(0)

        return features

    def cpu(self):
        """Move tensor to CPU."""
        if self.tensor is not None:
            self.tensor = self.tensor.cpu()
        return self

    def detach(self):
        """Detach tensor from computation graph."""
        if self.tensor is not None:
            self.tensor = self.tensor.detach()
        return self
