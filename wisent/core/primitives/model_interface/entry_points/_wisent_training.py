"""Training and generation mixin for Wisent."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import logging
import torch
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations

logger = logging.getLogger(__name__)


class WisentTrainingMixin:
    """Mixin providing train and generate methods."""

    def train(
        self,
        traits: List[str] | None = None,
        layers: List[str] | None = None,
        aggregation: str = "mean",
    ) -> "Wisent":
        """
        Train steering vectors from stored pairs.

        Args:
            traits: Which traits to train (None = all)
            layers: Which layers to use (None = recommended)
            aggregation: How to aggregate across pairs ("mean", "median")

        Returns:
            Self for chaining
        """
        target_traits = traits if traits else list(self._pairs.keys())

        if not target_traits:
            logger.warning("No pairs added. Nothing to train.")
            return self

        # Determine target layers
        if layers is None:
            intervention_points = self.adapter.get_intervention_points()
            target_layers = [ip.name for ip in intervention_points if ip.recommended]
            if not target_layers:
                target_layers = [ip.name for ip in intervention_points]

        else:
            target_layers = layers

        # Train each trait
        for trait in target_traits:
            if trait not in self._pairs:
                logger.warning(f"No pairs for trait '{trait}'. Skipping.")
                continue

            pairs = self._pairs[trait]
            logger.info(f"Training '{trait}' with {len(pairs)} pairs on {len(target_layers)} layers")

            # Collect activations for all pairs
            layer_positives: Dict[str, List[torch.Tensor]] = {l: [] for l in target_layers}
            layer_negatives: Dict[str, List[torch.Tensor]] = {l: [] for l in target_layers}

            for pos_content, neg_content in pairs:
                # Extract activations
                pos_acts = self.adapter.extract_activations(pos_content, target_layers)
                neg_acts = self.adapter.extract_activations(neg_content, target_layers)

                for layer in target_layers:
                    if layer in pos_acts and pos_acts[layer] is not None:
                        # Pool across sequence dimension if present
                        pos_tensor = pos_acts[layer]
                        if pos_tensor.dim() > 2:
                            pos_tensor = pos_tensor.mean(dim=1)
                        elif pos_tensor.dim() == 2:
                            pos_tensor = pos_tensor.mean(dim=0, keepdim=True)
                        layer_positives[layer].append(pos_tensor.squeeze())

                    if layer in neg_acts and neg_acts[layer] is not None:
                        neg_tensor = neg_acts[layer]
                        if neg_tensor.dim() > 2:
                            neg_tensor = neg_tensor.mean(dim=1)
                        elif neg_tensor.dim() == 2:
                            neg_tensor = neg_tensor.mean(dim=0, keepdim=True)
                        layer_negatives[layer].append(neg_tensor.squeeze())

            # Compute steering vectors (CAA: mean(pos) - mean(neg))
            steering_vectors: Dict[str, torch.Tensor] = {}
            for layer in target_layers:
                if not layer_positives[layer] or not layer_negatives[layer]:
                    continue

                pos_stack = torch.stack(layer_positives[layer])
                neg_stack = torch.stack(layer_negatives[layer])

                if aggregation == "mean":
                    pos_agg = pos_stack.mean(dim=0)
                    neg_agg = neg_stack.mean(dim=0)
                elif aggregation == "median":
                    pos_agg = pos_stack.median(dim=0).values
                    neg_agg = neg_stack.median(dim=0).values
                else:
                    raise UnknownTypeError(entity_type="aggregation", value=aggregation, valid_values=["mean", "median"])

                steering_vectors[layer] = pos_agg - neg_agg

            # Store in trait config
            self._traits[trait].steering_vectors = LayerActivations(steering_vectors)
            self._traits[trait].layers = target_layers

        self._trained = True
        return self

    # ==================== Generation ====================

    def generate(
        self,
        content: ContentType,
        steer: Dict[str, float] | None = None,
        config: SteeringConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generate output with optional steering.

        Args:
            content: Input content (text, audio, video, etc.)
            steer: Dict mapping trait names to steering strengths
            config: Additional steering configuration
            **kwargs: Additional generation arguments

        Returns:
            Generated output (type depends on adapter)
        """
        # Wrap content if needed
        if isinstance(content, str):
            content = TextContent(text=content)

        # No steering requested
        if steer is None or not steer:
            return self.adapter.generate(content, **kwargs)

        # Ensure trained
        if not self._trained:
            logger.warning("Steering vectors not trained. Call train() first.")
            return self.adapter.generate(content, **kwargs)

        # Combine steering vectors from requested traits
        combined_vectors = self._combine_steering_vectors(steer)

        if combined_vectors is None:
            logger.warning("No valid steering vectors found.")
            return self.adapter.generate(content, **kwargs)

        return self.adapter.generate(
            content,
            steering_vectors=combined_vectors,
            config=config,
            **kwargs,
        )

    def _combine_steering_vectors(
        self,
        steer: Dict[str, float],
    ) -> LayerActivations | None:
        """
        Combine steering vectors from multiple traits.

        Args:
            steer: Dict mapping trait names to scales

        Returns:
            Combined LayerActivations or None
        """
        combined: Dict[str, torch.Tensor] = {}

        for trait, scale in steer.items():
            if trait not in self._traits:
                logger.warning(f"Unknown trait '{trait}'. Skipping.")
                continue

            trait_config = self._traits[trait]
            if trait_config.steering_vectors is None:
                logger.warning(f"Trait '{trait}' not trained. Skipping.")
                continue

            effective_scale = scale * trait_config.default_scale

            for layer, vector in trait_config.steering_vectors.items():
                if vector is None:
                    continue

                scaled_vector = vector * effective_scale

                if layer in combined:
                    combined[layer] = combined[layer] + scaled_vector
                else:
                    combined[layer] = scaled_vector

        if not combined:
            return None

        return LayerActivations(combined)

    # ==================== Convenience Methods ====================

