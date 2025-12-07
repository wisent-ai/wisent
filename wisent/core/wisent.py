"""
Unified Wisent interface for multi-modal contrastive steering.

This module provides a high-level API that works across all modalities
while maintaining backward compatibility with the text-only API.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, TypeVar
from dataclasses import dataclass, field
import logging

import torch

from wisent.core.adapters.base import BaseAdapter, SteeringConfig, AdapterError
from wisent.core.adapters.text import TextAdapter
from wisent.core.adapters.audio import AudioAdapter
from wisent.core.adapters.video import VideoAdapter
from wisent.core.adapters.robotics import RoboticsAdapter
from wisent.core.adapters.multimodal import MultimodalAdapter
from wisent.core.errors import UnknownTypeError, NoTrainedVectorsError
from wisent.core.modalities import (
    Modality,
    ModalityContent,
    ContentType,
    TextContent,
    AudioContent,
    VideoContent,
    ImageContent,
    RobotState,
    RobotTrajectory,
    MultimodalContent,
    wrap_content,
)
from wisent.core.activations.core.atoms import LayerActivations
from wisent.core.steering_methods.core.atoms import BaseSteeringMethod

__all__ = [
    "Wisent",
    "TraitConfig",
]

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT", bound=ModalityContent)
OutputT = TypeVar("OutputT")


@dataclass
class TraitConfig:
    """
    Configuration for a steering trait.

    Attributes:
        name: Unique identifier for the trait
        description: Human-readable description
        steering_vectors: Per-layer steering vectors
        default_scale: Default steering strength
        layers: Which layers to apply to (None = all recommended)
    """
    name: str
    description: str = ""
    steering_vectors: LayerActivations | None = None
    default_scale: float = 1.0
    layers: List[str] | None = None


class Wisent:
    """
    Unified interface for multi-modal contrastive steering.

    Wisent provides a simple, consistent API for steering AI models across
    different modalities (text, audio, video, robotics). The core workflow is:

    1. Create a Wisent instance with an appropriate adapter
    2. Add contrastive pairs (positive/negative examples)
    3. Train steering vectors from the pairs
    4. Generate outputs with steering applied

    Example (Text):
        >>> wisent = Wisent.for_text("meta-llama/Llama-3-8B-Instruct")
        >>> wisent.add_pair(
        ...     positive="I'd be happy to help you with that.",
        ...     negative="I refuse to help with that.",
        ...     trait="helpfulness"
        ... )
        >>> wisent.train()
        >>> response = wisent.generate("How do I cook pasta?", steer={"helpfulness": 1.5})

    Example (Audio):
        >>> wisent = Wisent.for_audio("openai/whisper-large-v3")
        >>> wisent.add_pair(
        ...     positive=AudioContent.from_file("calm_speech.wav"),
        ...     negative=AudioContent.from_file("angry_speech.wav"),
        ...     trait="calmness"
        ... )
        >>> wisent.train()
        >>> transcript = wisent.generate(audio_input, steer={"calmness": 1.2})

    Example (Video):
        >>> wisent = Wisent.for_video("MCG-NJU/videomae-base")
        >>> wisent.add_pair(
        ...     positive=VideoContent.from_file("safe_action.mp4"),
        ...     negative=VideoContent.from_file("unsafe_action.mp4"),
        ...     trait="safety"
        ... )
        >>> wisent.train()
        >>> embedding = wisent.generate(video_input, steer={"safety": 2.0})

    Example (Robotics):
        >>> wisent = Wisent.for_robotics(model=my_policy_network)
        >>> wisent.add_pair(
        ...     positive=gentle_trajectory,
        ...     negative=forceful_trajectory,
        ...     trait="gentleness"
        ... )
        >>> wisent.train()
        >>> action = wisent.act(robot_state, steer={"gentleness": 1.5})
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        steering_method: str = "caa",
        device: str | None = None,
    ):
        """
        Initialize Wisent with an adapter.

        Args:
            adapter: The modality-specific adapter to use
            steering_method: Method for computing steering vectors ("caa", etc.)
            device: Override device (uses adapter's device by default)
        """
        self.adapter = adapter
        self.steering_method_name = steering_method
        self.device = device or adapter.device

        # Storage for pairs and trained vectors
        self._pairs: Dict[str, List[tuple[ModalityContent, ModalityContent]]] = {}
        self._traits: Dict[str, TraitConfig] = {}
        self._trained = False

    # ==================== Factory Methods ====================

    @classmethod
    def for_text(
        cls,
        model_name: str,
        device: str | None = None,
        **kwargs: Any,
    ) -> "Wisent":
        """
        Create a Wisent instance for text/LLM steering.

        Args:
            model_name: HuggingFace model identifier
            device: Target device
            **kwargs: Additional adapter arguments

        Returns:
            Configured Wisent instance
        """
        adapter = TextAdapter(model_name=model_name, device=device, **kwargs)
        return cls(adapter)

    @classmethod
    def for_audio(
        cls,
        model_name: str,
        device: str | None = None,
        **kwargs: Any,
    ) -> "Wisent":
        """
        Create a Wisent instance for audio steering.

        Args:
            model_name: HuggingFace model identifier
            device: Target device
            **kwargs: Additional adapter arguments

        Returns:
            Configured Wisent instance
        """
        adapter = AudioAdapter(model_name=model_name, device=device, **kwargs)
        return cls(adapter)

    @classmethod
    def for_video(
        cls,
        model_name: str,
        device: str | None = None,
        **kwargs: Any,
    ) -> "Wisent":
        """
        Create a Wisent instance for video steering.

        Args:
            model_name: HuggingFace model identifier
            device: Target device
            **kwargs: Additional adapter arguments

        Returns:
            Configured Wisent instance
        """
        adapter = VideoAdapter(model_name=model_name, device=device, **kwargs)
        return cls(adapter)

    @classmethod
    def for_robotics(
        cls,
        model: torch.nn.Module,
        device: str | None = None,
        **kwargs: Any,
    ) -> "Wisent":
        """
        Create a Wisent instance for robotics policy steering.

        Args:
            model: Policy network
            device: Target device
            **kwargs: Additional adapter arguments

        Returns:
            Configured Wisent instance
        """
        adapter = RoboticsAdapter(model=model, device=device, **kwargs)
        return cls(adapter)

    @classmethod
    def for_multimodal(
        cls,
        model_name: str,
        device: str | None = None,
        **kwargs: Any,
    ) -> "Wisent":
        """
        Create a Wisent instance for multimodal (VLM, etc.) steering.

        Args:
            model_name: HuggingFace model identifier
            device: Target device
            **kwargs: Additional adapter arguments

        Returns:
            Configured Wisent instance
        """
        adapter = MultimodalAdapter(model_name=model_name, device=device, **kwargs)
        return cls(adapter)

    # ==================== Pair Management ====================

    def add_pair(
        self,
        positive: ContentType,
        negative: ContentType,
        trait: str,
        description: str | None = None,
    ) -> "Wisent":
        """
        Add a contrastive pair for a trait.

        Args:
            positive: Desired behavior example
            negative: Undesired behavior example
            trait: Name of the trait being steered
            description: Optional description of the trait

        Returns:
            Self for chaining
        """
        # Wrap raw content
        pos_content = wrap_content(positive) if isinstance(positive, str) else positive
        neg_content = wrap_content(negative) if isinstance(negative, str) else negative

        # Initialize trait if needed
        if trait not in self._pairs:
            self._pairs[trait] = []
            self._traits[trait] = TraitConfig(
                name=trait,
                description=description or f"Steering trait: {trait}",
            )

        self._pairs[trait].append((pos_content, neg_content))
        self._trained = False  # Need to retrain

        return self

    def add_pairs(
        self,
        pairs: List[tuple[ContentType, ContentType]],
        trait: str,
        description: str | None = None,
    ) -> "Wisent":
        """
        Add multiple contrastive pairs for a trait.

        Args:
            pairs: List of (positive, negative) tuples
            trait: Name of the trait
            description: Optional description

        Returns:
            Self for chaining
        """
        for pos, neg in pairs:
            self.add_pair(pos, neg, trait, description)
        return self

    def clear_pairs(self, trait: str | None = None) -> "Wisent":
        """
        Clear stored pairs.

        Args:
            trait: Specific trait to clear (None = all)

        Returns:
            Self for chaining
        """
        if trait is None:
            self._pairs.clear()
            self._traits.clear()
        elif trait in self._pairs:
            del self._pairs[trait]
            del self._traits[trait]

        self._trained = False
        return self

    # ==================== Training ====================

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

    def act(
        self,
        state: RobotState,
        steer: Dict[str, float] | None = None,
        config: SteeringConfig | None = None,
    ) -> Any:
        """
        Convenience method for robotics: get action from state.

        Args:
            state: Robot state observation
            steer: Steering configuration
            config: Additional config

        Returns:
            Robot action
        """
        if not isinstance(self.adapter, RoboticsAdapter):
            raise AdapterError("act() is only available for robotics adapter")

        return self.generate(state, steer=steer, config=config)

    def transcribe(
        self,
        audio: AudioContent,
        steer: Dict[str, float] | None = None,
        language: str | None = None,
    ) -> str:
        """
        Convenience method for audio: transcribe with steering.

        Args:
            audio: Audio content
            steer: Steering configuration
            language: Target language

        Returns:
            Transcribed text
        """
        if not isinstance(self.adapter, AudioAdapter):
            raise AdapterError("transcribe() is only available for audio adapter")

        if steer is None or not steer:
            return self.adapter.transcribe(audio, language=language)

        combined_vectors = self._combine_steering_vectors(steer)
        return self.adapter.transcribe(
            audio,
            steering_vectors=combined_vectors,
            language=language,
        )

    # ==================== Introspection ====================

    @property
    def traits(self) -> List[str]:
        """Get list of defined traits."""
        return list(self._traits.keys())

    @property
    def is_trained(self) -> bool:
        """Check if steering vectors have been trained."""
        return self._trained

    def get_trait_info(self, trait: str) -> TraitConfig | None:
        """Get configuration for a trait."""
        return self._traits.get(trait)

    def get_intervention_points(self) -> List[str]:
        """Get available steering intervention points."""
        return [ip.name for ip in self.adapter.get_intervention_points()]

    def get_recommended_layers(self) -> List[str]:
        """Get recommended layers for steering."""
        return [
            ip.name for ip in self.adapter.get_intervention_points()
            if ip.recommended
        ]

    # ==================== Persistence ====================

    def save_vectors(self, path: str) -> None:
        """
        Save trained steering vectors to file.

        Args:
            path: File path to save to
        """
        if not self._trained:
            raise NoTrainedVectorsError()

        data = {}
        for trait, config in self._traits.items():
            if config.steering_vectors is not None:
                data[trait] = {
                    "description": config.description,
                    "default_scale": config.default_scale,
                    "layers": config.layers,
                    "vectors": {
                        k: v.cpu() for k, v in config.steering_vectors.items()
                        if v is not None
                    },
                }

        torch.save(data, path)
        logger.info(f"Saved steering vectors to {path}")

    def load_vectors(self, path: str) -> "Wisent":
        """
        Load trained steering vectors from file.

        Args:
            path: File path to load from

        Returns:
            Self for chaining
        """
        data = torch.load(path, map_location=self.device)

        for trait, trait_data in data.items():
            self._traits[trait] = TraitConfig(
                name=trait,
                description=trait_data.get("description", ""),
                default_scale=trait_data.get("default_scale", 1.0),
                layers=trait_data.get("layers"),
                steering_vectors=LayerActivations(trait_data["vectors"]),
            )

        self._trained = True
        logger.info(f"Loaded steering vectors from {path}")
        return self

    def __repr__(self) -> str:
        return (
            f"Wisent(\n"
            f"  adapter={self.adapter!r},\n"
            f"  traits={self.traits},\n"
            f"  trained={self._trained}\n"
            f")"
        )
