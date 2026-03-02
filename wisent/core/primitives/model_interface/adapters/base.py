"""
Base adapter interface for multi-modal contrastive steering.

All modality-specific adapters inherit from BaseAdapter and implement
the core methods for encoding, decoding, and steering.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from contextlib import contextmanager
import inspect

import torch
import torch.nn as nn

from wisent.core.primitives.models.modalities import (
    Modality,
    ModalityContent,
    ContentType,
    wrap_content,
)
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations, RawActivationMap
from wisent.core.utils.infra_tools.errors import DuplicateNameError
from wisent.core.primitives.model_interface.adapters._base_steering_mixin import SteeringHookMixin

__all__ = [
    "AdapterError",
    "BaseAdapter",
    "InterventionPoint",
    "SteeringConfig",
]


class AdapterError(RuntimeError):
    """Raised when an adapter operation fails."""
    pass


@dataclass(frozen=True, slots=True)
class InterventionPoint:
    """
    Describes a point in the model where steering can be applied.

    Attributes:
        name: Unique identifier for this intervention point (e.g., "layer.12", "encoder.block.5")
        module_path: Dot-separated path to the module in the model
        description: Human-readable description
        recommended: Whether this is a recommended point for steering
    """
    name: str
    module_path: str
    description: str = ""
    recommended: bool = False


@dataclass
class SteeringConfig:
    """
    Configuration for how steering vectors should be applied.

    Attributes:
        scale: Multiplier for steering vector magnitude (or Dict[str, float] for per-layer)
        normalize: Whether to L2-normalize vectors before applying
        layers: Which layers to apply steering to (None = all available)
        temporal_mode: How to handle temporal sequences ("per_step", "aggregate", "first", "last")
        method: SteeringMethod instance for non-linear steering, or Dict[str, SteeringMethod]
                for per-layer methods (None = linear addition)
    """
    scale: float | Dict[str, float] = 1.0
    normalize: bool = True
    layers: List[str] | None = None
    temporal_mode: Optional[str] = None
    method: Any = None  # SteeringMethod, Dict[str, SteeringMethod], or None for linear


# Generic type for model output
OutputT = TypeVar("OutputT")
InputT = TypeVar("InputT", bound=ModalityContent)


class BaseAdapter(SteeringHookMixin, ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for modality adapters.

    An adapter wraps a model (encoder, decoder, or both) and provides:
    1. Encoding: Convert raw input to latent representations
    2. Decoding: Convert latents back to output
    3. Steering: Apply contrastive vectors at intervention points
    4. Activation extraction: Get intermediate representations for training

    Subclasses must implement the abstract methods for their specific modality.
    """

    name: Optional[str] = None
    modality: Modality = Modality.TEXT
    _REGISTRY: Dict[str, type["BaseAdapter"]] = {}

    def __init_subclass__(cls, **kwargs):
        """Auto-register concrete adapter subclasses."""
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        if not getattr(cls, "name", None):
            raise TypeError("BaseAdapter subclasses must define `name`.")
        if cls.name in BaseAdapter._REGISTRY:
            raise DuplicateNameError(name=cls.name, context="adapter registry")
        BaseAdapter._REGISTRY[cls.name] = cls

    def __init__(
        self,
        model: nn.Module | None = None,
        model_name: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the adapter.

        Args:
            model: Pre-loaded model (optional)
            model_name: Model identifier for loading (e.g., HuggingFace repo ID)
            device: Target device ('cuda', 'cpu', 'mps', etc.)
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self.device = device or self._resolve_device()
        self._kwargs = kwargs
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        # Move model to device if provided
        if model is not None:
            self._model = model.to(self.device)
        else:
            self._model = None

    def _resolve_device(self) -> str:
        """Resolve the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def model(self) -> nn.Module:
        """Get the underlying model, loading if necessary."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @abstractmethod
    def _load_model(self) -> nn.Module:
        """
        Load the model from model_name.

        Returns:
            The loaded model ready for inference.

        Raises:
            AdapterError: If model cannot be loaded.
        """
        ...

    @abstractmethod
    def encode(self, content: InputT) -> torch.Tensor:
        """
        Encode input content to latent representation.

        Args:
            content: Input content of the adapter's modality type

        Returns:
            Latent tensor representation, shape depends on modality:
            - Text: [batch, seq_len, hidden_dim]
            - Audio: [batch, time_steps, hidden_dim]
            - Video: [batch, frames, hidden_dim]
            - Robot: [batch, state_dim] or [batch, seq_len, state_dim]
        """
        ...

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> OutputT:
        """
        Decode latent representation back to output.

        Args:
            latent: Latent tensor representation

        Returns:
            Decoded output in the appropriate format for the modality
        """
        ...

    @abstractmethod
    def get_intervention_points(self) -> List[InterventionPoint]:
        """
        Get all available intervention points for steering.

        Returns:
            List of InterventionPoint describing where steering can be applied.
            This typically includes decoder/transformer layers.
        """
        ...

    @abstractmethod
    def extract_activations(
        self,
        content: InputT,
        layers: List[str] | None = None,
    ) -> LayerActivations:
        """
        Extract activations from specified layers for the given input.

        Args:
            content: Input content
            layers: Layer names to extract from (None = all intervention points)

        Returns:
            LayerActivations mapping layer names to activation tensors
        """
        ...

    @abstractmethod
    def forward_with_steering(
        self,
        content: InputT,
        steering_vectors: LayerActivations,
        config: SteeringConfig | None = None,
    ) -> OutputT:
        """
        Forward pass with steering vectors applied.

        Args:
            content: Input content
            steering_vectors: Per-layer steering vectors to add
            config: Configuration for how to apply steering
        Returns:
            Steered output
        """
        ...

    def generate(
        self,
        content: InputT,
        steering_vectors: LayerActivations | None = None,
        config: SteeringConfig | None = None,
        **generation_kwargs: Any,
    ) -> OutputT:
        """
        Generate output, optionally with steering.

        This is the main user-facing method. If steering_vectors is provided,
        applies steering during generation.

        Args:
            content: Input content
            steering_vectors: Optional steering vectors
            config: Steering configuration
            **generation_kwargs: Model-specific generation parameters

        Returns:
            Generated output
        """
        if steering_vectors is not None:
            return self.forward_with_steering(content, steering_vectors, config)
        return self._generate_unsteered(content, **generation_kwargs)

    @abstractmethod
    def _generate_unsteered(self, content: InputT, **kwargs: Any) -> OutputT:
        """Generate output without steering."""
        ...
