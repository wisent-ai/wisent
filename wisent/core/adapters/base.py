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

from wisent.core.modalities import (
    Modality,
    ModalityContent,
    ContentType,
    wrap_content,
)
from wisent.core.activations.core.atoms import LayerActivations, RawActivationMap
from wisent.core.errors import DuplicateNameError

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
        scale: Multiplier for steering vector magnitude
        normalize: Whether to L2-normalize vectors before applying
        layers: Which layers to apply steering to (None = all available)
        temporal_mode: How to handle temporal sequences ("per_step", "aggregate", "first", "last")
    """
    scale: float = 1.0
    normalize: bool = True
    layers: List[str] | None = None
    temporal_mode: str = "per_step"


# Generic type for model output
OutputT = TypeVar("OutputT")
InputT = TypeVar("InputT", bound=ModalityContent)


class BaseAdapter(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for modality adapters.

    An adapter wraps a model (encoder, decoder, or both) and provides:
    1. Encoding: Convert raw input to latent representations
    2. Decoding: Convert latents back to output
    3. Steering: Apply contrastive vectors at intervention points
    4. Activation extraction: Get intermediate representations for training

    Subclasses must implement the abstract methods for their specific modality.
    """

    name: str = "base"
    modality: Modality = Modality.TEXT
    _REGISTRY: Dict[str, type["BaseAdapter"]] = {}

    def __init_subclass__(cls, **kwargs):
        """Auto-register concrete adapter subclasses."""
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        if not getattr(cls, "name", None):
            raise TypeError("BaseAdapter subclasses must define `name`.")
        if cls.name != "base" and cls.name in BaseAdapter._REGISTRY:
            raise DuplicateNameError(name=cls.name, context="adapter registry")
        if cls.name != "base":
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

    @contextmanager
    def _steering_hooks(
        self,
        steering_vectors: LayerActivations,
        config: SteeringConfig | None = None,
    ):
        """
        Context manager that registers steering hooks and cleans up after.

        Args:
            steering_vectors: Steering vectors to apply
            config: Steering configuration

        Yields:
            None (hooks are active within context)
        """
        config = config or SteeringConfig()
        handles = []

        try:
            # Get intervention points
            intervention_points = {ip.name: ip for ip in self.get_intervention_points()}

            # Register hooks for each layer with a steering vector
            for layer_name, vector in steering_vectors.items():
                if vector is None:
                    continue
                if config.layers is not None and layer_name not in config.layers:
                    continue

                ip = intervention_points.get(layer_name)
                if ip is None:
                    continue

                # Get the module
                module = self._get_module_by_path(ip.module_path)
                if module is None:
                    continue

                # Create and register hook
                hook = self._create_steering_hook(vector, config)
                handle = module.register_forward_hook(hook)
                handles.append(handle)

            yield

        finally:
            # Clean up hooks
            for handle in handles:
                handle.remove()

    def _get_module_by_path(self, path: str) -> nn.Module | None:
        """Get a module by its dot-separated path."""
        module = self.model
        try:
            for attr in path.split("."):
                if attr.isdigit():
                    module = module[int(attr)]
                else:
                    module = getattr(module, attr)
            return module
        except (AttributeError, IndexError, KeyError):
            return None

    def _create_steering_hook(
        self,
        vector: torch.Tensor,
        config: SteeringConfig,
    ):
        """
        Create a forward hook that adds the steering vector.

        Args:
            vector: Steering vector to add
            config: Steering configuration

        Returns:
            Hook function
        """
        def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> torch.Tensor:
            # Prepare steering vector
            v = vector.to(output.device, output.dtype)

            # Normalize if configured
            if config.normalize:
                v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

            # Scale
            v = v * config.scale

            # Broadcast to match output shape
            while v.dim() < output.dim():
                v = v.unsqueeze(0)

            # Handle temporal modes for sequences
            if output.dim() >= 3 and config.temporal_mode != "per_step":
                if config.temporal_mode == "last":
                    # Only steer last position
                    steered = output.clone()
                    steered[:, -1] = steered[:, -1] + v.squeeze(1)
                    return steered
                elif config.temporal_mode == "first":
                    # Only steer first position
                    steered = output.clone()
                    steered[:, 0] = steered[:, 0] + v.squeeze(1)
                    return steered
                # "aggregate" or "per_step" - apply to all positions

            return output + v

        return hook

    @property
    def hidden_size(self) -> int:
        """Get the model's hidden dimension."""
        raise NotImplementedError("Subclass must implement hidden_size property")

    @property
    def num_layers(self) -> int:
        """Get the number of steerable layers."""
        return len(self.get_intervention_points())

    @classmethod
    def list_registered(cls) -> Dict[str, type["BaseAdapter"]]:
        """List all registered adapters."""
        return dict(cls._REGISTRY)

    @classmethod
    def get(cls, name: str) -> type["BaseAdapter"]:
        """
        Get a registered adapter class by name.

        Args:
            name: Adapter name

        Returns:
            Adapter class

        Raises:
            AdapterError: If adapter not found
        """
        try:
            return cls._REGISTRY[name]
        except KeyError as exc:
            raise AdapterError(f"Unknown adapter: {name!r}") from exc

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name!r}, device={self.device!r})"
