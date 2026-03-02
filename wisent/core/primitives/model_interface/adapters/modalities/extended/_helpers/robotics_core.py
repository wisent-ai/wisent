"""
Core robotics adapter: config, initialization, detection, encoding, decoding.

Extracted from robotics.py to keep files under 300 lines.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import numpy as np

from wisent.core.primitives.model_interface.adapters.base import (
    BaseAdapter,
    AdapterError,
    InterventionPoint,
    SteeringConfig,
)
from wisent.core.primitives.models.modalities import (
    Modality,
    RobotState,
    RobotAction,
    RobotTrajectory,
)
from wisent.core.utils.config_tools.constants import POLICY_LAYER_COUNT
from wisent.core.utils.infra_tools.errors import UnknownTypeError
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations

logger = logging.getLogger(__name__)


@dataclass
class RoboticsSteeringConfig(SteeringConfig):
    """
    Extended steering config for robotics.

    Attributes:
        action_space: Type of action space ("continuous", "discrete")
        safety_clamp: Whether to clamp actions to safe bounds
        action_bounds: Min/max bounds for continuous actions
        trajectory_mode: How to apply steering across trajectory
            - "per_step": Apply at each timestep independently
            - "cumulative": Accumulate steering effect
            - "goal_conditioned": Modulate based on goal proximity
        smooth_factor: Smoothing for action changes (0 = no smoothing)
    """
    action_space: str = "continuous"
    safety_clamp: bool = True
    action_bounds: tuple[float, float] | None = None
    trajectory_mode: str = "per_step"
    smooth_factor: float = 0.0


InputType = Union[RobotState, RobotTrajectory]


class RoboticsAdapterCore(BaseAdapter[InputType, RobotAction]):
    """
    Core adapter for robot policy network steering.

    Contains initialization, model loading, layer resolution,
    and property methods.
    """

    name = "robotics"
    modality = Modality.ROBOT_STATE

    POLICY_TYPE_MLP = "mlp"
    POLICY_TYPE_TRANSFORMER = "transformer"
    POLICY_TYPE_DIFFUSION = "diffusion"
    POLICY_TYPE_VLA = "vla"
    POLICY_TYPE_GENERIC = "generic"

    def __init__(
        self,
        model: nn.Module | None = None,
        model_name: str | None = None,
        device: str | None = None,
        policy_type: str | None = None,
        state_dim: int | None = None,
        action_dim: int | None = None,
        observation_encoder: nn.Module | None = None,
        action_decoder: nn.Module | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, model_name=model_name, device=device, **kwargs)
        self._policy_type = policy_type
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._observation_encoder = observation_encoder
        self._action_decoder = action_decoder
        self._policy_layers: List[nn.Module] | None = None
        self._hidden_size: int | None = None

    def _detect_policy_type(self) -> str:
        """Detect the type of policy network."""
        if self._policy_type:
            return self._policy_type
        if self._model is None:
            return self.POLICY_TYPE_GENERIC
        model_class = self._model.__class__.__name__.lower()
        if "transformer" in model_class or "attention" in model_class:
            return self.POLICY_TYPE_TRANSFORMER
        elif "diffusion" in model_class:
            return self.POLICY_TYPE_DIFFUSION
        has_linear_layers = any(
            isinstance(m, nn.Linear) for m in self._model.modules()
        )
        has_attention = any(
            "attention" in type(m).__name__.lower() for m in self._model.modules()
        )
        if has_attention:
            return self.POLICY_TYPE_TRANSFORMER
        elif has_linear_layers:
            return self.POLICY_TYPE_MLP
        return self.POLICY_TYPE_GENERIC

    def _load_model(self) -> nn.Module:
        """Load the policy model."""
        if self._model is not None:
            return self._model
        if self.model_name is None:
            raise AdapterError(
                "RoboticsAdapter requires either a model instance or model_name"
            )
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                **self._kwargs,
            )
            if self.device:
                model = model.to(self.device)
            return model
        except Exception:
            pass
        raise AdapterError(
            f"Could not load model {self.model_name}. "
            "For robotics, typically pass the model directly via model= parameter."
        )

    def _resolve_policy_layers(self) -> List[nn.Module]:
        """Find steerable layers in the policy network."""
        if self._policy_layers is not None:
            return self._policy_layers
        m = self.model
        policy_type = self._detect_policy_type()
        if policy_type == self.POLICY_TYPE_MLP:
            layers = []
            for name, module in m.named_modules():
                if isinstance(module, nn.Linear):
                    layers.append((name, module))
            self._policy_layers = [l[1] for l in layers]
            self._layer_names = [l[0] for l in layers]
            return self._policy_layers
        elif policy_type == self.POLICY_TYPE_TRANSFORMER:
            candidates = [
                "transformer.layers",
                "encoder.layers",
                "decoder.layers",
                "blocks",
                "layers",
            ]
            for path in candidates:
                obj = m
                try:
                    for attr in path.split("."):
                        obj = getattr(obj, attr)
                    if isinstance(obj, nn.ModuleList):
                        self._policy_layers = list(obj)
                        self._layer_path = path
                        return self._policy_layers
                except AttributeError:
                    continue
        layers = []
        for name, module in m.named_modules():
            if list(module.parameters(recurse=False)):
                layers.append((name, module))
        self._policy_layers = [l[1] for l in layers[-POLICY_LAYER_COUNT:]]
        self._layer_names = [l[0] for l in layers[-POLICY_LAYER_COUNT:]]
        return self._policy_layers

    @property
    def hidden_size(self) -> int:
        """Get the policy's hidden dimension."""
        if self._hidden_size is not None:
            return self._hidden_size
        if hasattr(self.model, "config"):
            config = self.model.config
            self._hidden_size = (
                getattr(config, "hidden_size", None)
                or getattr(config, "d_model", None)
            )
            if self._hidden_size:
                return self._hidden_size
        for p in self.model.parameters():
            if p.ndim == 2:
                self._hidden_size = max(p.shape)
                return self._hidden_size
        raise AdapterError("Could not determine hidden size")

    @property
    def state_dim(self) -> int:
        """Get the observation/state dimension."""
        if self._state_dim is not None:
            return self._state_dim
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                self._state_dim = module.in_features
                return self._state_dim
        raise AdapterError("Could not determine state dimension")

    @property
    def action_dim(self) -> int:
        """Get the action dimension."""
        if self._action_dim is not None:
            return self._action_dim
        layers = list(self.model.modules())
        for module in reversed(layers):
            if isinstance(module, nn.Linear):
                self._action_dim = module.out_features
                return self._action_dim
        raise AdapterError("Could not determine action dimension")
