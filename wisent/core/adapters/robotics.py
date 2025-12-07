"""
Robotics adapter for policy network steering.

Enables contrastive steering for robot behavior:
- Safe/gentle manipulation
- Goal-directed behavior steering
- Motion style control
- Human-robot interaction patterns
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
import logging

import torch
import torch.nn as nn
import numpy as np

from wisent.core.adapters.base import (
    BaseAdapter,
    AdapterError,
    InterventionPoint,
    SteeringConfig,
)
from wisent.core.modalities import (
    Modality,
    RobotState,
    RobotAction,
    RobotTrajectory,
)
from wisent.core.errors import UnknownTypeError
from wisent.core.activations.core.atoms import LayerActivations

__all__ = ["RoboticsAdapter", "RoboticsSteeringConfig"]

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


class RoboticsAdapter(BaseAdapter[InputType, RobotAction]):
    """
    Adapter for robot policy network steering.

    Supports various policy architectures:
    - MLP policies (standard RL)
    - Transformer policies (decision transformer, etc.)
    - Diffusion policies
    - Vision-language-action models

    The adapter hooks into the policy network to steer the action
    distribution toward desired behaviors (e.g., gentleness, safety).

    Example:
        >>> adapter = RoboticsAdapter(model=my_policy_network)
        >>> state = RobotState(joint_positions=np.array([...]))
        >>> # Steer toward gentle manipulation
        >>> action = adapter.act(state, steering_vectors=gentle_vectors)
    """

    name = "robotics"
    modality = Modality.ROBOT_STATE

    # Supported policy types
    POLICY_TYPE_MLP = "mlp"
    POLICY_TYPE_TRANSFORMER = "transformer"
    POLICY_TYPE_DIFFUSION = "diffusion"
    POLICY_TYPE_VLA = "vla"  # Vision-Language-Action
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
        """
        Initialize the robotics adapter.

        Args:
            model: Policy network (required)
            model_name: Optional model identifier
            device: Target device
            policy_type: Type of policy architecture
            state_dim: Dimension of state/observation space
            action_dim: Dimension of action space
            observation_encoder: Optional separate observation encoder
            action_decoder: Optional separate action decoder
            **kwargs: Additional arguments
        """
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

        # Check for common MLP patterns
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

        # Try to load from common robotics model hubs
        try:
            # Try HuggingFace
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
            # For MLP, collect all Linear layers
            layers = []
            for name, module in m.named_modules():
                if isinstance(module, nn.Linear):
                    layers.append((name, module))
            self._policy_layers = [l[1] for l in layers]
            self._layer_names = [l[0] for l in layers]
            return self._policy_layers

        elif policy_type == self.POLICY_TYPE_TRANSFORMER:
            # Look for transformer layers
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

        # Fallback: collect all modules with parameters
        layers = []
        for name, module in m.named_modules():
            if list(module.parameters(recurse=False)):
                layers.append((name, module))
        self._policy_layers = [l[1] for l in layers[-5:]]  # Last 5 layers
        self._layer_names = [l[0] for l in layers[-5:]]
        return self._policy_layers

    @property
    def hidden_size(self) -> int:
        """Get the policy's hidden dimension."""
        if self._hidden_size is not None:
            return self._hidden_size

        # Try to infer from model config
        if hasattr(self.model, "config"):
            config = self.model.config
            self._hidden_size = (
                getattr(config, "hidden_size", None)
                or getattr(config, "d_model", None)
            )
            if self._hidden_size:
                return self._hidden_size

        # Infer from layer dimensions
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

        # Try to infer from first layer
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

        # Try to infer from last layer
        layers = list(self.model.modules())
        for module in reversed(layers):
            if isinstance(module, nn.Linear):
                self._action_dim = module.out_features
                return self._action_dim

        raise AdapterError("Could not determine action dimension")

    def encode(self, content: InputType) -> torch.Tensor:
        """
        Encode robot state/trajectory to latent representation.

        Args:
            content: RobotState or RobotTrajectory

        Returns:
            State embedding tensor
        """
        if isinstance(content, RobotTrajectory):
            # Encode trajectory as sequence
            states = torch.stack([s.to_tensor() for s in content.states])
            if states.dim() == 2:
                states = states.unsqueeze(0)  # Add batch dim
        else:
            states = content.to_tensor()
            if states.dim() == 1:
                states = states.unsqueeze(0)  # Add batch dim

        states = states.float().to(self.device)

        # Use observation encoder if available
        if self._observation_encoder is not None:
            with torch.no_grad():
                return self._observation_encoder(states)

        # For MLP policies, the first few layers act as encoder
        policy_type = self._detect_policy_type()
        if policy_type == self.POLICY_TYPE_MLP:
            with torch.no_grad():
                x = states
                layers = self._resolve_policy_layers()
                # Pass through first half of layers as "encoder"
                n_encoder = len(layers) // 2
                for layer in layers[:n_encoder]:
                    if isinstance(layer, nn.Linear):
                        x = layer(x)
                        x = torch.relu(x)
                return x

        # For transformer policies
        elif policy_type == self.POLICY_TYPE_TRANSFORMER:
            with torch.no_grad():
                if hasattr(self.model, "encode"):
                    return self.model.encode(states)
                outputs = self.model(states, output_hidden_states=True)
                if hasattr(outputs, "last_hidden_state"):
                    return outputs.last_hidden_state
                return outputs.hidden_states[-1]

        # Generic: just return input
        return states

    def decode(self, latent: torch.Tensor) -> RobotAction:
        """
        Decode latent representation to action.

        Args:
            latent: Encoded state/hidden representation

        Returns:
            Robot action
        """
        if self._action_decoder is not None:
            with torch.no_grad():
                action_tensor = self._action_decoder(latent)
        else:
            # Use second half of MLP or output head
            policy_type = self._detect_policy_type()
            if policy_type == self.POLICY_TYPE_MLP:
                layers = self._resolve_policy_layers()
                n_encoder = len(layers) // 2
                x = latent
                for i, layer in enumerate(layers[n_encoder:]):
                    if isinstance(layer, nn.Linear):
                        x = layer(x)
                        if i < len(layers[n_encoder:]) - 1:
                            x = torch.relu(x)
                action_tensor = x
            else:
                # For transformer, use final projection
                if hasattr(self.model, "action_head"):
                    action_tensor = self.model.action_head(latent)
                else:
                    action_tensor = latent[..., :self.action_dim]

        # Convert to RobotAction
        action_np = action_tensor.detach().cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np.squeeze(0)

        return RobotAction(raw_action=action_np)

    def get_intervention_points(self) -> List[InterventionPoint]:
        """Get available intervention points in the policy network."""
        points = []
        layers = self._resolve_policy_layers()
        policy_type = self._detect_policy_type()

        if policy_type == self.POLICY_TYPE_MLP:
            for i, name in enumerate(self._layer_names):
                recommended = i >= len(layers) // 2  # Recommend later layers
                points.append(
                    InterventionPoint(
                        name=f"layer.{i}",
                        module_path=name,
                        description=f"Policy layer {i} ({name})",
                        recommended=recommended,
                    )
                )
        else:
            for i in range(len(layers)):
                recommended = i >= len(layers) // 2
                points.append(
                    InterventionPoint(
                        name=f"layer.{i}",
                        module_path=f"{self._layer_path}.{i}" if hasattr(self, "_layer_path") else f"layers.{i}",
                        description=f"Policy layer {i}",
                        recommended=recommended,
                    )
                )

        return points

    def extract_activations(
        self,
        content: InputType,
        layers: List[str] | None = None,
    ) -> LayerActivations:
        """
        Extract activations from policy layers.

        Args:
            content: Robot state or trajectory
            layers: Layer names to extract (None = all)

        Returns:
            LayerActivations with per-layer tensors
        """
        all_points = {ip.name: ip for ip in self.get_intervention_points()}
        target_layers = layers if layers else list(all_points.keys())

        activations: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                activations[layer_name] = output.detach().cpu()
            return hook

        try:
            for layer_name in target_layers:
                if layer_name not in all_points:
                    continue
                ip = all_points[layer_name]
                module = self._get_module_by_path(ip.module_path)
                if module is not None:
                    handle = module.register_forward_hook(make_hook(layer_name))
                    hooks.append(handle)

            # Forward pass
            _ = self.forward_policy(content)

        finally:
            for handle in hooks:
                handle.remove()

        return LayerActivations(activations)

    def forward_policy(self, content: InputType) -> torch.Tensor:
        """
        Forward pass through the policy network.

        Args:
            content: Robot state or trajectory

        Returns:
            Raw action tensor
        """
        if isinstance(content, RobotTrajectory):
            states = torch.stack([s.to_tensor() for s in content.states])
        else:
            states = content.to_tensor()

        if states.dim() == 1:
            states = states.unsqueeze(0)

        states = states.float().to(self.device)

        with torch.no_grad():
            return self.model(states)

    def forward_with_steering(
        self,
        content: InputType,
        steering_vectors: LayerActivations,
        config: SteeringConfig | RoboticsSteeringConfig | None = None,
    ) -> RobotAction:
        """
        Get action with steering applied.

        Args:
            content: Robot state or trajectory
            steering_vectors: Steering vectors to apply
            config: Steering configuration

        Returns:
            Steered robot action
        """
        config = config or RoboticsSteeringConfig()

        with self._steering_hooks(steering_vectors, config):
            action_tensor = self.forward_policy(content)

        # Apply safety clamping if configured
        if isinstance(config, RoboticsSteeringConfig) and config.safety_clamp:
            if config.action_bounds:
                action_tensor = torch.clamp(
                    action_tensor,
                    config.action_bounds[0],
                    config.action_bounds[1],
                )

        action_np = action_tensor.detach().cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np.squeeze(0)

        return RobotAction(raw_action=action_np)

    def _generate_unsteered(
        self,
        content: InputType,
        **kwargs: Any,
    ) -> RobotAction:
        """Get action without steering."""
        action_tensor = self.forward_policy(content)
        action_np = action_tensor.detach().cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np.squeeze(0)
        return RobotAction(raw_action=action_np)

    def act(
        self,
        state: RobotState,
        steering_vectors: LayerActivations | None = None,
        config: RoboticsSteeringConfig | None = None,
    ) -> RobotAction:
        """
        Convenience method to get an action from a state.

        Args:
            state: Current robot state
            steering_vectors: Optional steering vectors
            config: Steering configuration

        Returns:
            Action to execute
        """
        if steering_vectors is not None:
            return self.forward_with_steering(state, steering_vectors, config)
        return self._generate_unsteered(state)

    def rollout(
        self,
        initial_state: RobotState,
        env_step_fn: Callable[[RobotAction], RobotState],
        num_steps: int,
        steering_vectors: LayerActivations | None = None,
        config: RoboticsSteeringConfig | None = None,
    ) -> RobotTrajectory:
        """
        Execute a trajectory with optional steering.

        Args:
            initial_state: Starting state
            env_step_fn: Function that takes action and returns next state
            num_steps: Number of steps to execute
            steering_vectors: Optional steering vectors
            config: Steering configuration

        Returns:
            Executed trajectory
        """
        states = [initial_state]
        actions = []
        state = initial_state

        for _ in range(num_steps):
            action = self.act(state, steering_vectors, config)
            actions.append(action)
            state = env_step_fn(action)
            states.append(state)

        return RobotTrajectory(
            states=tuple(states),
            actions=tuple(actions),
        )

    def compute_trajectory_steering_vector(
        self,
        positive_trajectory: RobotTrajectory,
        negative_trajectory: RobotTrajectory,
        layer: str,
        aggregation: str = "mean",
    ) -> torch.Tensor:
        """
        Compute steering vector from trajectory pairs.

        Args:
            positive_trajectory: Desired behavior trajectory
            negative_trajectory: Undesired behavior trajectory
            layer: Layer to extract from
            aggregation: How to aggregate ("mean", "last", "weighted")

        Returns:
            Steering vector
        """
        # Extract activations for each state in trajectories
        pos_activations = []
        neg_activations = []

        for state in positive_trajectory.states:
            acts = self.extract_activations(state, [layer])
            pos_activations.append(acts[layer])

        for state in negative_trajectory.states:
            acts = self.extract_activations(state, [layer])
            neg_activations.append(acts[layer])

        pos_tensor = torch.stack(pos_activations)
        neg_tensor = torch.stack(neg_activations)

        if aggregation == "mean":
            pos_agg = pos_tensor.mean(dim=0)
            neg_agg = neg_tensor.mean(dim=0)
        elif aggregation == "last":
            pos_agg = pos_tensor[-1]
            neg_agg = neg_tensor[-1]
        elif aggregation == "weighted":
            # Weight later timesteps more
            weights = torch.linspace(0.1, 1.0, len(pos_activations))
            weights = weights / weights.sum()
            pos_agg = (pos_tensor * weights.view(-1, 1, 1)).sum(dim=0)
            weights = torch.linspace(0.1, 1.0, len(neg_activations))
            weights = weights / weights.sum()
            neg_agg = (neg_tensor * weights.view(-1, 1, 1)).sum(dim=0)
        else:
            raise UnknownTypeError(entity_type="aggregation", value=aggregation, valid_values=["mean", "max", "min", "last", "weighted"])

        return pos_agg.squeeze() - neg_agg.squeeze()
