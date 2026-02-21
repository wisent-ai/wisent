"""Steering hook mixin and utility methods for BaseAdapter.

Extracted from base.py to keep file under 300 lines.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict

import torch
import torch.nn as nn

from wisent.core.activations.core.atoms import LayerActivations


class SteeringHookMixin:
    """Mixin providing steering hook management and utility methods for BaseAdapter."""

    @contextmanager
    def _steering_hooks(
        self,
        steering_vectors: LayerActivations,
        config: Any = None,
    ):
        """
        Context manager that registers steering hooks and cleans up after.

        Args:
            steering_vectors: Steering vectors to apply
            config: Steering configuration

        Yields:
            None (hooks are active within context)
        """
        from wisent.core.adapters.base import SteeringConfig
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

                # Create and register hook (pass layer_name for per-layer config)
                hook = self._create_steering_hook(vector, config, layer_name)
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

    def _create_steering_hook(self, vector: torch.Tensor, config: Any, layer_name: str = None):
        """Create a forward hook that adds the steering vector."""
        # Get per-layer method if config.method is a dict
        method = config.method.get(layer_name) if isinstance(config.method, dict) else config.method
        scale = config.scale.get(layer_name, 1.0) if isinstance(config.scale, dict) else config.scale

        def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> torch.Tensor:
            if method is not None:
                if output.dim() >= 3:
                    steered = output.clone()
                    steered[:, -1] = method.transform(output[:, -1])
                    return steered
                return method.transform(output)
            # Default: linear steering
            v = vector.to(output.device, output.dtype)
            if config.normalize:
                v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
            v = v * scale
            while v.dim() < output.dim():
                v = v.unsqueeze(0)
            if output.dim() >= 3 and config.temporal_mode != "per_step":
                steered = output.clone()
                if config.temporal_mode == "last":
                    steered[:, -1] = steered[:, -1] + v.squeeze(1)
                elif config.temporal_mode == "first":
                    steered[:, 0] = steered[:, 0] + v.squeeze(1)
                return steered
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
    def list_registered(cls) -> Dict[str, type]:
        """List all registered adapters."""
        return dict(cls._REGISTRY)

    @classmethod
    def get(cls, name: str) -> type:
        """
        Get a registered adapter class by name.

        Args:
            name: Adapter name

        Returns:
            Adapter class

        Raises:
            AdapterError: If adapter not found
        """
        from wisent.core.adapters.base import AdapterError
        try:
            return cls._REGISTRY[name]
        except KeyError as exc:
            raise AdapterError(f"Unknown adapter: {name!r}") from exc

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name!r}, device={self.device!r})"
