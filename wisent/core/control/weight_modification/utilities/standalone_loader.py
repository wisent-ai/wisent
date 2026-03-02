"""
Standalone loader for GROM and TETNO models.

This file is saved alongside exported models so users can load them
without installing the full wisent package.

Usage:
    from standalone_loader import load_model
    model, tokenizer, hooks = load_model("./my_grom_model")
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from wisent.core.constants import (
    DEFAULT_LAYER, GROM_HIDDEN_DIM, GROM_ROUTER_TEMPERATURE,
    GROM_ROUTER_HIDDEN_DIM, GROM_MAX_ALPHA, DEFAULT_STRENGTH,
    GROM_GATING_SHRINK_FACTOR,
)

from wisent.core.weight_modification._standalone_loader_helpers import (
    TETNOHooks,
    load_model as _load_model,
)


class GatingNetwork(nn.Module):
    """Learned gating network matching GROM architecture."""
    def __init__(self, input_dim: int, hidden_dim: int = GROM_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // GROM_GATING_SHRINK_FACTOR),
            nn.GELU(),
            nn.Linear(hidden_dim // GROM_GATING_SHRINK_FACTOR, 1),
        )

    def forward(self, x: torch.Tensor,
                temperature: float = GROM_ROUTER_TEMPERATURE) -> torch.Tensor:
        logit = self.net(x)
        return torch.sigmoid(logit / temperature)


class IntensityNetwork(nn.Module):
    """Predicts per-layer steering intensity matching GROM architecture."""
    def __init__(self, input_dim: int, num_layers: int,
                 hidden_dim: int = GROM_ROUTER_HIDDEN_DIM, max_alpha: float = GROM_MAX_ALPHA):
        super().__init__()
        self.max_alpha = max_alpha
        self.num_layers = num_layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_layers),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        raw = self.net(x)
        return self.max_alpha * torch.sigmoid(raw)


class GROMHooks:
    """Runtime hooks for GROM dynamic steering."""

    def __init__(self, model, grom_data: Dict[str, Any],
                 base_strength: float = DEFAULT_STRENGTH):
        self.model = model
        self.grom_data = grom_data
        self.base_strength = base_strength

        self._hooks = []
        self._current_gate = None
        self._current_intensities = None

        # Get model layers
        if hasattr(model, "model"):
            self._layers = model.model.layers
        elif hasattr(model, "transformer"):
            self._layers = model.transformer.h
        else:
            self._layers = model.layers

        # Setup layer mapping
        self.layer_order = grom_data["layer_order"]
        self._layer_name_to_idx = {}
        for layer_name in self.layer_order:
            try:
                idx = int(str(layer_name).split("_")[-1])
                self._layer_name_to_idx[layer_name] = idx
            except (ValueError, IndexError):
                pass

        # Find sensor layer
        sensor_layer = grom_data.get("sensor_layer")
        if sensor_layer is None:
            sensor_layer = self._layer_name_to_idx.get(
                self.layer_order[len(self.layer_order) // 2], DEFAULT_LAYER)
        self._sensor_layer_idx = sensor_layer

        # Load directions
        self.directions = {
            k: v.to(model.device) for k, v in grom_data["directions"].items()}
        self.direction_weights = {
            k: v.to(model.device)
            for k, v in grom_data["direction_weights"].items()}

        # Load networks if present
        self.gate_network = None
        self.intensity_network = None

        if "gate_network_state" in grom_data:
            config = grom_data["gate_network_config"]
            self.gate_network = GatingNetwork(
                config["input_dim"], config.get("hidden_dim", GROM_HIDDEN_DIM))
            self.gate_network.load_state_dict(grom_data["gate_network_state"])
            self.gate_network = self.gate_network.to(model.device).eval()

        if "intensity_network_state" in grom_data:
            config = grom_data["intensity_network_config"]
            self.intensity_network = IntensityNetwork(
                config["input_dim"], config["num_layers"],
                config.get("hidden_dim", GROM_ROUTER_HIDDEN_DIM),
                max_alpha=grom_data.get("max_alpha", GROM_MAX_ALPHA))
            self.intensity_network.load_state_dict(
                grom_data["intensity_network_state"])
            self.intensity_network = (
                self.intensity_network.to(model.device).eval())

        self.gate_temperature = grom_data.get("gate_temperature", GROM_ROUTER_TEMPERATURE)
        self.max_alpha = grom_data.get("max_alpha", GROM_MAX_ALPHA)

    def _get_effective_direction(self, layer_name: str) -> torch.Tensor:
        """Get weighted combination of directions for a layer."""
        dirs = self.directions[layer_name]
        weights = self.direction_weights[layer_name]
        weights_norm = F.softmax(weights, dim=0)
        return (dirs * weights_norm.unsqueeze(1)).sum(dim=0)

    def install(self) -> None:
        """Install forward hooks."""
        self.remove()
        if self._sensor_layer_idx < len(self._layers):
            hook = self._layers[self._sensor_layer_idx].register_forward_hook(
                self._sensor_hook)
            self._hooks.append(hook)
        for layer_name in self.layer_order:
            layer_idx = self._layer_name_to_idx.get(layer_name)
            if layer_idx is not None and layer_idx < len(self._layers):
                hook = self._layers[layer_idx].register_forward_hook(
                    lambda m, i, o, ln=layer_name: self._steering_hook(
                        m, i, o, ln))
                self._hooks.append(hook)

    def remove(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._current_gate = None
        self._current_intensities = None

    def _sensor_hook(self, module, input, output):
        """Capture sensor activation and compute gate/intensity."""
        hidden = output[0] if isinstance(output, tuple) else output
        sensor_hidden = (hidden[:, -1, :]
                         if hidden.dim() == 3 else hidden)

        if self.gate_network is not None:
            with torch.no_grad():
                self._current_gate = self.gate_network(sensor_hidden.float())
        else:
            self._current_gate = torch.ones(
                sensor_hidden.shape[0], 1, device=sensor_hidden.device)

        if self.intensity_network is not None:
            with torch.no_grad():
                intensities = self.intensity_network(sensor_hidden.float())
                self._current_intensities = {
                    layer: intensities[:, i:i+1]
                    for i, layer in enumerate(self.layer_order)
                }
        else:
            self._current_intensities = {
                layer: torch.ones(
                    sensor_hidden.shape[0], 1, device=sensor_hidden.device)
                for layer in self.layer_order
            }
        return output

    def _steering_hook(self, module, input, output, layer_name):
        """Apply dynamic steering."""
        if self._current_gate is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None
        direction = self._get_effective_direction(layer_name).to(hidden.device)
        gate = self._current_gate.to(hidden.device)
        intensity = self._current_intensities.get(
            layer_name, torch.ones_like(gate)).to(hidden.device)
        if hidden.dim() == 3:
            gate = gate.view(-1, 1, 1)
            intensity = intensity.view(-1, 1, 1)
            direction = direction.view(1, 1, -1)
        else:
            gate = gate.view(-1, 1)
            intensity = intensity.view(-1, 1)
            direction = direction.view(1, -1)
        hidden = hidden + gate * intensity * self.base_strength * direction
        return (hidden,) + rest if rest else hidden

    def get_current_gate(self) -> Optional[float]:
        if self._current_gate is not None:
            return self._current_gate.mean().item()
        return None

    def get_current_intensities(self) -> Optional[Dict[str, float]]:
        if self._current_intensities is not None:
            return {k: v.mean().item()
                    for k, v in self._current_intensities.items()}
        return None


def load_model(
    model_path: str,
    device_map: str = "auto",
    torch_dtype=None,
    install_hooks: bool = True,
) -> Tuple[Any, Any, Optional[Any]]:
    """
    Load a GROM or TETNO steered model.

    Args:
        model_path: Path to model directory or HuggingFace repo
        device_map: Device placement
        torch_dtype: Model dtype
        install_hooks: Whether to install dynamic steering hooks

    Returns:
        Tuple of (model, tokenizer, hooks)
    """
    return _load_model(
        model_path=model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        install_hooks=install_hooks,
        GROMHooksClass=GROMHooks,
        TETNOHooksClass=TETNOHooks,
    )
