"""TETNO runtime hooks and GROM convenience function."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
from wisent.core.utils.config_tools.constants import DEFAULT_LAYER, DEFAULT_STRENGTH, TETNO_GATE_TEMPERATURE, DEFAULT_LAYER_WEIGHT
from wisent.core.utils.cli.cli_logger import setup_logger, bind
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from torch.nn import Module

_LOG = setup_logger(__name__)


class TETNORuntimeHooks:
    """Runtime hooks for TETNO conditional steering."""

    def __init__(self, model: Module, tetno_result, base_strength: float = DEFAULT_STRENGTH, gate_temperature: float = TETNO_GATE_TEMPERATURE) -> None:
        self.model = model
        self.tetno_result = tetno_result
        self.base_strength = base_strength
        self.gate_temperature = gate_temperature
        self._hooks = []
        self._current_gate = None
        self._sensor_activation = None
        if hasattr(model, "model"):
            self._layers = model.model.layers
        elif hasattr(model, "transformer"):
            self._layers = model.transformer.h
        else:
            self._layers = model.layers
        sensor_layer = tetno_result.metadata.get("sensor_layer") if hasattr(tetno_result, "metadata") else None
        if sensor_layer is None:
            layer_indices = []
            for layer_name in tetno_result.behavior_vectors.keys():
                try:
                    idx = int(str(layer_name).split("_")[-1])
                    layer_indices.append(idx)
                except (ValueError, IndexError):
                    pass
            sensor_layer = layer_indices[len(layer_indices)//2] if layer_indices else DEFAULT_LAYER
        self._sensor_layer_idx = sensor_layer
        self._layer_name_to_idx = {}
        for layer_name in tetno_result.behavior_vectors.keys():
            try:
                idx = int(str(layer_name).split("_")[-1])
                self._layer_name_to_idx[layer_name] = idx
            except (ValueError, IndexError):
                pass

    def install(self) -> None:
        """Install forward hooks on the model."""
        self.remove()
        if self._sensor_layer_idx < len(self._layers):
            sensor_hook = self._layers[self._sensor_layer_idx].register_forward_hook(self._sensor_hook)
            self._hooks.append(sensor_hook)
        for layer_name in self.tetno_result.behavior_vectors.keys():
            layer_idx = self._layer_name_to_idx.get(layer_name)
            if layer_idx is not None and layer_idx < len(self._layers):
                steering_hook = self._layers[layer_idx].register_forward_hook(
                    lambda module, input, output, ln=layer_name: self._steering_hook(module, input, output, ln))
                self._hooks.append(steering_hook)

    def remove(self) -> None:
        """Remove all installed hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._current_gate = None
        self._sensor_activation = None

    def _sensor_hook(self, module, input, output):
        """Capture sensor layer activation and compute gate."""
        hidden_states = output[0] if isinstance(output, tuple) else output
        sensor_hidden = hidden_states[:, -1, :] if hidden_states.dim() == 3 else hidden_states
        self._sensor_activation = sensor_hidden
        if hasattr(self.tetno_result, 'compute_gate'):
            self._current_gate = self.tetno_result.compute_gate(sensor_hidden, self.gate_temperature)
        else:
            h_norm = F.normalize(sensor_hidden, p=2, dim=-1)
            c_norm = F.normalize(self.tetno_result.condition_vector.to(sensor_hidden.device), p=2, dim=-1)
            similarity = (h_norm * c_norm).sum(dim=-1)
            self._current_gate = torch.sigmoid((similarity - self.tetno_result.optimal_threshold) / self.gate_temperature)
        return output

    def _steering_hook(self, module, input, output, layer_name):
        """Apply conditional steering to layer output."""
        if self._current_gate is None:
            return output
        hidden_states = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None
        behavior_vector = self.tetno_result.behavior_vectors.get(layer_name)
        if behavior_vector is None:
            return output
        behavior_vector = behavior_vector.to(hidden_states.device)
        layer_scale = self.tetno_result.layer_scales.get(layer_name, DEFAULT_LAYER_WEIGHT)
        gate = self._current_gate.to(hidden_states.device)
        if hidden_states.dim() == 3:
            gate = gate.view(-1, 1, 1)
            behavior_vector = behavior_vector.view(1, 1, -1)
        elif hidden_states.dim() == 2:
            gate = gate.view(-1, 1)
            behavior_vector = behavior_vector.view(1, -1)
        steering_delta = gate * self.base_strength * layer_scale * behavior_vector
        hidden_states = hidden_states + steering_delta
        return (hidden_states,) + rest if rest is not None else hidden_states

    def get_current_gate(self) -> float | None:
        """Get the current gate value."""
        return self._current_gate.mean().item() if self._current_gate is not None else None


def apply_grom_steering(
    model: Module, grom_result, mode: str = "hybrid", base_strength: float = DEFAULT_STRENGTH,
    components: list[str] | None = None, verbose: bool = True
) -> dict:
    """Apply GROM steering to a model with the specified mode."""
    from .grom import GROMRuntimeHooks, project_weights_grom
    result = {}
    if mode in ("static", "hybrid"):
        stats = project_weights_grom(model=model, grom_result=grom_result, components=components,
                                       base_strength=base_strength if mode == "static" else 1.0,
                                       use_learned_intensities=True, verbose=verbose)
        result["stats"] = stats
    if mode in ("dynamic", "hybrid"):
        hooks = GROMRuntimeHooks(model=model, grom_result=grom_result, base_strength=base_strength, use_soft_gating=True)
        hooks.install()
        result["hooks"] = hooks
        if verbose:
            print(f"\nGROM Runtime Hooks installed\n  Sensor layer: {hooks._sensor_layer_idx}\n  Steering layers: {len(grom_result.layer_order)}\n  Mode: {mode}")
    return result
