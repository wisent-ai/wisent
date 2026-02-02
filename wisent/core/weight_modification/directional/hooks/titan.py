"""TITAN runtime hooks and weight modification."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from wisent.core.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from torch.nn import Module

_LOG = setup_logger(__name__)


class TITANRuntimeHooks:
    """Runtime hook system for TITAN dynamic steering."""

    def __init__(self, model: Module, titan_result, base_strength: float = 1.0, gate_threshold: float = 0.5, use_soft_gating: bool = True):
        self.model = model
        self.titan_result = titan_result
        self.base_strength = base_strength
        self.gate_threshold = gate_threshold
        self.use_soft_gating = use_soft_gating
        self._hooks = []
        self._sensor_activation = None
        self._current_gate = None
        self._current_intensities = None
        if hasattr(model, "model"):
            self._layers = model.model.layers
        elif hasattr(model, "transformer"):
            self._layers = model.transformer.h
        else:
            self._layers = model.layers
        self._layer_name_to_idx = {}
        for layer_name in titan_result.layer_order:
            try:
                idx = int(str(layer_name).split("_")[-1])
                self._layer_name_to_idx[layer_name] = idx
            except (ValueError, IndexError):
                pass
        sensor_layer_name = titan_result.metadata.get("sensor_layer")
        self._sensor_layer_idx = self._layer_name_to_idx.get(sensor_layer_name, 15)

    def install(self) -> None:
        """Install forward hooks on the model."""
        self.remove()
        if self._sensor_layer_idx < len(self._layers):
            sensor_hook = self._layers[self._sensor_layer_idx].register_forward_hook(self._sensor_hook)
            self._hooks.append(sensor_hook)
        for layer_name in self.titan_result.layer_order:
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
        self._sensor_activation = None
        self._current_gate = None
        self._current_intensities = None

    def _sensor_hook(self, module, input, output):
        """Capture sensor layer activation and compute gate/intensities."""
        hidden_states = output[0] if isinstance(output, tuple) else output
        if hidden_states.dim() == 3:
            sensor_h = hidden_states[:, -1, :]
        else:
            sensor_h = hidden_states
        self._sensor_activation = sensor_h.detach()
        with torch.no_grad():
            if self.use_soft_gating:
                self._current_gate = self.titan_result.predict_gate(sensor_h)
            else:
                gate_value = self.titan_result.predict_gate(sensor_h)
                self._current_gate = (gate_value > self.gate_threshold).float()
            self._current_intensities = self.titan_result.predict_intensity(sensor_h)
        return output

    def _steering_hook(self, module, input, output, layer_name):
        """Apply dynamic steering to layer output."""
        if self._current_gate is None or self._current_intensities is None:
            return output
        hidden_states = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None
        direction = self.titan_result.get_effective_direction(layer_name).to(hidden_states.device)
        intensity = self._current_intensities.get(layer_name, torch.ones(1)).to(hidden_states.device)
        gate = self._current_gate.to(hidden_states.device)
        if hidden_states.dim() == 3:
            gate = gate.view(-1, 1, 1)
            intensity = intensity.view(-1, 1, 1)
            direction = direction.view(1, 1, -1)
        elif hidden_states.dim() == 2:
            gate = gate.view(-1, 1)
            intensity = intensity.view(-1, 1)
            direction = direction.view(1, -1)
        steering_delta = gate * intensity * self.base_strength * direction
        hidden_states = hidden_states + steering_delta
        return (hidden_states,) + rest if rest is not None else hidden_states

    def get_current_gate(self) -> float | None:
        """Get the current gate value."""
        return self._current_gate.mean().item() if self._current_gate is not None else None

    def get_current_intensities(self) -> dict | None:
        """Get current per-layer intensities."""
        return {k: v.mean().item() for k, v in self._current_intensities.items()} if self._current_intensities else None


def project_weights_titan(
    model: Module, titan_result, components: list[str] | None = None, base_strength: float = 1.0,
    use_learned_intensities: bool = True, verbose: bool = True
) -> dict[str, int]:
    """Bake TITAN effective directions into model weights using ADDITIVE steering."""
    from wisent.core.weight_modification.methods.additive import bake_steering_into_weights
    from wisent.core.activations.core.atoms import LayerActivations
    log = bind(_LOG, num_layers=len(titan_result.directions))
    if components is None:
        components = ["self_attn.o_proj", "mlp.down_proj"]
    effective_vectors, layer_weights = {}, {}
    for layer_name in titan_result.layer_order:
        eff_dir = titan_result.get_effective_direction(layer_name)
        try:
            layer_idx = int(str(layer_name).split("_")[-1])
        except (ValueError, IndexError):
            continue
        effective_vectors[layer_idx] = eff_dir
        if use_learned_intensities:
            dir_weights = titan_result.direction_weights.get(layer_name)
            weight = 1.0 + (dir_weights.max() - dir_weights.min()).item() if dir_weights is not None else 1.0
            layer_weights[layer_idx] = weight
    if verbose:
        print(f"\n{'='*60}\nTITAN WEIGHT MODIFICATION (ADDITIVE)\n{'='*60}")
        print(f"Layers: {len(effective_vectors)}, Components: {components}, Base strength: {base_strength}\n{'='*60}\n")
    weighted_vectors = {layer_idx: vec * layer_weights.get(layer_idx, 1.0) if use_learned_intensities else vec
                        for layer_idx, vec in effective_vectors.items()}
    steering_vectors = LayerActivations(weighted_vectors)
    stats = bake_steering_into_weights(model=model, steering_vectors=steering_vectors, alpha=base_strength, components=components, verbose=verbose)
    stats["titan_layers"] = len(titan_result.layer_order)
    stats["titan_directions_per_layer"] = titan_result.directions[titan_result.layer_order[0]].shape[0]
    stats["method"] = "additive"
    return stats
