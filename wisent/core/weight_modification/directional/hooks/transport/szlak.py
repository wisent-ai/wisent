"""SZLAK runtime hooks for geodesic OT steering at inference time."""

from __future__ import annotations

import torch
from typing import Dict, TYPE_CHECKING
from wisent.core.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from torch.nn import Module

_LOG = setup_logger(__name__)


class SzlakRuntimeHooks:
    """
    Runtime hook system for SZLAK geodesic OT steering.

    Installs forward hooks on transformer layers that apply per-source
    inverse-distance-weighted displacement interpolation at inference.
    """

    def __init__(
        self,
        model: Module,
        source_points: Dict[int, torch.Tensor],
        displacements: Dict[int, torch.Tensor],
        inference_k: int = 5,
        base_strength: float = 1.0,
    ):
        self.model = model
        self.source_points = source_points
        self.displacements = displacements
        self.inference_k = inference_k
        self.base_strength = base_strength
        self._hooks = []

        if hasattr(model, "model"):
            self._layers = model.model.layers
        elif hasattr(model, "transformer"):
            self._layers = model.transformer.h
        else:
            self._layers = model.layers

    def install(self) -> None:
        """Install forward hooks on steered layers."""
        self.remove()
        for layer_idx in self.source_points:
            if layer_idx < len(self._layers):
                hook = self._layers[layer_idx].register_forward_hook(
                    lambda mod, inp, out, li=layer_idx: self._szlak_hook(
                        mod, inp, out, li,
                    ),
                )
                self._hooks.append(hook)

    def remove(self) -> None:
        """Remove all installed hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _szlak_hook(self, module, input, output, layer_idx: int):
        """Apply geodesic OT displacement to layer output."""
        hidden_states = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None

        h_new = self._apply_displacement(hidden_states, layer_idx)

        if rest is not None:
            return (h_new,) + rest
        return h_new

    def _apply_displacement(
        self, hidden_states: torch.Tensor, layer_idx: int,
    ) -> torch.Tensor:
        """k-NN lookup + IDW interpolation of precomputed displacements."""
        src = self.source_points[layer_idx].float()
        disp = self.displacements[layer_idx].float()

        original_shape = hidden_states.shape
        original_dtype = hidden_states.dtype

        if hidden_states.dim() == 3:
            b, s, hd = hidden_states.shape
            h_2d = hidden_states.reshape(b * s, hd)
        elif hidden_states.dim() == 2:
            h_2d = hidden_states
        else:
            h_2d = hidden_states.unsqueeze(0)

        h_float = h_2d.float()
        src_dev = src.to(h_float.device)
        disp_dev = disp.to(h_float.device)

        dists = torch.cdist(h_float, src_dev)
        K = min(self.inference_k, src_dev.shape[0])
        topk_dists, topk_idx = torch.topk(dists, K, dim=1, largest=False)

        eps = 1e-8
        inv_dists = 1.0 / (topk_dists + eps)
        weights = inv_dists / inv_dists.sum(dim=1, keepdim=True)

        batch_size = h_float.shape[0]
        topk_disps = disp_dev[topk_idx.reshape(-1)].reshape(
            batch_size, K, -1,
        )
        weighted_disp = (weights.unsqueeze(-1) * topk_disps).sum(dim=1)

        h_new = h_float + self.base_strength * weighted_disp
        h_new = h_new.to(original_dtype)
        return h_new.reshape(original_shape)

    @property
    def steered_layers(self) -> list[int]:
        """Return list of layer indices being steered."""
        return sorted(self.source_points.keys())


def project_weights_szlak(
    model: Module,
    steering_obj,
    base_strength: float = 1.0,
    components: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, int]:
    """Static projection: bake mean displacement into model weights."""
    from wisent.core.weight_modification.methods.additive import (
        bake_steering_into_weights,
    )
    from wisent.core.activations.core.atoms import LayerActivations

    if components is None:
        components = ["self_attn.o_proj", "mlp.down_proj"]

    mean_vectors = {}
    for layer_idx, disp in steering_obj.displacements.items():
        mean_vectors[layer_idx] = disp.mean(dim=0)

    if verbose:
        print(f"\n{'='*60}")
        print("SZLAK WEIGHT MODIFICATION (ADDITIVE — mean displacement)")
        print(f"{'='*60}")
        print(f"Layers: {len(mean_vectors)}, Strength: {base_strength}")

    steering_vectors = LayerActivations(mean_vectors)
    stats = bake_steering_into_weights(
        model=model, steering_vectors=steering_vectors,
        alpha=base_strength, components=components, verbose=verbose,
    )
    stats["szlak_layers"] = len(steering_obj.source_points)
    stats["method"] = "additive"
    return stats
