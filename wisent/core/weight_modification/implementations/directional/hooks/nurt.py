"""Concept Flow runtime hooks for full nonlinear transport."""

from __future__ import annotations

import torch
from typing import Dict, Optional, TYPE_CHECKING
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from torch.nn import Module

_LOG = setup_logger(__name__)


class NurtRuntimeHooks:
    """
    Runtime hook system for Concept Flow steering.

    Unlike GROM/TETNO which apply linear direction + gate, this applies
    full nonlinear flow transport: project -> Euler integrate -> reconstruct.
    The orthogonal complement is preserved exactly.
    """

    def __init__(
        self,
        model: Module,
        flow_networks: Dict[int, torch.nn.Module],
        concept_bases: Dict[int, torch.Tensor],
        layer_weights: Dict[int, float],
        base_strength: float,
        t_max: float,
        num_integration_steps: int,
    ):
        self.model = model
        self.flow_networks = flow_networks
        self.concept_bases = concept_bases
        self.layer_weights = layer_weights
        self.t_max = t_max
        self.num_integration_steps = num_integration_steps
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
        for layer_idx in self.flow_networks:
            if layer_idx < len(self._layers):
                hook = self._layers[layer_idx].register_forward_hook(
                    lambda mod, inp, out, li=layer_idx: self._flow_hook(
                        mod, inp, out, li,
                    ),
                )
                self._hooks.append(hook)

    def remove(self) -> None:
        """Remove all installed hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _flow_hook(self, module, input, output, layer_idx: int):
        """Apply full nonlinear concept flow transport."""
        hidden_states = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None

        network = self.flow_networks[layer_idx]
        Vh = self.concept_bases[layer_idx]

        h_new = self._apply_flow(hidden_states, network, Vh, layer_idx)

        if rest is not None:
            return (h_new,) + rest
        return h_new

    def _apply_flow(
        self,
        hidden_states: torch.Tensor,
        network: torch.nn.Module,
        Vh: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Project, integrate, reconstruct — preserving orthogonal complement."""
        from wisent.core.control.steering_methods.methods.nurt.flow_network import (
            euler_integrate,
        )

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
        Vh_f = Vh.float().to(h_float.device)
        network = network.to(h_float.device)
        network.eval()

        # Project to concept subspace
        z = h_float @ Vh_f.T

        # Euler integrate through velocity field
        n_layers = max(len(self.flow_networks), 1)
        layer_w = self.layer_weights.get(layer_idx, 1.0 / n_layers)
        t_eff = self.t_max * self.base_strength * layer_w

        if t_eff > 0:
            z_new = euler_integrate(
                network, z,
                t_max=t_eff,
                num_steps=self.num_integration_steps,
            )
        else:
            z_new = z

        # Reconstruct preserving orthogonal complement
        z_old = h_float @ Vh_f.T
        h_ortho = h_float - z_old @ Vh_f
        h_new = z_new.float() @ Vh_f + h_ortho

        h_new = h_new.to(original_dtype)
        return h_new.reshape(original_shape)

    @property
    def steered_layers(self) -> list[int]:
        """Return list of layer indices being steered."""
        return sorted(self.flow_networks.keys())
