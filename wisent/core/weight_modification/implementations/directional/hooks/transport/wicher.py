"""WICHER runtime hooks for subspace-projected Broyden steering."""

from __future__ import annotations

import torch
from typing import Dict, Optional, TYPE_CHECKING
from wisent.core.utils.cli.cli_logger import setup_logger, bind
from wisent.core.utils.config_tools.constants import (
    NORM_EPS,
    SEPARATOR_WIDTH_STANDARD,
    WICHER_DEFAULT_SOLVER,
)

if TYPE_CHECKING:
    from torch.nn import Module

_LOG = setup_logger(__name__)


class WicherRuntimeHooks:
    """
    Runtime hook system for WICHER Broyden-based steering.

    Installs forward hooks on transformer layers that run Broyden
    iterations in the SVD concept subspace. No lm_head weight needed.
    """

    def __init__(
        self,
        model: Module,
        concept_directions: Dict[int, torch.Tensor],
        concept_bases: Dict[int, torch.Tensor],
        component_variances: Dict[int, torch.Tensor],
        layer_variance: Dict[int, float],
        base_strength: float,
        num_steps: int,
        alpha: float,
        eta: float,
        beta: float,
        alpha_decay: float,
        solver: str = WICHER_DEFAULT_SOLVER,
    ):
        self.model = model
        self.concept_directions = concept_directions
        self.concept_bases = concept_bases
        self.component_variances = component_variances
        self.layer_variance = layer_variance
        self.num_steps = num_steps
        self.alpha = alpha
        self.eta = eta
        self.beta = beta
        self.alpha_decay = alpha_decay
        self.base_strength = base_strength
        self.solver = solver
        self._hooks = []

        self._variance_weights: Dict[int, float] = {}
        if self.layer_variance:
            total = sum(self.layer_variance.values())
            if total > 0:
                n_layers = len(self.layer_variance)
                for layer, var in self.layer_variance.items():
                    self._variance_weights[layer] = (var / total) * n_layers

        if hasattr(model, "model"):
            self._layers = model.model.layers
        elif hasattr(model, "transformer"):
            self._layers = model.transformer.h
        else:
            self._layers = model.layers

    def install(self) -> None:
        """Install forward hooks on steered layers."""
        self.remove()
        for layer_idx in self.concept_directions:
            if layer_idx < len(self._layers):
                hook = self._layers[layer_idx].register_forward_hook(
                    lambda mod, inp, out, li=layer_idx: self._wicher_hook(
                        mod, inp, out, li,
                    ),
                )
                self._hooks.append(hook)

    def remove(self) -> None:
        """Remove all installed hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _wicher_hook(self, module, input, output, layer_idx: int):
        """Apply Broyden-based steering to layer output."""
        hidden_states = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None

        h_new = self._apply_broyden(hidden_states, layer_idx)

        if rest is not None:
            return (h_new,) + rest
        return h_new

    def _apply_broyden(
        self, hidden_states: torch.Tensor, layer_idx: int,
    ) -> torch.Tensor:
        """Run solver iterations in SVD concept subspace."""
        from wisent.core.control.steering_methods.methods.wicher.solvers import (
            get_solver_fn,
        )
        solver_fn = get_solver_fn(self.solver)

        concept_dir = self.concept_directions[layer_idx].float()
        concept_dir = concept_dir / concept_dir.norm().clamp(min=NORM_EPS)
        basis = self.concept_bases[layer_idx].float()
        comp_var = self.component_variances[layer_idx].float()

        original_shape = hidden_states.shape
        original_dtype = hidden_states.dtype

        effective_strength = self.base_strength

        if hidden_states.dim() == 1:
            h = hidden_states.unsqueeze(0).float()
        elif hidden_states.dim() == 2:
            h = hidden_states.float()
        else:
            b, s, hd = hidden_states.shape
            h = hidden_states.reshape(b * s, hd).float()

        cd = concept_dir.to(h.device)
        basis_dev = basis.to(h.device)
        comp_var_dev = comp_var.to(h.device)

        h_new = solver_fn(
            h, cd * effective_strength,
            concept_basis=basis_dev,
            component_variances=comp_var_dev,
            num_steps=self.num_steps,
            alpha=self.alpha,
            eta=self.eta,
            beta=self.beta,
            alpha_decay=self.alpha_decay,
        )

        h_new = h_new.to(original_dtype)
        return h_new.reshape(original_shape)

    @property
    def steered_layers(self) -> list[int]:
        """Return list of layer indices being steered."""
        return sorted(self.concept_directions.keys())


def project_weights_wicher(
    model: Module,
    steering_obj,
    base_strength: float,
    components: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, int]:
    """Static projection: bake concept directions into model weights."""
    from wisent.core.weight_modification.methods.additive import (
        bake_steering_into_weights,
    )
    from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations
    import torch.nn.functional as F

    if components is None:
        components = ["self_attn.o_proj", "mlp.down_proj"]

    direction_vectors = {}
    for layer_idx, cd in steering_obj.concept_directions.items():
        direction_vectors[layer_idx] = F.normalize(cd.float(), p=2, dim=-1)

    if verbose:
        print(f"\n{'='*SEPARATOR_WIDTH_STANDARD}")
        print("WICHER WEIGHT MODIFICATION (ADDITIVE — concept direction)")
        print(f"{'='*SEPARATOR_WIDTH_STANDARD}")
        print(f"Layers: {len(direction_vectors)}, Strength: {base_strength}")

    steering_vectors = LayerActivations(direction_vectors)
    stats = bake_steering_into_weights(
        model=model, steering_vectors=steering_vectors,
        alpha=base_strength, method="bias", components=components, verbose=verbose,
    )
    stats["wicher_layers"] = len(steering_obj.concept_directions)
    stats["method"] = "additive"
    return stats
