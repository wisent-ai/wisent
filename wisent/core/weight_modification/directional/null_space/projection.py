"""Component-level and model-level null-space constrained weight projection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, TYPE_CHECKING

from wisent.core.cli.cli_logger import setup_logger, bind
from .projector import PreservedKeyMatrix

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

__all__ = [
    "project_component_null_space",
    "project_weights_null_space",
]

_LOG = setup_logger(__name__)


def project_component_null_space(
    weight_matrix: "Tensor",
    steering_direction: "Tensor",
    P_null: "Tensor",
    strength: float = 1.0,
    preserve_norms: bool = True,
    in_place: bool = True,
) -> "Tensor":
    """Project a weight component with null-space constraint.

    Steps:
        1. Compute raw delta: delta_W = strength * v * (v @ W)
        2. Constrain: delta_W = delta_W @ P_null
        3. Apply: W = W - delta_W
        4. If preserve_norms, rescale rows to original norms
    """
    original_dtype = weight_matrix.dtype
    device = weight_matrix.device

    with torch.no_grad():
        W = weight_matrix.float()
        v = steering_direction.to(device).float()
        if v.dim() > 1:
            v = v.view(-1)
        v = F.normalize(v, p=2, dim=0)

        if preserve_norms:
            W_norm = torch.norm(W, p=2, dim=1, keepdim=True)

        vW = v @ W  # [in_dim]
        delta_W = strength * v.unsqueeze(1) * vW.unsqueeze(0)  # [out_dim, in_dim]
        delta_W = delta_W @ P_null.to(device)
        W = W - delta_W

        if preserve_norms:
            W_new_norm = torch.norm(W, p=2, dim=1, keepdim=True)
            W = W * (W_norm / (W_new_norm + 1e-8))

        result = W.to(original_dtype)
        if in_place:
            weight_matrix.copy_(result)
            result = weight_matrix

    return result


def project_weights_null_space(
    model: "Module",
    steering_vectors: Dict[int, "Tensor"],
    preserved_keys: PreservedKeyMatrix,
    harmless_vectors: Optional[Dict[int, "Tensor"]] = None,
    components: Optional[List[str]] = None,
    layer_weights: Optional[Dict[int, float]] = None,
    strength: float = 1.0,
    preserve_norms: bool = True,
    use_biprojection: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """Model-level null-space constrained weight modification.

    Iterates layers, applying null-space constrained projection where
    preserved keys exist, falling back to standard norm-preserved projection.
    """
    from wisent.core.weight_modification.directional.core import orthogonalize_direction
    from wisent.core.weight_modification.directional.projection import project_component_norm_preserved

    log = bind(_LOG, num_layers=len(steering_vectors))

    if components is None:
        components = ["self_attn.o_proj", "mlp.down_proj"]

    if hasattr(model, "model"):
        layers = model.model.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        layers = model.layers

    layers_modified, components_modified, total_params, null_space_applied = 0, 0, 0, 0

    if verbose:
        print(f"\n{'='*60}")
        print("NULL-SPACE CONSTRAINED WEIGHT MODIFICATION (AlphaEdit)")
        print(f"{'='*60}")
        print(f"Layers: {len(steering_vectors)}, Components: {components}")
        print(f"Strength: {strength}, Biprojection: {use_biprojection}")
        print(f"Preserved keys: {preserved_keys.summary()}")
        print(f"{'='*60}\n")

    for layer_idx, steering_vector in steering_vectors.items():
        if layer_idx >= len(layers):
            log.warning(f"Layer {layer_idx} out of range, skipping")
            continue

        layer = layers[layer_idx]
        layer_weight = 1.0 if layer_weights is None else layer_weights.get(layer_idx, 1.0)
        effective_strength = strength * layer_weight
        if abs(effective_strength) < 1e-8:
            continue

        if use_biprojection and harmless_vectors is not None and layer_idx in harmless_vectors:
            direction = orthogonalize_direction(steering_vector, harmless_vectors[layer_idx])
        else:
            direction = F.normalize(steering_vector.float(), p=2, dim=0)

        P_null = preserved_keys.get_projector(layer_idx)
        layer_modified = False

        for component_name in components:
            try:
                component = layer
                for attr in component_name.split("."):
                    component = getattr(component, attr)
                if not hasattr(component, "weight"):
                    continue

                weight_matrix = component.weight
                if P_null is not None:
                    project_component_null_space(
                        weight_matrix, direction, P_null,
                        strength=effective_strength,
                        preserve_norms=preserve_norms, in_place=True,
                    )
                    null_space_applied += 1
                else:
                    project_component_norm_preserved(
                        weight_matrix, direction,
                        strength=effective_strength, in_place=True,
                    )

                components_modified += 1
                total_params += weight_matrix.numel()
                layer_modified = True

                if verbose:
                    mode = "null-space" if P_null is not None else "standard"
                    print(f"  Layer {layer_idx:3d} | {component_name:20s} | "
                          f"strength={effective_strength:.3f} | mode={mode}")
            except AttributeError as e:
                log.warning(f"Could not access {component_name} in layer {layer_idx}: {e}")

        if layer_modified:
            layers_modified += 1

    if verbose:
        print(f"\n{'='*60}")
        print("NULL-SPACE MODIFICATION COMPLETE")
        print(f"{'='*60}")
        print(f"  Layers: {layers_modified}, Components: {components_modified}")
        print(f"  Null-space constrained: {null_space_applied}, Params: {total_params:,}")
        print(f"{'='*60}\n")

    return {
        "layers_modified": layers_modified,
        "components_modified": components_modified,
        "total_parameters_modified": total_params,
        "null_space_constrained": null_space_applied,
        "norm_preserved": preserve_norms,
    }
