"""Model-level weight projection functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
from wisent.core.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

_LOG = setup_logger(__name__)


def project_weights_norm_preserved(
    model: Module,
    steering_vectors: dict[int, Tensor],
    harmless_vectors: dict[int, Tensor] | None = None,
    components: list[str] | None = None,
    layer_weights: dict[int, float] | None = None,
    strength: float = 1.0,
    use_biprojection: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """Norm-Preserving Biprojected Directional Modification - the recommended method."""
    from .core import compute_projection_kernel
    from .projection import project_component_norm_preserved
    log = bind(_LOG, num_layers=len(steering_vectors))
    if components is None:
        components = ["self_attn.o_proj", "mlp.down_proj"]
    kernel = compute_projection_kernel(steering_vectors, harmless_vectors=harmless_vectors,
                                        layer_weights=layer_weights, normalize=True, use_biprojection=use_biprojection)
    if hasattr(model, "model"):
        layers = model.model.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        layers = model.layers
    layers_modified, components_modified, total_params = 0, 0, 0
    if verbose:
        print(f"\n{'='*60}\nNORM-PRESERVING BIPROJECTED DIRECTIONAL MODIFICATION\n{'='*60}")
        print(f"Layers: {len(steering_vectors)}, Components: {components}, Strength: {strength}")
        print(f"Biprojection: {use_biprojection and harmless_vectors is not None}\n{'='*60}\n")
    for layer_idx, (steering_direction, layer_weight) in kernel.items():
        if layer_idx >= len(layers):
            log.warning(f"Layer {layer_idx} out of range, skipping")
            continue
        layer = layers[layer_idx]
        layer_modified = False
        effective_strength = strength * layer_weight
        for component_name in components:
            try:
                component = layer
                for attr in component_name.split("."):
                    component = getattr(component, attr)
                if not hasattr(component, "weight"):
                    log.warning(f"Component {component_name} has no weight attribute")
                    continue
                weight_matrix = component.weight
                norm_before = torch.norm(weight_matrix.float(), p=2, dim=1).mean().item()
                project_component_norm_preserved(weight_matrix, steering_direction, effective_strength, in_place=True)
                norm_after = torch.norm(weight_matrix.float(), p=2, dim=1).mean().item()
                norm_change = abs(norm_after - norm_before) / norm_before * 100
                components_modified += 1
                total_params += weight_matrix.numel()
                layer_modified = True
                if verbose:
                    print(f"  Layer {layer_idx:3d} | {component_name:20s} | strength={effective_strength:.3f} | norm_change={norm_change:.4f}%")
            except AttributeError as e:
                log.warning(f"Could not access {component_name} in layer {layer_idx}: {e}")
        if layer_modified:
            layers_modified += 1
    if verbose:
        print(f"\n{'='*60}\nDIRECTIONAL MODIFICATION COMPLETE\n{'='*60}")
        print(f"  Layers modified: {layers_modified}, Components: {components_modified}, Params: {total_params:,}\n{'='*60}\n")
    return {"layers_modified": layers_modified, "components_modified": components_modified,
            "total_parameters_modified": total_params, "norm_preserved": True}


def project_weights(
    model: Module,
    steering_vectors: dict[int, Tensor],
    harmless_vectors: dict[int, Tensor] | None = None,
    components: list[str] | None = None,
    layer_weights: dict[int, float] | None = None,
    strength: float = 1.0,
    normalize_vectors: bool = True,
    norm_preserve: bool = True,
    use_biprojection: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """Directionally modify model weights - unified interface."""
    from .projection import project_component
    if norm_preserve:
        return project_weights_norm_preserved(model=model, steering_vectors=steering_vectors, harmless_vectors=harmless_vectors,
                                               components=components, layer_weights=layer_weights, strength=strength,
                                               use_biprojection=use_biprojection, verbose=verbose)
    log = bind(_LOG, num_layers=len(steering_vectors))
    if verbose:
        print("\nWARNING: Using legacy projection (norms NOT preserved)\nConsider using norm_preserve=True for better model quality\n")
    if components is None:
        components = ["self_attn.o_proj", "mlp.down_proj"]
    if hasattr(model, "model"):
        layers = model.model.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        layers = model.layers
    layers_modified, components_modified, total_params = 0, 0, 0
    for layer_idx, steering_vector in steering_vectors.items():
        if layer_idx >= len(layers):
            continue
        layer = layers[layer_idx]
        layer_modified = False
        v = F.normalize(steering_vector.float(), p=2, dim=0) if normalize_vectors else steering_vector.float()
        projector = torch.outer(v, v)
        layer_weight = 1.0 if layer_weights is None else layer_weights.get(layer_idx, 1.0)
        effective_strength = strength * layer_weight
        for component_name in components:
            try:
                component = layer
                for attr in component_name.split("."):
                    component = getattr(component, attr)
                if hasattr(component, "weight"):
                    project_component(component.weight, projector, effective_strength, in_place=True)
                    components_modified += 1
                    total_params += component.weight.numel()
                    layer_modified = True
            except AttributeError:
                pass
        if layer_modified:
            layers_modified += 1
    return {"layers_modified": layers_modified, "components_modified": components_modified,
            "total_parameters_modified": total_params, "norm_preserved": False}
