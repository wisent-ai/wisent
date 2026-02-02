"""Multi-directional weight projection for PRISM."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
from wisent.core.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

_LOG = setup_logger(__name__)


def project_component_multi_direction(
    weight_matrix: Tensor, steering_vectors: Tensor, strengths: list[float] | Tensor | None = None, in_place: bool = True
) -> Tensor:
    """Project a weight matrix against MULTIPLE directions while preserving row norms."""
    log = bind(_LOG)
    num_directions = steering_vectors.shape[0]
    if strengths is None:
        strengths = [1.0] * num_directions
    elif isinstance(strengths, torch.Tensor):
        strengths = strengths.tolist()
    original_dtype = weight_matrix.dtype
    device = weight_matrix.device
    with torch.no_grad():
        W = weight_matrix.float()
        original_norms = torch.norm(W, p=2, dim=1, keepdim=True)
        for i in range(num_directions):
            v = steering_vectors[i].to(device).float()
            if v.dim() > 1:
                v = v.view(-1)
            v = F.normalize(v, p=2, dim=0)
            strength = strengths[i]
            W_norm = torch.norm(W, p=2, dim=1, keepdim=True)
            W_direction = F.normalize(W, p=2, dim=1)
            v_row = v.unsqueeze(0)
            weighted_sum = v_row @ W_direction
            projection_term = v.unsqueeze(1) @ weighted_sum
            W_direction_new = W_direction - strength * projection_term
            W_direction_new = F.normalize(W_direction_new, p=2, dim=1)
            W = W_norm * W_direction_new
        result = W.to(original_dtype)
        if in_place:
            weight_matrix.copy_(result)
            result = weight_matrix
        log.debug("Multi-direction projection", extra={"num_directions": num_directions, "strengths": strengths})
    return result


def project_weights_multi_direction(
    model: Module,
    multi_steering_vectors: dict[int, Tensor],
    harmless_vectors: dict[int, Tensor] | None = None,
    components: list[str] | None = None,
    layer_weights: dict[int, float] | None = None,
    direction_strengths: list[float] | None = None,
    global_strength: float = 1.0,
    use_biprojection: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """Norm-Preserving Multi-Directional Weight Modification for PRISM."""
    from .core import orthogonalize_direction
    log = bind(_LOG, num_layers=len(multi_steering_vectors))
    if components is None:
        components = ["self_attn.o_proj", "mlp.down_proj"]
    if hasattr(model, "model"):
        layers = model.model.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        layers = model.layers
    layers_modified, components_modified, total_params, total_directions_applied = 0, 0, 0, 0
    if verbose:
        first_layer_vecs = next(iter(multi_steering_vectors.values()))
        num_dirs = first_layer_vecs.shape[0] if first_layer_vecs.dim() > 1 else 1
        print(f"\n{'='*60}\nPRISM MULTI-DIRECTIONAL WEIGHT MODIFICATION\n{'='*60}")
        print(f"Layers: {len(multi_steering_vectors)}, Directions per layer: {num_dirs}")
        print(f"Components: {components}, Global strength: {global_strength}\n{'='*60}\n")
    for layer_idx, steering_vectors in multi_steering_vectors.items():
        if layer_idx >= len(layers):
            log.warning(f"Layer {layer_idx} out of range, skipping")
            continue
        layer = layers[layer_idx]
        layer_modified = False
        layer_weight = 1.0 if layer_weights is None else layer_weights.get(layer_idx, 1.0)
        effective_global_strength = global_strength * layer_weight
        if steering_vectors.dim() == 1:
            steering_vectors = steering_vectors.unsqueeze(0)
        num_directions = steering_vectors.shape[0]
        if direction_strengths is None:
            strengths = [effective_global_strength] * num_directions
        else:
            strengths = [s * effective_global_strength for s in direction_strengths[:num_directions]]
            while len(strengths) < num_directions:
                strengths.append(effective_global_strength)
        if use_biprojection and harmless_vectors is not None and layer_idx in harmless_vectors:
            harmless_vec = harmless_vectors[layer_idx]
            processed_vectors = [orthogonalize_direction(steering_vectors[i], harmless_vec) for i in range(num_directions)]
            steering_vectors = torch.stack(processed_vectors)
        else:
            steering_vectors = F.normalize(steering_vectors.float(), p=2, dim=1)
        for component_name in components:
            try:
                component = layer
                for attr in component_name.split("."):
                    component = getattr(component, attr)
                if not hasattr(component, "weight"):
                    continue
                weight_matrix = component.weight
                norm_before = torch.norm(weight_matrix.float(), p=2, dim=1).mean().item()
                project_component_multi_direction(weight_matrix, steering_vectors, strengths=strengths, in_place=True)
                norm_after = torch.norm(weight_matrix.float(), p=2, dim=1).mean().item()
                norm_change = abs(norm_after - norm_before) / norm_before * 100
                components_modified += 1
                total_params += weight_matrix.numel()
                total_directions_applied += num_directions
                layer_modified = True
                if verbose:
                    print(f"  Layer {layer_idx:3d} | {component_name:20s} | dirs={num_directions} | norm_change={norm_change:.4f}%")
            except AttributeError as e:
                log.warning(f"Could not access {component_name} in layer {layer_idx}: {e}")
        if layer_modified:
            layers_modified += 1
    if verbose:
        print(f"\n{'='*60}\nPRISM MODIFICATION COMPLETE\n{'='*60}")
        print(f"  Layers: {layers_modified}, Components: {components_modified}, Directions: {total_directions_applied}, Params: {total_params:,}\n{'='*60}\n")
    return {"layers_modified": layers_modified, "components_modified": components_modified,
            "total_parameters_modified": total_params, "total_directions_applied": total_directions_applied, "norm_preserved": True}
