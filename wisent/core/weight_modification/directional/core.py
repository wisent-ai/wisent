"""Core directional projection utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
from wisent.core.cli.cli_logger import setup_logger, bind
from wisent.core.errors import InvalidValueError

if TYPE_CHECKING:
    from torch import Tensor

_LOG = setup_logger(__name__)


def orthogonalize_direction(steering_vector: Tensor, harmless_vector: Tensor | None = None) -> Tensor:
    """Orthogonalize steering direction against harmless direction (biprojection)."""
    steering_normalized = F.normalize(steering_vector.float(), p=2, dim=0)
    if harmless_vector is None:
        return steering_normalized
    harmless_normalized = F.normalize(harmless_vector.float(), p=2, dim=0)
    projection_scalar = torch.dot(steering_normalized, harmless_normalized)
    orthogonalized = steering_normalized - projection_scalar * harmless_normalized
    return F.normalize(orthogonalized, p=2, dim=0)


def compute_projection_kernel(
    steering_vectors: dict[int, Tensor],
    harmless_vectors: dict[int, Tensor] | None = None,
    layer_weights: dict[int, float] | None = None,
    normalize: bool = True,
    use_biprojection: bool = True,
) -> dict[int, tuple[Tensor, float]]:
    """Compute directional projection parameters for each layer."""
    log = bind(_LOG, num_layers=len(steering_vectors))
    kernel = {}
    for layer_idx, steering_vector in steering_vectors.items():
        harmless_vector = None
        if use_biprojection and harmless_vectors is not None:
            harmless_vector = harmless_vectors.get(layer_idx)
        if use_biprojection and harmless_vector is not None:
            v = orthogonalize_direction(steering_vector, harmless_vector)
        elif normalize:
            v = F.normalize(steering_vector.float(), p=2, dim=0)
        else:
            v = steering_vector.float()
        weight = 1.0 if layer_weights is None else layer_weights.get(layer_idx, 1.0)
        kernel[layer_idx] = (v, weight)
        log.debug("Computed projection kernel", extra={"layer": layer_idx, "weight": weight, "vector_norm": v.norm().item(), "biprojected": harmless_vector is not None})
    return kernel


def project_with_kernel(
    model,
    kernel: dict[int, tuple[Tensor, float]],
    components: list[str] | None = None,
    strength: float = 1.0,
    preserve_norms: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """Apply pre-computed projection kernel to model weights."""
    from .projection import project_component_norm_preserved, project_component
    log = bind(_LOG, num_layers=len(kernel), preserve_norms=preserve_norms)
    if components is None:
        components = ["mlp.down_proj", "mlp.gate_proj", "mlp.up_proj", "self_attn.o_proj"]
    stats = {"total_projections": 0, "layers_modified": 0}
    for layer_idx, (steering_direction, layer_weight) in kernel.items():
        combined_strength = strength * layer_weight
        if abs(combined_strength) < 1e-8:
            continue
        layer_modified = False
        for component_name in components:
            full_param_name = f"model.layers.{layer_idx}.{component_name}.weight"
            try:
                param = dict(model.named_parameters()).get(full_param_name)
                if param is None:
                    continue
                if preserve_norms:
                    project_component_norm_preserved(param, steering_direction, strength=combined_strength, in_place=True)
                else:
                    projector = steering_direction.unsqueeze(1) @ steering_direction.unsqueeze(0)
                    project_component(param, projector, strength=combined_strength, in_place=True)
                stats["total_projections"] += 1
                layer_modified = True
                if verbose:
                    log.debug(f"Projected layer {layer_idx} {component_name}")
            except Exception as e:
                log.warning(f"Failed to project {full_param_name}: {e}")
        if layer_modified:
            stats["layers_modified"] += 1
    log.info("Projection complete", extra=stats)
    return stats


def verify_weight_modification_preservation(original_norms: dict, modified_norms: dict, tolerance: float = 0.01) -> tuple[bool, dict]:
    """Verify weight modification preserved norms within tolerance."""
    results = {"preserved": True, "max_deviation": 0.0, "violations": []}
    for key in original_norms:
        if key not in modified_norms:
            continue
        deviation = abs(modified_norms[key] - original_norms[key]) / (original_norms[key] + 1e-8)
        results["max_deviation"] = max(results["max_deviation"], deviation)
        if deviation > tolerance:
            results["preserved"] = False
            results["violations"].append({"param": key, "original": original_norms[key], "modified": modified_norms[key], "deviation": deviation})
    return results["preserved"], results
