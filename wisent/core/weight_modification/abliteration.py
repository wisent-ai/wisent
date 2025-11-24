"""
Orthogonal abliteration: Permanently remove capabilities via weight modification.

Implements Heretic-style abliteration adapted for Wisent's contrastive pairs.

Key difference from Heretic:
- Heretic removes refusal (harmful → harmless direction)
- Wisent removes incorrect behavior (positive → negative direction, reversed)

Mathematical operation:
    W' = W - λ·(vvᵀ)·W

Where:
- W: Original weight matrix [out_dim, in_dim]
- v: Steering vector (normalized) [out_dim]
- λ: Abliteration strength (0 = no change, 1 = full abliteration)
- vvᵀ: Rank-1 projection matrix onto v [out_dim, out_dim]

Effect: W' cannot produce outputs with components parallel to v in the output space.
This works for any weight matrix that outputs to the residual stream (hidden_size).
The projection is applied on the left (P @ W) to project the output rows.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

__all__ = [
    "abliterate_weights",
    "abliterate_component",
    "compute_abliteration_kernel",
]

_LOG = setup_logger(__name__)


def compute_abliteration_kernel(
    steering_vectors: dict[int, Tensor],
    layer_weights: dict[int, float] | None = None,
    normalize: bool = True,
) -> dict[int, tuple[Tensor, float]]:
    """
    Compute abliteration parameters for each layer.

    Args:
        steering_vectors: Per-layer steering vectors {layer_idx: [H]}
        layer_weights: Optional per-layer abliteration strength {layer_idx: weight}
                      If None, uses uniform weight of 1.0
        normalize: Whether to normalize steering vectors before abliteration

    Returns:
        Dictionary mapping layer_idx to (projector, weight) where:
        - projector: vvᵀ projection matrix [H, H]
        - weight: abliteration strength for this layer

    Example:
        >>> steering = {0: torch.randn(4096), 1: torch.randn(4096)}
        >>> kernel = compute_abliteration_kernel(steering, layer_weights={0: 1.0, 1: 0.5})
    """
    log = bind(_LOG, num_layers=len(steering_vectors))

    kernel = {}

    for layer_idx, steering_vector in steering_vectors.items():
        # Normalize if requested
        if normalize:
            v = F.normalize(steering_vector, p=2, dim=0)
        else:
            v = steering_vector

        # Compute rank-1 projector: vvᵀ
        projector = torch.outer(v, v)  # [H, H]

        # Get layer weight
        weight = 1.0 if layer_weights is None else layer_weights.get(layer_idx, 1.0)

        kernel[layer_idx] = (projector, weight)

        log.debug(
            "Computed abliteration kernel",
            extra={
                "layer": layer_idx,
                "projector_shape": projector.shape,
                "weight": weight,
                "vector_norm": v.norm().item(),
            },
        )

    return kernel


def abliterate_component(
    weight_matrix: Tensor,
    projector: Tensor,
    strength: float,
    in_place: bool = True,
) -> Tensor:
    """
    Abliterate a single weight matrix.

    Args:
        weight_matrix: Weight matrix to modify [out_dim, in_dim]
        projector: Projection matrix vvᵀ [out_dim, out_dim]
        strength: Abliteration strength λ (0 = no change, 1 = full abliteration)
        in_place: Whether to modify in-place (True) or return copy (False)

    Returns:
        Modified weight matrix

    Mathematical operation:
        W' = W - λ·(vvᵀ)·W = W - λ·P·W

        This projects the OUTPUT space, removing the component of the output
        that aligns with the steering vector v. The projection is applied
        on the left of W since we want to project the rows (output vectors).

    Example:
        >>> W = torch.randn(4096, 4096)
        >>> v = torch.randn(4096)
        >>> P = torch.outer(v, v)
        >>> W_new = abliterate_component(W, P, strength=1.0)
    """
    log = bind(_LOG)

    # Ensure projector is on same device as weight
    device_projector = projector.to(weight_matrix.device)

    # Compute projection: P @ W
    # W is [out_dim, in_dim], P is [out_dim, out_dim]
    # P @ W = [out_dim, out_dim] @ [out_dim, in_dim] = [out_dim, in_dim]
    projected = device_projector @ weight_matrix  # [out_dim, in_dim]

    # Apply abliteration: W' = W - λ·(P·W)
    if in_place:
        with torch.no_grad():
            weight_matrix.sub_(strength * projected)
        result = weight_matrix
    else:
        result = weight_matrix - strength * projected

    log.debug(
        "Abliterated component",
        extra={
            "weight_shape": weight_matrix.shape,
            "projector_shape": device_projector.shape,
            "strength": strength,
            "in_place": in_place,
            "modification_norm": (strength * projected).norm().item(),
        },
    )

    return result


def abliterate_weights(
    model: Module,
    steering_vectors: dict[int, Tensor],
    components: list[str] | None = None,
    layer_weights: dict[int, float] | None = None,
    strength: float = 1.0,
    normalize_vectors: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Abliterate model weights based on steering vectors.

    Permanently modifies model weights to remove capability of expressing
    behavior in the steering direction.

    Args:
        model: Model to modify (in-place)
        steering_vectors: Per-layer steering vectors {layer_idx: [H]}
        components: Component names to abliterate (e.g., ["attn.o_proj", "mlp.down_proj"])
                   If None, uses ["self_attn.o_proj", "mlp.down_proj"]
        layer_weights: Optional per-layer strength {layer_idx: weight}
        strength: Global abliteration strength multiplier
        normalize_vectors: Whether to normalize steering vectors
        verbose: Whether to print progress

    Returns:
        Dictionary with modification statistics:
        - "layers_modified": Number of layers modified
        - "components_modified": Number of components modified
        - "total_parameters_modified": Total parameters changed

    Example:
        >>> from wisent.core.weight_modification import abliterate_weights
        >>> # Compute steering vectors from contrastive pairs
        >>> steering_vectors = {...}
        >>> # Permanently abliterate model
        >>> stats = abliterate_weights(model, steering_vectors, strength=1.0)
        >>> # Model is now modified - no runtime overhead
        >>> model.save_pretrained("path/to/abliterated-model")
    """
    log = bind(_LOG, num_layers=len(steering_vectors))

    if components is None:
        # Default components (attention output + MLP down projection)
        components = ["self_attn.o_proj", "mlp.down_proj"]

    # Compute abliteration kernel
    kernel = compute_abliteration_kernel(
        steering_vectors,
        layer_weights=layer_weights,
        normalize=normalize_vectors,
    )

    # Get model layers
    if hasattr(model, "model"):
        layers = model.model.layers  # LlamaForCausalLM, etc.
    elif hasattr(model, "transformer"):
        layers = model.transformer.h  # GPT-style
    else:
        layers = model.layers  # Direct access

    layers_modified = 0
    components_modified = 0
    total_params = 0

    if verbose:
        print(f"\nAbliterating {len(steering_vectors)} layers...")
        print(f"Components: {components}")
        print(f"Global strength: {strength}")

    for layer_idx, (projector, layer_weight) in kernel.items():
        if layer_idx >= len(layers):
            log.warning(f"Layer {layer_idx} out of range, skipping")
            continue

        layer = layers[layer_idx]
        layer_modified = False

        for component_name in components:
            try:
                # Navigate to component (e.g., "self_attn.o_proj")
                component = layer
                for attr in component_name.split("."):
                    component = getattr(component, attr)

                # Get weight matrix
                if hasattr(component, "weight"):
                    weight_matrix = component.weight
                else:
                    log.warning(f"Component {component_name} has no weight attribute")
                    continue

                # Abliterate
                effective_strength = strength * layer_weight
                abliterate_component(
                    weight_matrix,
                    projector,
                    effective_strength,
                    in_place=True,
                )

                components_modified += 1
                total_params += weight_matrix.numel()
                layer_modified = True

                if verbose:
                    print(
                        f"  Layer {layer_idx:3d} | {component_name:20s} | "
                        f"strength={effective_strength:.3f} | "
                        f"params={weight_matrix.numel():,}"
                    )

            except AttributeError as e:
                log.warning(
                    f"Could not access {component_name} in layer {layer_idx}: {e}"
                )

        if layer_modified:
            layers_modified += 1

    if verbose:
        print(f"\nAbliteration complete:")
        print(f"  Layers modified: {layers_modified}")
        print(f"  Components modified: {components_modified}")
        print(f"  Total parameters: {total_params:,}")

    return {
        "layers_modified": layers_modified,
        "components_modified": components_modified,
        "total_parameters_modified": total_params,
    }


def abliterate_with_kernel(
    model: Module,
    steering_vectors: dict[int, Tensor],
    max_weight: float = 1.0,
    max_weight_position: float | None = None,
    min_weight: float = 0.0,
    min_weight_distance: float | None = None,
    components: list[str] | None = None,
    normalize_vectors: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Abliterate with Heretic-style kernel-based layer weights.

    Creates a smooth weight distribution across layers with maximum
    at max_weight_position, tapering to min_weight at edges.

    Args:
        model: Model to modify
        steering_vectors: Per-layer steering vectors
        max_weight: Peak abliteration strength
        max_weight_position: Layer index of peak (None = middle layer)
        min_weight: Minimum abliteration strength
        min_weight_distance: Distance over which weight decays (None = 60% of layers)
        components: Components to abliterate
        normalize_vectors: Whether to normalize steering vectors
        verbose: Whether to print progress

    Returns:
        Modification statistics

    Example:
        >>> # Abliterate with Gaussian-like kernel centered at layer 15
        >>> stats = abliterate_with_kernel(
        ...     model,
        ...     steering_vectors,
        ...     max_weight=1.5,
        ...     max_weight_position=15.0,
        ...     min_weight=0.3,
        ...     min_weight_distance=10.0,
        ... )
    """
    num_layers = len(steering_vectors)

    # Set defaults
    if max_weight_position is None:
        max_weight_position = (num_layers - 1) / 2

    if min_weight_distance is None:
        min_weight_distance = 0.6 * (num_layers - 1)

    # Compute per-layer weights using kernel
    layer_weights = {}
    for layer_idx in range(num_layers):
        distance = abs(layer_idx - max_weight_position)

        if distance > min_weight_distance:
            weight = 0.0  # Too far from center
        else:
            # Linear interpolation from max_weight to min_weight
            weight = max_weight + (distance / min_weight_distance) * (
                min_weight - max_weight
            )

        layer_weights[layer_idx] = weight

    if verbose:
        print(f"\nKernel configuration:")
        print(f"  Peak: {max_weight:.2f} at layer {max_weight_position:.1f}")
        print(f"  Min: {min_weight:.2f} within distance {min_weight_distance:.1f}")
        print(f"  Active layers: {sum(1 for w in layer_weights.values() if w > 0)}/{num_layers}")

    return abliterate_weights(
        model,
        steering_vectors,
        components=components,
        layer_weights=layer_weights,
        strength=1.0,  # Already incorporated in layer_weights
        normalize_vectors=normalize_vectors,
        verbose=verbose,
    )
