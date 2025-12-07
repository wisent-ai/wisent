"""
Norm-Preserving Biprojected Directional Weight Modification.

Permanently modifies model weights by projecting out or projecting onto
specific directions in the representation space. Can be used to:
- Remove behaviors (e.g., refusal removal)
- Add behaviors (e.g., personality traits, speaking styles)
- Any directional steering that should be baked into weights

Implements norm-preserving directional projection.

Key improvements over standard projection:

1. **Biprojection (Targeting)**: Refines the steering direction to be
   mathematically orthogonal to "harmless" directions. This ensures we don't
   accidentally remove healthy, harmless concepts.

2. **Decomposition**: Instead of raw subtraction, we decompose weight matrices
   into Magnitude and Direction components.

3. **Norm-Preservation**: We modify the steering component solely from the
   directional aspect of weights, then recombine with original magnitudes.

Standard projection:
    W' = W - λ·(vvᵀ)·W  (CHANGES NORMS - damages model)

Norm-preserving projection:
    1. Decompose: W = ||W|| * W_direction
    2. Project direction: W_direction' = W_direction - λ·v·(v·W_direction)
    3. Renormalize: W_direction' = normalize(W_direction')
    4. Recombine: W' = ||W|| * W_direction'  (PRESERVES NORMS)

The result maintains the "importance" structure of the neural network, avoiding
the "Safety Tax" that degrades reasoning in standard modified models.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
from wisent.core.cli_logger import setup_logger, bind
from wisent.core.errors import InvalidValueError

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

__all__ = [
    "project_weights",
    "project_weights_norm_preserved",
    "project_component",
    "project_component_norm_preserved",
    "compute_projection_kernel",
    "project_with_kernel",
    "orthogonalize_direction",
]

_LOG = setup_logger(__name__)


def orthogonalize_direction(
    steering_vector: Tensor,
    harmless_vector: Tensor | None = None,
) -> Tensor:
    """
    Orthogonalize steering direction against harmless direction (biprojection).

    This ensures that when we ablate the steering behavior, we don't accidentally
    remove healthy, harmless concepts that may share some similarity.

    Uses Gram-Schmidt orthonormalization to compute the component of the steering
    direction that is orthogonal to the harmless direction.

    Args:
        steering_vector: The direction to ablate (e.g., refusal direction)
        harmless_vector: Optional harmless direction to preserve
                        If None, returns normalized steering_vector unchanged

    Returns:
        Orthogonalized and normalized steering direction

    Mathematical operation:
        v_orth = v_steer - (v_steer · v_harmless) * v_harmless
        v_orth = normalize(v_orth)
    """
    # Normalize steering vector
    steering_normalized = F.normalize(steering_vector.float(), p=2, dim=0)

    if harmless_vector is None:
        return steering_normalized

    # Normalize harmless direction
    harmless_normalized = F.normalize(harmless_vector.float(), p=2, dim=0)

    # Project steering onto harmless and subtract (Gram-Schmidt)
    projection_scalar = torch.dot(steering_normalized, harmless_normalized)
    orthogonalized = steering_normalized - projection_scalar * harmless_normalized

    # Renormalize the orthogonalized direction
    orthogonalized = F.normalize(orthogonalized, p=2, dim=0)

    return orthogonalized


def project_component_norm_preserved(
    weight_matrix: Tensor,
    steering_vector: Tensor,
    strength: float = 1.0,
    in_place: bool = True,
) -> Tensor:
    """
    Project a weight matrix while PRESERVING ROW NORMS.

    This is the key innovation of the norm-preserving technique.
    Instead of directly subtracting from weights (which changes magnitudes),
    we decompose into direction and magnitude, project only the direction,
    then recombine with original magnitudes.

    The steering vector represents a direction in the OUTPUT space of the layer.
    We want to reduce how much the layer's output aligns with this direction.

    Args:
        weight_matrix: Weight matrix to modify [out_dim, in_dim]
        steering_vector: Steering direction in output space [out_dim]
        strength: Projection strength (scale_factor, typically 1.0)
        in_place: Whether to modify in-place

    Returns:
        Modified weight matrix with preserved row norms

    Mathematical operation (for output-space steering):
        Standard projection: W' = W - λ(vvᵀ)W

        Norm-preserving version:
        1. W_norm = ||W[i,:]||_2 for each row i (magnitudes)
        2. W_direction = normalize(W, dim=1) (unit direction per row)
        3. projection[i] = v[i] (how much output i contributes to steering)
        4. W_direction'[i,:] = W_direction[i,:] - strength * projection[i] * v_expanded
        5. W_direction' = normalize(W_direction', dim=1) (re-normalize)
        6. W' = W_norm * W_direction' (restore original magnitudes)
    """
    log = bind(_LOG)

    original_dtype = weight_matrix.dtype
    device = weight_matrix.device

    with torch.no_grad():
        # Work in float32 for numerical stability
        W = weight_matrix.float()
        v = steering_vector.to(device).float()

        # Flatten if needed
        if v.dim() > 1:
            v = v.view(-1)

        # Ensure steering vector is normalized
        v = F.normalize(v, p=2, dim=0)

        out_dim, in_dim = W.shape

        # Verify dimensions match
        if v.shape[0] != out_dim:
            raise InvalidValueError(
                param_name="steering_vector dimension",
                actual=v.shape[0],
                expected=f"weight matrix output dimension {out_dim} (weight shape: {W.shape}, steering shape: {v.shape})"
            )

        # Step 1: Decompose weight matrix into magnitude and direction
        # W is [out_dim, in_dim], each row is one output neuron's weights
        W_norm = torch.norm(W, p=2, dim=1, keepdim=True)  # [out_dim, 1]
        W_direction = F.normalize(W, p=2, dim=1)  # [out_dim, in_dim], unit vectors

        # Step 2: Compute the projection term
        # For output-space projection: W' = W - λ(vvᵀ)W
        # (vvᵀ)W = v @ (vᵀ @ W) = v @ (v.unsqueeze(0) @ W)
        # This gives us the component of W that projects onto v in output space

        # vᵀ @ W gives [1, in_dim] - the weighted sum of rows by v
        v_row = v.unsqueeze(0)  # [1, out_dim]
        weighted_sum = v_row @ W_direction  # [1, in_dim]

        # v @ (vᵀ @ W) gives [out_dim, in_dim] - outer product expansion
        # This is the rank-1 component to remove
        projection_term = v.unsqueeze(1) @ weighted_sum  # [out_dim, in_dim]

        # Step 3: Project the directional component
        W_direction_new = W_direction - strength * projection_term

        # Step 4: Re-normalize the adjusted directions (crucial for norm preservation)
        W_direction_new = F.normalize(W_direction_new, p=2, dim=1)

        # Step 5: Recombine with original magnitudes
        W_modified = W_norm * W_direction_new

        # Convert back to original dtype
        result = W_modified.to(original_dtype)

        if in_place:
            weight_matrix.copy_(result)
            result = weight_matrix

        # Compute projection magnitude for logging
        projection_magnitude = (v.unsqueeze(1) * W_direction).sum(dim=1)

        log.debug(
            "Norm-preserved directional projection",
            extra={
                "weight_shape": weight_matrix.shape,
                "strength": strength,
                "in_place": in_place,
                "mean_projection": projection_magnitude.abs().mean().item(),
                "max_projection": projection_magnitude.abs().max().item(),
            },
        )

    return result


def project_component(
    weight_matrix: Tensor,
    projector: Tensor,
    strength: float,
    in_place: bool = True,
) -> Tensor:
    """
    Standard directional projection (DOES NOT preserve norms - legacy method).

    WARNING: This method alters weight magnitudes, which can degrade model quality.
    Use project_component_norm_preserved() instead for better results.

    Args:
        weight_matrix: Weight matrix to modify [out_dim, in_dim]
        projector: Projection matrix vvᵀ [out_dim, out_dim]
        strength: Projection strength λ
        in_place: Whether to modify in-place

    Returns:
        Modified weight matrix (with altered norms)
    """
    log = bind(_LOG)

    device_projector = projector.to(weight_matrix.device)
    projected = device_projector @ weight_matrix

    if in_place:
        with torch.no_grad():
            weight_matrix.sub_(strength * projected)
        result = weight_matrix
    else:
        result = weight_matrix - strength * projected

    log.debug(
        "Standard directional projection (norms NOT preserved)",
        extra={
            "weight_shape": weight_matrix.shape,
            "strength": strength,
            "in_place": in_place,
        },
    )

    return result


def compute_projection_kernel(
    steering_vectors: dict[int, Tensor],
    harmless_vectors: dict[int, Tensor] | None = None,
    layer_weights: dict[int, float] | None = None,
    normalize: bool = True,
    use_biprojection: bool = True,
) -> dict[int, tuple[Tensor, float]]:
    """
    Compute directional projection parameters for each layer.

    Args:
        steering_vectors: Per-layer steering vectors {layer_idx: [H]}
        harmless_vectors: Optional per-layer harmless directions for biprojection
        layer_weights: Optional per-layer strength {layer_idx: weight}
        normalize: Whether to normalize steering vectors
        use_biprojection: Whether to orthogonalize against harmless directions

    Returns:
        Dictionary mapping layer_idx to (steering_direction, weight)
    """
    log = bind(_LOG, num_layers=len(steering_vectors))

    kernel = {}

    for layer_idx, steering_vector in steering_vectors.items():
        # Get harmless vector for this layer if available
        harmless_vector = None
        if use_biprojection and harmless_vectors is not None:
            harmless_vector = harmless_vectors.get(layer_idx)

        # Orthogonalize and normalize
        if use_biprojection and harmless_vector is not None:
            v = orthogonalize_direction(steering_vector, harmless_vector)
        elif normalize:
            v = F.normalize(steering_vector.float(), p=2, dim=0)
        else:
            v = steering_vector.float()

        # Get layer weight
        weight = 1.0 if layer_weights is None else layer_weights.get(layer_idx, 1.0)

        kernel[layer_idx] = (v, weight)

        log.debug(
            "Computed projection kernel",
            extra={
                "layer": layer_idx,
                "weight": weight,
                "vector_norm": v.norm().item(),
                "biprojected": harmless_vector is not None,
            },
        )

    return kernel


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
    """
    Norm-Preserving Biprojected Directional Modification - the recommended method.

    Permanently modifies model weights to project out or onto a steering direction,
    while PRESERVING weight magnitudes to maintain model quality and reasoning
    capabilities. Can be used to remove or add behaviors.

    This implements the norm-preserving directional projection technique.

    Args:
        model: Model to modify (in-place)
        steering_vectors: Per-layer steering vectors {layer_idx: [H]}
        harmless_vectors: Optional per-layer harmless directions for biprojection
                         If provided, steering directions are orthogonalized
                         against these to avoid removing harmless concepts
        components: Component names to modify
                   Default: ["self_attn.o_proj", "mlp.down_proj"]
        layer_weights: Optional per-layer strength {layer_idx: weight}
        strength: Global projection strength (scale_factor)
        use_biprojection: Whether to use biprojection (orthogonalize against harmless)
        verbose: Whether to print progress

    Returns:
        Dictionary with modification statistics

    Example:
        >>> from wisent.core.weight_modification import project_weights_norm_preserved
        >>> # Project with norm preservation
        >>> stats = project_weights_norm_preserved(
        ...     model,
        ...     steering_vectors,
        ...     harmless_vectors=harmless_vectors,  # Optional
        ...     strength=1.0,
        ... )
        >>> model.save_pretrained("path/to/modified-model")
    """
    log = bind(_LOG, num_layers=len(steering_vectors))

    if components is None:
        components = ["self_attn.o_proj", "mlp.down_proj"]

    # Compute projection kernel with optional biprojection
    kernel = compute_projection_kernel(
        steering_vectors,
        harmless_vectors=harmless_vectors,
        layer_weights=layer_weights,
        normalize=True,
        use_biprojection=use_biprojection,
    )

    # Get model layers
    if hasattr(model, "model"):
        layers = model.model.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        layers = model.layers

    layers_modified = 0
    components_modified = 0
    total_params = 0

    if verbose:
        print(f"\n{'='*60}")
        print("NORM-PRESERVING BIPROJECTED DIRECTIONAL MODIFICATION")
        print(f"{'='*60}")
        print(f"Layers: {len(steering_vectors)}")
        print(f"Components: {components}")
        print(f"Strength: {strength}")
        print(f"Biprojection: {use_biprojection and harmless_vectors is not None}")
        print(f"{'='*60}\n")

    for layer_idx, (steering_direction, layer_weight) in kernel.items():
        if layer_idx >= len(layers):
            log.warning(f"Layer {layer_idx} out of range, skipping")
            continue

        layer = layers[layer_idx]
        layer_modified = False
        effective_strength = strength * layer_weight

        for component_name in components:
            try:
                # Navigate to component
                component = layer
                for attr in component_name.split("."):
                    component = getattr(component, attr)

                if not hasattr(component, "weight"):
                    log.warning(f"Component {component_name} has no weight attribute (may be quantized - use additive method instead)")
                    continue

                weight_matrix = component.weight

                # Compute norm before for verification
                norm_before = torch.norm(weight_matrix.float(), p=2, dim=1).mean().item()

                # Apply norm-preserving directional projection
                project_component_norm_preserved(
                    weight_matrix,
                    steering_direction,
                    effective_strength,
                    in_place=True,
                )

                # Verify norm preservation
                norm_after = torch.norm(weight_matrix.float(), p=2, dim=1).mean().item()
                norm_change = abs(norm_after - norm_before) / norm_before * 100

                components_modified += 1
                total_params += weight_matrix.numel()
                layer_modified = True

                if verbose:
                    print(
                        f"  Layer {layer_idx:3d} | {component_name:20s} | "
                        f"strength={effective_strength:.3f} | "
                        f"norm_change={norm_change:.4f}%"
                    )

            except AttributeError as e:
                log.warning(f"Could not access {component_name} in layer {layer_idx}: {e}")

        if layer_modified:
            layers_modified += 1

    if verbose:
        print(f"\n{'='*60}")
        print("DIRECTIONAL MODIFICATION COMPLETE")
        print(f"{'='*60}")
        print(f"  Layers modified: {layers_modified}")
        print(f"  Components modified: {components_modified}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Norms preserved: YES")
        print(f"{'='*60}\n")

    return {
        "layers_modified": layers_modified,
        "components_modified": components_modified,
        "total_parameters_modified": total_params,
        "norm_preserved": True,
    }


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
    """
    Directionally modify model weights - unified interface.

    By default, uses norm-preserving biprojected modification (recommended).
    Set norm_preserve=False for legacy standard projection.

    Args:
        model: Model to modify (in-place)
        steering_vectors: Per-layer steering vectors {layer_idx: [H]}
        harmless_vectors: Optional per-layer harmless directions for biprojection
        components: Component names to modify
        layer_weights: Optional per-layer strength
        strength: Global projection strength
        normalize_vectors: Whether to normalize steering vectors
        norm_preserve: Use norm-preserving method (default True, RECOMMENDED)
        use_biprojection: Orthogonalize against harmless directions
        verbose: Whether to print progress

    Returns:
        Modification statistics
    """
    if norm_preserve:
        return project_weights_norm_preserved(
            model=model,
            steering_vectors=steering_vectors,
            harmless_vectors=harmless_vectors,
            components=components,
            layer_weights=layer_weights,
            strength=strength,
            use_biprojection=use_biprojection,
            verbose=verbose,
        )

    # Legacy standard projection (not recommended)
    log = bind(_LOG, num_layers=len(steering_vectors))

    if verbose:
        print("\nWARNING: Using legacy projection (norms NOT preserved)")
        print("Consider using norm_preserve=True for better model quality\n")

    if components is None:
        components = ["self_attn.o_proj", "mlp.down_proj"]

    # Get model layers
    if hasattr(model, "model"):
        layers = model.model.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        layers = model.layers

    layers_modified = 0
    components_modified = 0
    total_params = 0

    for layer_idx, steering_vector in steering_vectors.items():
        if layer_idx >= len(layers):
            continue

        layer = layers[layer_idx]
        layer_modified = False

        # Normalize
        v = F.normalize(steering_vector.float(), p=2, dim=0) if normalize_vectors else steering_vector.float()

        # Compute projector for legacy method
        projector = torch.outer(v, v)

        # Get layer weight
        layer_weight = 1.0 if layer_weights is None else layer_weights.get(layer_idx, 1.0)
        effective_strength = strength * layer_weight

        for component_name in components:
            try:
                component = layer
                for attr in component_name.split("."):
                    component = getattr(component, attr)

                if hasattr(component, "weight"):
                    project_component(
                        component.weight,
                        projector,
                        effective_strength,
                        in_place=True,
                    )
                    components_modified += 1
                    total_params += component.weight.numel()
                    layer_modified = True

            except AttributeError:
                pass

        if layer_modified:
            layers_modified += 1

    return {
        "layers_modified": layers_modified,
        "components_modified": components_modified,
        "total_parameters_modified": total_params,
        "norm_preserved": False,
    }


def project_with_kernel(
    model: Module,
    steering_vectors: dict[int, Tensor],
    harmless_vectors: dict[int, Tensor] | None = None,
    max_weight: float = 1.0,
    max_weight_position: float | None = None,
    min_weight: float = 0.0,
    min_weight_distance: float | None = None,
    components: list[str] | None = None,
    normalize_vectors: bool = True,
    norm_preserve: bool = True,
    use_biprojection: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Directionally modify weights with kernel-based layer weights distribution.

    Creates a smooth weight distribution across layers with maximum
    at max_weight_position, tapering to min_weight at edges.

    By default uses norm-preserving biprojected modification.

    Args:
        model: Model to modify
        steering_vectors: Per-layer steering vectors
        harmless_vectors: Optional harmless directions for biprojection
        max_weight: Peak projection strength
        max_weight_position: Layer index of peak (None = middle)
        min_weight: Minimum projection strength
        min_weight_distance: Distance over which weight decays
        components: Components to modify
        normalize_vectors: Whether to normalize steering vectors
        norm_preserve: Use norm-preserving method (RECOMMENDED)
        use_biprojection: Orthogonalize against harmless
        verbose: Whether to print progress

    Returns:
        Modification statistics
    """
    num_layers = len(steering_vectors)

    if max_weight_position is None:
        max_weight_position = (num_layers - 1) / 2

    if min_weight_distance is None:
        min_weight_distance = 0.6 * (num_layers - 1)

    # Compute per-layer weights
    layer_weights = {}
    for layer_idx in range(num_layers):
        distance = abs(layer_idx - max_weight_position)

        if distance > min_weight_distance:
            weight = 0.0
        else:
            weight = max_weight + (distance / min_weight_distance) * (
                min_weight - max_weight
            )

        layer_weights[layer_idx] = weight

    if verbose:
        print(f"\nKernel configuration:")
        print(f"  Peak: {max_weight:.2f} at layer {max_weight_position:.1f}")
        print(f"  Min: {min_weight:.2f} within distance {min_weight_distance:.1f}")
        print(f"  Active layers: {sum(1 for w in layer_weights.values() if w > 0)}/{num_layers}")

    return project_weights(
        model,
        steering_vectors,
        harmless_vectors=harmless_vectors,
        components=components,
        layer_weights=layer_weights,
        strength=1.0,
        normalize_vectors=normalize_vectors,
        norm_preserve=norm_preserve,
        use_biprojection=use_biprojection,
        verbose=verbose,
    )
