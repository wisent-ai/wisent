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
    "project_weights_multi_direction",
    "project_weights_titan",
    "project_component",
    "project_component_norm_preserved",
    "project_component_multi_direction",
    "compute_projection_kernel",
    "project_with_kernel",
    "orthogonalize_direction",
    "verify_weight_modification_preservation",
    "TITANRuntimeHooks",
    "apply_titan_steering",
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


def project_component_multi_direction(
    weight_matrix: Tensor,
    steering_vectors: Tensor,
    strengths: list[float] | Tensor | None = None,
    in_place: bool = True,
) -> Tensor:
    """
    Project a weight matrix against MULTIPLE directions while preserving row norms.
    
    This extends project_component_norm_preserved to handle multiple directions,
    as discovered by PRISM or similar multi-directional steering methods.
    
    Directions are applied sequentially, with each projection operating on
    the result of the previous one. This is important because the order can
    matter for non-orthogonal directions.
    
    Args:
        weight_matrix: Weight matrix to modify [out_dim, in_dim]
        steering_vectors: Multiple steering directions [num_directions, out_dim]
        strengths: Per-direction strengths (default: all 1.0)
        in_place: Whether to modify in-place
        
    Returns:
        Modified weight matrix with preserved row norms
    """
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
        
        # Store original norms for final restoration
        original_norms = torch.norm(W, p=2, dim=1, keepdim=True)
        
        # Apply each direction's projection sequentially
        for i in range(num_directions):
            v = steering_vectors[i].to(device).float()
            if v.dim() > 1:
                v = v.view(-1)
            v = F.normalize(v, p=2, dim=0)
            strength = strengths[i]
            
            # Decompose into magnitude and direction
            W_norm = torch.norm(W, p=2, dim=1, keepdim=True)
            W_direction = F.normalize(W, p=2, dim=1)
            
            # Compute projection term
            v_row = v.unsqueeze(0)
            weighted_sum = v_row @ W_direction
            projection_term = v.unsqueeze(1) @ weighted_sum
            
            # Project direction
            W_direction_new = W_direction - strength * projection_term
            W_direction_new = F.normalize(W_direction_new, p=2, dim=1)
            
            # Recombine with current norms
            W = W_norm * W_direction_new
        
        # Final norm restoration (optional - could use original_norms instead)
        # Using current norms preserves the sequential projection behavior
        result = W.to(original_dtype)
        
        if in_place:
            weight_matrix.copy_(result)
            result = weight_matrix
        
        log.debug(
            "Multi-direction norm-preserved projection",
            extra={
                "weight_shape": weight_matrix.shape,
                "num_directions": num_directions,
                "strengths": strengths,
                "in_place": in_place,
            },
        )
    
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
    """
    Norm-Preserving Multi-Directional Weight Modification for PRISM.
    
    Projects out multiple steering directions per layer, as discovered by
    PRISM or similar multi-directional methods. Each layer can have k
    directions that are ablated together.
    
    Args:
        model: Model to modify (in-place)
        multi_steering_vectors: Per-layer multi-direction tensors 
                                {layer_idx: [num_directions, hidden_dim]}
        harmless_vectors: Optional per-layer harmless directions for biprojection
        components: Component names to modify
                   Default: ["self_attn.o_proj", "mlp.down_proj"]
        layer_weights: Optional per-layer global strength multiplier
        direction_strengths: Per-direction strengths within each layer
                            If None, all directions use strength=1.0
        global_strength: Global multiplier for all projections
        use_biprojection: Whether to orthogonalize against harmless directions
        verbose: Whether to print progress
        
    Returns:
        Dictionary with modification statistics
        
    Example:
        >>> from wisent.core.weight_modification import project_weights_multi_direction
        >>> # multi_vectors[layer_idx] has shape [num_directions, hidden_dim]
        >>> stats = project_weights_multi_direction(
        ...     model,
        ...     multi_vectors,
        ...     direction_strengths=[1.0, 0.8, 0.6],  # Decay for subsequent directions
        ...     global_strength=1.0,
        ... )
    """
    log = bind(_LOG, num_layers=len(multi_steering_vectors))
    
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
    total_directions_applied = 0
    
    if verbose:
        # Determine number of directions from first layer
        first_layer_vecs = next(iter(multi_steering_vectors.values()))
        num_directions = first_layer_vecs.shape[0] if first_layer_vecs.dim() > 1 else 1
        
        print(f"\n{'='*60}")
        print("PRISM MULTI-DIRECTIONAL WEIGHT MODIFICATION")
        print(f"{'='*60}")
        print(f"Layers: {len(multi_steering_vectors)}")
        print(f"Directions per layer: {num_directions}")
        print(f"Components: {components}")
        print(f"Global strength: {global_strength}")
        print(f"Direction strengths: {direction_strengths or 'uniform (1.0)'}")
        print(f"Biprojection: {use_biprojection and harmless_vectors is not None}")
        print(f"{'='*60}\n")
    
    for layer_idx, steering_vectors in multi_steering_vectors.items():
        if layer_idx >= len(layers):
            log.warning(f"Layer {layer_idx} out of range, skipping")
            continue
        
        layer = layers[layer_idx]
        layer_modified = False
        
        # Get layer weight multiplier
        layer_weight = 1.0 if layer_weights is None else layer_weights.get(layer_idx, 1.0)
        effective_global_strength = global_strength * layer_weight
        
        # Handle single direction case (backward compatibility)
        if steering_vectors.dim() == 1:
            steering_vectors = steering_vectors.unsqueeze(0)
        
        num_directions = steering_vectors.shape[0]
        
        # Compute per-direction strengths
        if direction_strengths is None:
            strengths = [effective_global_strength] * num_directions
        else:
            strengths = [s * effective_global_strength for s in direction_strengths[:num_directions]]
            # Pad if not enough strengths provided
            while len(strengths) < num_directions:
                strengths.append(effective_global_strength)
        
        # Optional: orthogonalize against harmless
        if use_biprojection and harmless_vectors is not None and layer_idx in harmless_vectors:
            harmless_vec = harmless_vectors[layer_idx]
            processed_vectors = []
            for i in range(num_directions):
                v = orthogonalize_direction(steering_vectors[i], harmless_vec)
                processed_vectors.append(v)
            steering_vectors = torch.stack(processed_vectors)
        else:
            # Normalize each direction
            steering_vectors = F.normalize(steering_vectors.float(), p=2, dim=1)
        
        for component_name in components:
            try:
                component = layer
                for attr in component_name.split("."):
                    component = getattr(component, attr)
                
                if not hasattr(component, "weight"):
                    log.warning(f"Component {component_name} has no weight attribute")
                    continue
                
                weight_matrix = component.weight
                
                # Compute norm before
                norm_before = torch.norm(weight_matrix.float(), p=2, dim=1).mean().item()
                
                # Apply multi-direction projection
                project_component_multi_direction(
                    weight_matrix,
                    steering_vectors,
                    strengths=strengths,
                    in_place=True,
                )
                
                # Verify norm preservation
                norm_after = torch.norm(weight_matrix.float(), p=2, dim=1).mean().item()
                norm_change = abs(norm_after - norm_before) / norm_before * 100
                
                components_modified += 1
                total_params += weight_matrix.numel()
                total_directions_applied += num_directions
                layer_modified = True
                
                if verbose:
                    print(
                        f"  Layer {layer_idx:3d} | {component_name:20s} | "
                        f"dirs={num_directions} | "
                        f"norm_change={norm_change:.4f}%"
                    )
                    
            except AttributeError as e:
                log.warning(f"Could not access {component_name} in layer {layer_idx}: {e}")
        
        if layer_modified:
            layers_modified += 1
    
    if verbose:
        print(f"\n{'='*60}")
        print("PRISM MODIFICATION COMPLETE")
        print(f"{'='*60}")
        print(f"  Layers modified: {layers_modified}")
        print(f"  Components modified: {components_modified}")
        print(f"  Total directions applied: {total_directions_applied}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Norms preserved: YES")
        print(f"{'='*60}\n")
    
    return {
        "layers_modified": layers_modified,
        "components_modified": components_modified,
        "total_parameters_modified": total_params,
        "total_directions_applied": total_directions_applied,
        "norm_preserved": True,
    }


# =============================================================================
# TITAN WEIGHT MODIFICATION AND RUNTIME HOOKS
# =============================================================================

class TITANRuntimeHooks:
    """
    Runtime hook system for TITAN dynamic steering.
    
    TITAN uses a hybrid approach:
    1. Effective directions are pre-baked into model weights (static)
    2. Gate and intensity networks run at inference time (dynamic)
    
    This class manages the forward hooks that enable dynamic gating
    and intensity modulation during inference.
    
    Usage:
        # After training TITAN
        titan_result = method.train_titan(pair_set)
        
        # Bake directions into weights
        project_weights_titan(model, titan_result)
        
        # Install runtime hooks for dynamic behavior
        hooks = TITANRuntimeHooks(model, titan_result)
        hooks.install()
        
        # Run inference (hooks automatically apply gating/intensity)
        output = model.generate(...)
        
        # Remove hooks when done
        hooks.remove()
    """
    
    def __init__(
        self,
        model: Module,
        titan_result,  # TITANResult
        base_strength: float = 1.0,
        gate_threshold: float = 0.5,
        use_soft_gating: bool = True,
    ):
        """
        Initialize TITAN runtime hooks.
        
        Args:
            model: The model to hook into
            titan_result: TITANResult from TITAN training
            base_strength: Base steering strength multiplier
            gate_threshold: Threshold for hard gating (if use_soft_gating=False)
            use_soft_gating: Use soft sigmoid gating vs hard threshold
        """
        self.model = model
        self.titan_result = titan_result
        self.base_strength = base_strength
        self.gate_threshold = gate_threshold
        self.use_soft_gating = use_soft_gating
        
        self._hooks = []
        self._sensor_activation = None
        self._current_gate = None
        self._current_intensities = None
        
        # Get model layers
        if hasattr(model, "model"):
            self._layers = model.model.layers
        elif hasattr(model, "transformer"):
            self._layers = model.transformer.h
        else:
            self._layers = model.layers
        
        # Map layer names to indices
        self._layer_name_to_idx = {}
        for layer_name in titan_result.layer_order:
            try:
                idx = int(str(layer_name).split("_")[-1])
                self._layer_name_to_idx[layer_name] = idx
            except (ValueError, IndexError):
                pass
        
        # Find sensor layer index
        sensor_layer_name = titan_result.metadata.get("sensor_layer")
        self._sensor_layer_idx = self._layer_name_to_idx.get(sensor_layer_name, 15)
    
    def install(self) -> None:
        """Install forward hooks on the model."""
        self.remove()  # Clear any existing hooks
        
        # Install sensor hook to capture activation for gating
        if self._sensor_layer_idx < len(self._layers):
            sensor_hook = self._layers[self._sensor_layer_idx].register_forward_hook(
                self._sensor_hook
            )
            self._hooks.append(sensor_hook)
        
        # Install steering hooks on all steering layers
        for layer_name in self.titan_result.layer_order:
            layer_idx = self._layer_name_to_idx.get(layer_name)
            if layer_idx is not None and layer_idx < len(self._layers):
                # Use post-hook to modify output after layer computation
                steering_hook = self._layers[layer_idx].register_forward_hook(
                    lambda module, input, output, ln=layer_name: self._steering_hook(
                        module, input, output, ln
                    )
                )
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
        # Extract hidden states from output
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Use last token's hidden state for gating decision
        if hidden_states.dim() == 3:  # [batch, seq, hidden]
            sensor_h = hidden_states[:, -1, :]  # [batch, hidden]
        else:
            sensor_h = hidden_states
        
        # Store for steering hooks
        self._sensor_activation = sensor_h.detach()
        
        # Compute gate
        with torch.no_grad():
            if self.use_soft_gating:
                self._current_gate = self.titan_result.predict_gate(sensor_h)
            else:
                gate_value = self.titan_result.predict_gate(sensor_h)
                self._current_gate = (gate_value > self.gate_threshold).float()
            
            # Compute per-layer intensities
            self._current_intensities = self.titan_result.predict_intensity(sensor_h)
        
        return output
    
    def _steering_hook(self, module, input, output, layer_name):
        """Apply dynamic steering to layer output."""
        # Skip if gate/intensities not computed yet
        if self._current_gate is None or self._current_intensities is None:
            return output
        
        # Extract hidden states
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None
        
        # Get effective direction for this layer
        direction = self.titan_result.get_effective_direction(layer_name)
        direction = direction.to(hidden_states.device)
        
        # Get intensity for this layer
        intensity = self._current_intensities.get(layer_name, torch.ones(1))
        intensity = intensity.to(hidden_states.device)
        
        # Apply steering: h' = h + gate * intensity * base_strength * direction
        gate = self._current_gate.to(hidden_states.device)
        
        # Handle dimensions
        if hidden_states.dim() == 3:  # [batch, seq, hidden]
            # Expand gate and intensity for broadcasting
            gate = gate.view(-1, 1, 1)  # [batch, 1, 1]
            intensity = intensity.view(-1, 1, 1)  # [batch, 1, 1]
            direction = direction.view(1, 1, -1)  # [1, 1, hidden]
        elif hidden_states.dim() == 2:  # [batch, hidden]
            gate = gate.view(-1, 1)
            intensity = intensity.view(-1, 1)
            direction = direction.view(1, -1)
        
        # Apply steering
        steering_delta = gate * intensity * self.base_strength * direction
        hidden_states = hidden_states + steering_delta
        
        # Return modified output
        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states
    
    def get_current_gate(self) -> float | None:
        """Get the current gate value (for debugging/monitoring)."""
        if self._current_gate is not None:
            return self._current_gate.mean().item()
        return None
    
    def get_current_intensities(self) -> dict | None:
        """Get current per-layer intensities (for debugging/monitoring)."""
        if self._current_intensities is not None:
            return {k: v.mean().item() for k, v in self._current_intensities.items()}
        return None


def project_weights_titan(
    model: Module,
    titan_result,  # TITANResult
    components: list[str] | None = None,
    base_strength: float = 1.0,
    use_learned_intensities: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Bake TITAN effective directions into model weights.
    
    This is the static part of TITAN steering - the effective directions
    (weighted combination of manifold directions) are projected into the
    model weights using norm-preserving projection.
    
    For full dynamic behavior, also install TITANRuntimeHooks after this.
    
    Args:
        model: Model to modify (in-place)
        titan_result: TITANResult from TITAN training
        components: Component names to modify
                   Default: ["self_attn.o_proj", "mlp.down_proj"]
        base_strength: Base projection strength
        use_learned_intensities: Use TITAN's learned layer intensities as weights
        verbose: Whether to print progress
        
    Returns:
        Dictionary with modification statistics
        
    Example:
        >>> # Train TITAN
        >>> titan_result = method.train_titan(pair_set)
        >>> 
        >>> # Bake directions into weights
        >>> stats = project_weights_titan(model, titan_result)
        >>> 
        >>> # Optionally install runtime hooks for dynamic gating
        >>> hooks = TITANRuntimeHooks(model, titan_result)
        >>> hooks.install()
    """
    log = bind(_LOG, num_layers=len(titan_result.directions))
    
    if components is None:
        components = ["self_attn.o_proj", "mlp.down_proj"]
    
    # Get model layers
    if hasattr(model, "model"):
        layers = model.model.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        layers = model.layers
    
    # Compute effective directions and layer weights
    effective_vectors = {}
    layer_weights = {}
    
    for layer_name in titan_result.layer_order:
        # Get effective direction (weighted combination of manifold)
        eff_dir = titan_result.get_effective_direction(layer_name)
        
        # Map layer name to index
        try:
            layer_idx = int(str(layer_name).split("_")[-1])
        except (ValueError, IndexError):
            continue
        
        effective_vectors[layer_idx] = eff_dir
        
        # Optionally use learned intensities as layer weights
        if use_learned_intensities:
            # Get average intensity for this layer from a neutral input
            # For static baking, we use the learned direction weights as proxy
            dir_weights = titan_result.direction_weights.get(layer_name)
            if dir_weights is not None:
                # Higher weight entropy = more important layer
                weight = 1.0 + (dir_weights.max() - dir_weights.min()).item()
            else:
                weight = 1.0
            layer_weights[layer_idx] = weight
    
    if verbose:
        print(f"\n{'='*60}")
        print("TITAN WEIGHT MODIFICATION")
        print(f"{'='*60}")
        print(f"Layers: {len(effective_vectors)}")
        print(f"Components: {components}")
        print(f"Base strength: {base_strength}")
        print(f"Using learned intensities: {use_learned_intensities}")
        print(f"{'='*60}\n")
    
    # Use standard norm-preserving projection
    stats = project_weights_norm_preserved(
        model=model,
        steering_vectors=effective_vectors,
        harmless_vectors=None,
        components=components,
        layer_weights=layer_weights if use_learned_intensities else None,
        strength=base_strength,
        use_biprojection=False,
        verbose=verbose,
    )
    
    stats["titan_layers"] = len(titan_result.layer_order)
    stats["titan_directions_per_layer"] = titan_result.directions[titan_result.layer_order[0]].shape[0]
    
    return stats


def apply_titan_steering(
    model: Module,
    titan_result,  # TITANResult
    mode: str = "hybrid",
    base_strength: float = 1.0,
    components: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Apply TITAN steering to a model with the specified mode.
    
    Modes:
    - "static": Only bake directions into weights (no dynamic behavior)
    - "dynamic": Only use runtime hooks (no weight modification)  
    - "hybrid": Bake directions + install hooks (recommended)
    
    Args:
        model: Model to modify
        titan_result: TITANResult from TITAN training
        mode: Application mode ("static", "dynamic", or "hybrid")
        base_strength: Base steering strength
        components: Weight components to modify (for static/hybrid)
        verbose: Whether to print progress
        
    Returns:
        Dictionary with:
        - "stats": Weight modification statistics (if applicable)
        - "hooks": TITANRuntimeHooks instance (if applicable)
        
    Example:
        >>> result = apply_titan_steering(model, titan_result, mode="hybrid")
        >>> 
        >>> # Run inference
        >>> output = model.generate(...)
        >>> 
        >>> # Check gate value
        >>> print(f"Gate: {result['hooks'].get_current_gate()}")
        >>> 
        >>> # Remove hooks when done
        >>> result['hooks'].remove()
    """
    result = {}
    
    if mode in ("static", "hybrid"):
        stats = project_weights_titan(
            model=model,
            titan_result=titan_result,
            components=components,
            base_strength=base_strength if mode == "static" else 1.0,
            use_learned_intensities=True,
            verbose=verbose,
        )
        result["stats"] = stats
    
    if mode in ("dynamic", "hybrid"):
        hooks = TITANRuntimeHooks(
            model=model,
            titan_result=titan_result,
            base_strength=base_strength,
            use_soft_gating=True,
        )
        hooks.install()
        result["hooks"] = hooks
        
        if verbose:
            print(f"\nTITAN Runtime Hooks installed")
            print(f"  Sensor layer: {hooks._sensor_layer_idx}")
            print(f"  Steering layers: {len(titan_result.layer_order)}")
            print(f"  Mode: {mode}")
    
    return result


def verify_weight_modification_preservation(
    original_weights: "Tensor",
    modified_weights: "Tensor",
    threshold: float = 0.95,
) -> tuple[bool, dict[str, float]]:
    """
    Verify that weight modification preserved subspace membership.
    
    Based on the Universal Subspace Hypothesis, good modifications should
    keep weights within the same low-dimensional subspace that models learn.
    
    This is a wrapper around `wisent.core.universal_subspace.verify_subspace_preservation`
    for use within the weight modification module.
    
    Arguments:
        original_weights: Original weight matrix
        modified_weights: Modified weight matrix  
        threshold: Minimum alignment score for preservation
        
    Returns:
        Tuple of (is_preserved, metrics_dict)
    """
    from wisent.core.universal_subspace import verify_subspace_preservation
    return verify_subspace_preservation(original_weights, modified_weights, threshold)

