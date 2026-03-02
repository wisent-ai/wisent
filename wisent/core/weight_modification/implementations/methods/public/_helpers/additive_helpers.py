"""Extracted from additive.py - bake_steering_with_kernel implementation."""
from wisent.core.utils.config_tools.constants import KERNEL_DISTANCE_FRACTION


def bake_steering_with_kernel_impl(
    model,
    steering_vectors,
    max_alpha: float,
    max_alpha_position: float,
    min_alpha: float,
    min_alpha_distance: float,
    components: int,
    method: str,
    verbose: bool,
):
    """Implement the kernel-based alpha distribution for steering.

    Computes per-layer steering strengths based on distance from a peak
    position using linear interpolation between maximum and minimum alpha
    values, then applies the weighted steering vectors to model weights.

    Args:
        model: The model to modify
        steering_vectors: Dict mapping layer indices to steering vectors
        max_alpha: Maximum steering strength at the peak
        max_alpha_position: Layer index where alpha is maximum
        min_alpha: Minimum steering strength
        min_alpha_distance: Distance from peak where alpha reaches minimum
        components: Number of steering components
        method: Weight modification method name
        verbose: Whether to print configuration details

    Returns:
        Modified model with steering baked into weights
    """
    from wisent.core.weight_modification.methods.additive import (
        bake_steering_into_weights,
    )

    num_layers = len(steering_vectors)

    if max_alpha_position is None:
        max_alpha_position = (num_layers - 1) / 2

    if min_alpha_distance is None:
        min_alpha_distance = KERNEL_DISTANCE_FRACTION * (num_layers - 1)

    # Compute per-layer alphas
    layer_weights = {}
    for layer_idx in range(num_layers):
        distance = abs(layer_idx - max_alpha_position)

        if distance > min_alpha_distance:
            alpha_scale = min_alpha / max_alpha
        else:
            # Linear interpolation
            alpha_scale = 1.0 + (distance / min_alpha_distance) * (
                min_alpha / max_alpha - 1.0
            )

        layer_weights[layer_idx] = alpha_scale

    if verbose:
        print(f"\nKernel configuration:")
        print(f"  Peak: a={max_alpha:.2f} at layer "
              f"{max_alpha_position:.1f}")
        print(f"  Min: a={min_alpha:.2f} within distance "
              f"{min_alpha_distance:.1f}")

    return bake_steering_into_weights(
        model,
        steering_vectors,
        components=components,
        alpha=max_alpha,
        layer_weights=layer_weights,
        method=method,
        verbose=verbose,
    )
