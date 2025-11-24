"""
Direction interpolation for smooth steering between layers.

Allows float layer indices (e.g., 15.5) by linearly interpolating
between adjacent layer steering vectors.

Adapted from Heretic's abliterate() direction selection logic.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["interpolate_steering_vectors", "interpolate_steering_vector"]


def interpolate_steering_vector(
    steering_vectors: dict[int, Tensor],
    layer_index: float,
) -> Tensor:
    """
    Interpolate steering vector at a float layer index.

    For integer indices, returns the exact steering vector.
    For float indices (e.g., 15.5), linearly interpolates between
    adjacent layers (15 and 16).

    Args:
        steering_vectors: Dictionary mapping layer index to steering vector [H]
        layer_index: Float layer index (e.g., 15.5)

    Returns:
        Interpolated steering vector [H]

    Raises:
        ValueError: If layer_index is out of bounds

    Example:
        >>> vectors = {15: torch.randn(4096), 16: torch.randn(4096)}
        >>> v = interpolate_steering_vector(vectors, 15.5)
        >>> # Returns: 0.5 * vectors[15] + 0.5 * vectors[16]
    """
    # Get min and max layer indices
    min_layer = min(steering_vectors.keys())
    max_layer = max(steering_vectors.keys())

    if layer_index < min_layer or layer_index > max_layer:
        raise ValueError(
            f"layer_index {layer_index} out of bounds [{min_layer}, {max_layer}]"
        )

    # Integer index - return exact vector
    if layer_index == int(layer_index):
        return steering_vectors[int(layer_index)]

    # Float index - interpolate
    lower_idx = int(layer_index)
    upper_idx = lower_idx + 1

    # Handle edge case where upper_idx exceeds max
    if upper_idx > max_layer:
        return steering_vectors[lower_idx]

    # Linear interpolation weight
    weight = layer_index - lower_idx

    # Interpolate: (1-w) * lower + w * upper
    lower_vector = steering_vectors[lower_idx]
    upper_vector = steering_vectors[upper_idx]

    interpolated = (1.0 - weight) * lower_vector + weight * upper_vector

    return interpolated


def interpolate_steering_vectors(
    steering_vectors: dict[int, Tensor],
    layer_indices: list[float] | None = None,
) -> dict[int, Tensor]:
    """
    Create new steering vector dictionary with interpolated values.

    This allows using a single interpolated direction across all layers
    or specific interpolated directions at float layer positions.

    Args:
        steering_vectors: Original steering vectors per layer
        layer_indices: List of float layer indices to interpolate at.
                      If None, returns original vectors unchanged.

    Returns:
        Dictionary mapping integer layer index to interpolated steering vector

    Example:
        >>> # Use direction at layer 15.5 for all layers
        >>> vectors = {i: torch.randn(4096) for i in range(32)}
        >>> global_direction = interpolate_steering_vector(vectors, 15.5)
        >>> steered = {i: global_direction for i in range(32)}

        >>> # Or interpolate at specific positions
        >>> indices = [0, 7.5, 15.5, 23.5, 31]
        >>> steered = interpolate_steering_vectors(vectors, indices)
    """
    if layer_indices is None:
        return steering_vectors

    # Map float indices to integer layer positions
    min_layer = min(steering_vectors.keys())
    max_layer = max(steering_vectors.keys())
    all_layers = list(range(min_layer, max_layer + 1))

    # If fewer indices than layers, repeat or interpolate
    if len(layer_indices) == 1:
        # Single global direction - use for all layers
        global_vector = interpolate_steering_vector(steering_vectors, layer_indices[0])
        return {layer: global_vector.clone() for layer in all_layers}

    # Multiple indices - map to layers
    interpolated = {}
    for layer in all_layers:
        # Find nearest index in layer_indices
        # Simple linear mapping: scale layer to index range
        scaled = (layer - min_layer) / (max_layer - min_layer)
        target_idx = scaled * (len(layer_indices) - 1)

        # Interpolate within layer_indices
        if target_idx == int(target_idx):
            idx = int(target_idx)
            vector = interpolate_steering_vector(
                steering_vectors,
                layer_indices[idx],
            )
        else:
            lower_idx = int(target_idx)
            upper_idx = lower_idx + 1
            weight = target_idx - lower_idx

            lower_vector = interpolate_steering_vector(
                steering_vectors,
                layer_indices[lower_idx],
            )
            upper_vector = interpolate_steering_vector(
                steering_vectors,
                layer_indices[upper_idx],
            )

            vector = (1.0 - weight) * lower_vector + weight * upper_vector

        interpolated[layer] = vector

    return interpolated


def get_global_steering_direction(
    steering_vectors: dict[int, Tensor],
    direction_index: float,
) -> dict[int, Tensor]:
    """
    Create global steering direction by using a single interpolated vector
    across all layers.

    This is the Heretic-style "global direction" approach adapted for steering.

    Args:
        steering_vectors: Per-layer steering vectors
        direction_index: Float layer index to interpolate at (e.g., 15.5)

    Returns:
        Dictionary with same keys as steering_vectors, all pointing to
        the interpolated direction

    Example:
        >>> vectors = {i: torch.randn(4096) for i in range(32)}
        >>> # Use direction at layer 15.5 globally
        >>> global_steered = get_global_steering_direction(vectors, 15.5)
        >>> # All layers now use the same steering direction
    """
    global_vector = interpolate_steering_vector(steering_vectors, direction_index)
    return {layer: global_vector.clone() for layer in steering_vectors.keys()}


def compute_interpolated_metrics(
    steering_vectors: dict[int, Tensor],
    layer_indices: list[float],
) -> dict[str, float]:
    """
    Compute metrics about interpolated steering vectors.

    Useful for understanding smoothness and coherence of interpolation.

    Args:
        steering_vectors: Original steering vectors
        layer_indices: Float layer indices to analyze

    Returns:
        Dictionary with metrics:
        - mean_norm: Average L2 norm of interpolated vectors
        - std_norm: Standard deviation of L2 norms
        - mean_cosine_similarity: Average cosine similarity between adjacent
        - min_cosine_similarity: Minimum cosine similarity between adjacent

    Example:
        >>> vectors = {i: torch.randn(4096) for i in range(32)}
        >>> indices = [0, 15.5, 31]
        >>> metrics = compute_interpolated_metrics(vectors, indices)
        >>> print(metrics)
    """
    interpolated_vectors = [
        interpolate_steering_vector(steering_vectors, idx) for idx in layer_indices
    ]

    # Compute norms
    norms = [v.norm(p=2).item() for v in interpolated_vectors]
    mean_norm = sum(norms) / len(norms)
    std_norm = (
        sum((n - mean_norm) ** 2 for n in norms) / len(norms)
    ) ** 0.5

    # Compute cosine similarities between adjacent
    cosine_sims = []
    for i in range(len(interpolated_vectors) - 1):
        v1 = interpolated_vectors[i]
        v2 = interpolated_vectors[i + 1]
        cos_sim = torch.nn.functional.cosine_similarity(
            v1.unsqueeze(0),
            v2.unsqueeze(0),
        ).item()
        cosine_sims.append(cos_sim)

    mean_cosine = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 1.0
    min_cosine = min(cosine_sims) if cosine_sims else 1.0

    return {
        "mean_norm": mean_norm,
        "std_norm": std_norm,
        "mean_cosine_similarity": mean_cosine,
        "min_cosine_similarity": min_cosine,
    }
