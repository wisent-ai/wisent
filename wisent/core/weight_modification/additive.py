"""
Additive weight modification: Bake steering vectors into model weights.

This is a more conservative approach than directional projection. Instead of removing
dimensions, we add bias toward the desired direction directly into the weights.

The goal: Make the steering effect permanent without runtime hooks, but preserve
all model capabilities.

Key insight: Activation steering does `h' = h + αv` at runtime. We can
approximate this by modifying weights to produce similar effects without hooks.

Approaches:

1. **Output bias addition** (simplest):
   Add steering vector as bias to layer outputs

2. **Weight shifting** (more sophisticated):
   Modify weight matrices to shift outputs in steering direction

Trade-off:
- Additive is less aggressive than directional projection
- Preserves original model capabilities better
- But may not be as effective at changing behavior
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from wisent.core.errors import UnknownTypeError
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

__all__ = [
    "bake_steering_into_weights",
    "bake_steering_into_component",
    "add_output_bias",
    "bake_steering_with_kernel",
]

_LOG = setup_logger(__name__)


def add_output_bias(
    module: Module,
    bias_vector: Tensor,
    alpha: float = 1.0,
) -> None:
    """
    Add bias vector to a module's output.

    If module has a bias parameter, adds to it. Otherwise, creates new bias.

    Args:
        module: Module to modify (typically Linear layer)
        bias_vector: Bias to add [out_dim]
        alpha: Scaling factor for bias

    Example:
        >>> layer = model.model.layers[15].self_attn.o_proj
        >>> steering_vec = torch.randn(4096)
        >>> add_output_bias(layer, steering_vec, alpha=1.0)
    """
    log = bind(_LOG)

    scaled_bias = alpha * bias_vector.to(device=module.weight.device, dtype=module.weight.dtype)

    if module.bias is None:
        # Create new bias parameter
        module.bias = torch.nn.Parameter(scaled_bias)
        log.debug("Created new bias parameter", extra={"shape": scaled_bias.shape})
    else:
        # Add to existing bias
        module.bias.data.add_(scaled_bias)
        log.debug("Added to existing bias", extra={"shape": scaled_bias.shape})


def bake_steering_into_component(
    module: Module,
    steering_vector: Tensor,
    alpha: float,
    method: str = "bias",
) -> None:
    """
    Bake steering vector into a single component.

    Args:
        module: Module to modify (e.g., attention o_proj, MLP down_proj)
        steering_vector: Steering vector [hidden_dim]
        alpha: Steering strength
        method: How to bake steering:
               - "bias": Add as output bias (simplest, most reliable)
               - "weight": Modify weight matrix (more sophisticated)

    Example:
        >>> o_proj = model.model.layers[15].self_attn.o_proj
        >>> steering = torch.randn(4096)
        >>> bake_steering_into_component(o_proj, steering, alpha=1.0, method="bias")
    """
    log = bind(_LOG, method=method)

    if method == "bias":
        add_output_bias(module, steering_vector, alpha)
    elif method == "weight":
        # Modify weights to add steering effect
        # We want: h_out = W @ h_in + α*v
        # To bake this into weights, we add α*v as a constant output shift
        # Since this needs to work for any input, we use a rank-1 approximation

        # The idea: Add a small rank-1 perturbation to W that biases outputs toward v
        # W' = W + α * v @ e^T where e is a unit vector
        # But since we don't know which input dimensions to weight, we use mean field approximation

        # Simpler approach: Add steering vector uniformly across all output directions
        # This adds α*v/d to each row of W, where d is input dimension
        # Result: For typical centered inputs, output gets shifted by ~α*v

        weight = module.weight  # [out_dim, in_dim]
        in_dim = weight.shape[1]
        scaled_steering = (alpha / in_dim) * steering_vector.to(device=weight.device, dtype=weight.dtype)  # [out_dim]

        # Add steering vector contribution to each column of weight matrix
        # This means: W'[i,j] = W[i,j] + α*v[i]/in_dim
        # When multiplied by input, this adds approximately α*v[i] to output[i]
        with torch.no_grad():
            for i in range(weight.shape[1]):
                weight[:, i].add_(scaled_steering)

        log.debug(
            "Modified weights with rank-1 steering",
            extra={"alpha": alpha, "in_dim": in_dim}
        )
    else:
        raise UnknownTypeError(entity_type="method", value=method, valid_values=["bias", "rank1"])


def bake_steering_into_weights(
    model: Module,
    steering_vectors: dict[int, Tensor],
    components: list[str] | None = None,
    alpha: float = 1.0,
    layer_weights: dict[int, float] | None = None,
    method: str = "bias",
    verbose: bool = True,
) -> dict[str, int]:
    """
    Bake steering vectors into model weights permanently.

    This creates a model that exhibits steering behavior without needing
    runtime hooks or steering vectors.

    Args:
        model: Model to modify (in-place)
        steering_vectors: Per-layer steering vectors {layer_idx: [H]}
        components: Component names to modify (e.g., ["self_attn.o_proj"])
                   If None, modifies attention output projections only
        alpha: Global steering strength
        layer_weights: Optional per-layer scaling {layer_idx: weight}
        method: "bias" or "weight"
        verbose: Whether to print progress

    Returns:
        Dictionary with modification statistics

    Example:
        >>> from wisent.core.weight_modification import bake_steering_into_weights
        >>> # Compute steering vectors from contrastive pairs
        >>> steering_vectors = {...}
        >>> # Bake steering into model weights
        >>> stats = bake_steering_into_weights(model, steering_vectors, alpha=1.0)
        >>> # Model now exhibits steering behavior permanently
        >>> model.save_pretrained("path/to/steered-model")
    """
    log = bind(_LOG, num_layers=len(steering_vectors))

    if components is None:
        # Default: only modify attention output projection
        # (MLP is riskier to modify permanently)
        components = ["self_attn.o_proj"]

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
        print(f"\nBaking steering into {len(steering_vectors)} layers...")
        print(f"Components: {components}")
        print(f"Global alpha: {alpha}")
        print(f"Method: {method}")

    for layer_idx, steering_vector in steering_vectors.items():
        if layer_idx >= len(layers):
            log.warning(f"Layer {layer_idx} out of range, skipping")
            continue

        layer = layers[layer_idx]
        layer_modified = False

        # Get effective alpha for this layer
        layer_alpha = alpha
        if layer_weights is not None:
            layer_alpha *= layer_weights.get(layer_idx, 1.0)

        for component_name in components:
            try:
                # Navigate to component
                component = layer
                for attr in component_name.split("."):
                    component = getattr(component, attr)

                # Bake steering
                bake_steering_into_component(
                    component,
                    steering_vector,
                    layer_alpha,
                    method=method,
                )

                components_modified += 1
                total_params += component.weight.numel()
                if component.bias is not None:
                    total_params += component.bias.numel()
                layer_modified = True

                if verbose:
                    print(
                        f"  Layer {layer_idx:3d} | {component_name:20s} | "
                        f"alpha={layer_alpha:.3f} | "
                        f"steering_norm={steering_vector.norm():.2f}"
                    )

            except AttributeError as e:
                log.warning(
                    f"Could not access {component_name} in layer {layer_idx}: {e}"
                )

        if layer_modified:
            layers_modified += 1

    if verbose:
        print(f"\nSteering baked into weights:")
        print(f"  Layers modified: {layers_modified}")
        print(f"  Components modified: {components_modified}")
        print(f"  Total parameters: {total_params:,}")

    return {
        "layers_modified": layers_modified,
        "components_modified": components_modified,
        "total_parameters_modified": total_params,
    }


def bake_steering_with_kernel(
    model: Module,
    steering_vectors: dict[int, Tensor],
    max_alpha: float = 2.0,
    max_alpha_position: float | None = None,
    min_alpha: float = 0.5,
    min_alpha_distance: float | None = None,
    components: list[str] | None = None,
    method: str = "bias",
    verbose: bool = True,
) -> dict[str, int]:
    """
    Bake steering with kernel-based alpha distribution across layers.

    Args:
        model: Model to modify
        steering_vectors: Per-layer steering vectors
        max_alpha: Peak steering strength
        max_alpha_position: Layer of peak (None = middle)
        min_alpha: Minimum steering strength
        min_alpha_distance: Decay distance (None = 60% of layers)
        components: Components to modify
        method: "bias" or "weight"
        verbose: Whether to print progress

    Returns:
        Modification statistics

    Example:
        >>> # Bake steering with maximum at layer 15, tapering to edges
        >>> stats = bake_steering_with_kernel(
        ...     model,
        ...     steering_vectors,
        ...     max_alpha=2.0,
        ...     max_alpha_position=15.0,
        ...     min_alpha=0.5,
        ... )
    """
    num_layers = len(steering_vectors)

    if max_alpha_position is None:
        max_alpha_position = (num_layers - 1) / 2

    if min_alpha_distance is None:
        min_alpha_distance = 0.6 * (num_layers - 1)

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
        print(f"  Peak: α={max_alpha:.2f} at layer {max_alpha_position:.1f}")
        print(f"  Min: α={min_alpha:.2f} within distance {min_alpha_distance:.1f}")

    return bake_steering_into_weights(
        model,
        steering_vectors,
        components=components,
        alpha=max_alpha,
        layer_weights=layer_weights,
        method=method,
        verbose=verbose,
    )
