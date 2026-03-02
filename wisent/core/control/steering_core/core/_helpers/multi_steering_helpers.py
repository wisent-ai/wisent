"""Convenience function for multi-steering operations.

Extracted from multi_steering.py to keep file under 300 lines.
"""
from __future__ import annotations

from wisent.core.utils import resolve_default_device

def run_multi_steer(
    vector_specs: list[str],
    model_name: str,
    prompt: str,
    method: str = "CAA",
    layer: int | None = None,
    max_new_tokens: int = None,
    temperature: float = None,
    top_p: float = None,
    device: str | None = None,
    verbose: bool = True,
) -> str:
    """Convenience function to run multi-steering.

    Args:
        vector_specs: List of "path:weight" specifications
        model_name: Name of model to load
        prompt: Input prompt
        method: Steering method to use (kept for backward compatibility)
        layer: Target layer (will be inferred from vectors if not specified)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (uses inference config if None)
        top_p: Top-p sampling parameter (uses inference config if None)
        device: Device to use
        verbose: Whether to print progress

    Returns:
        Generated text
    """
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.steering_core.multi_steering import MultiSteering

    # Initialize model
    if verbose:
        print(f"\nLoading model: {model_name}")

    chosen_device = device or resolve_default_device()

    # Load model using WisentModel
    model = WisentModel(
        model_name=model_name,
        device=chosen_device
    )

    # Initialize multi-steering with specified method
    multi_steer = MultiSteering(device=chosen_device, method=method)

    # Load vectors
    multi_steer.load_vectors(vector_specs)

    # Override layer if specified
    if layer is not None:
        multi_steer.layer = layer
        if verbose:
            print(f"Overriding layer to: {layer}")

    # Combine vectors with normalization
    multi_steer.combine_vectors(normalize=True)

    # Apply steering (non-streaming)
    output = multi_steer.apply_steering(
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )

    return output
