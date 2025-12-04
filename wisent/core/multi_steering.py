"""Multi-steering functionality for combining multiple steering vectors."""

from __future__ import annotations

import torch
from typing import Iterable
from pathlib import Path

from wisent.core.models.wisent_model import WisentModel
from wisent.core.models.core.atoms import SteeringPlan
from wisent.core.activations.core.atoms import RawActivationMap
from wisent.core.prompts.core.atom import ChatMessage
from wisent.core.utils.device import resolve_default_device
from wisent.core.models.inference_config import get_generate_kwargs

__all__ = ["MultiSteering", "MultiSteeringError"]


class MultiSteeringError(Exception):
    """Exception raised for multi-steering errors."""
    pass


class MultiSteering:
    """Handles multi-steering vector combination and application using WisentModel."""

    def __init__(self, device: str | None = None, method: str = "CAA"):
        """Initialize multi-steering handler.

        Args:
            device: Device to use for computations (cpu/cuda/mps)
            method: Steering method to use (kept for backward compatibility, not used in new API)
        """
        self.device = device or resolve_default_device()
        self.method = method  # Kept for backward compatibility
        self.loaded_vectors: list[dict] = []
        self.weights: list[float] = []
        self.combined_vector: torch.Tensor | None = None
        self.combined_scale: float = 1.0  # Track the total steering scale
        self.layer: int | None = None

    def load_vectors(self, vector_specs: list[str]) -> None:
        """Load and validate steering vectors from file paths.

        Args:
            vector_specs: List of "path:weight" specifications

        Raises:
            MultiSteeringError: If vectors cannot be loaded or are incompatible
        """
        if not vector_specs:
            raise MultiSteeringError("No vectors specified")

        self.loaded_vectors = []
        self.weights = []
        layers_found = set()

        for spec in vector_specs:
            parts = spec.split(":")
            if len(parts) != 2:
                raise MultiSteeringError(f"Invalid vector specification: {spec}. Expected format: path:weight")

            vector_path = parts[0]
            try:
                weight = float(parts[1])
            except ValueError:
                raise MultiSteeringError(f"Invalid weight in {spec}. Must be a number.")

            if not Path(vector_path).exists():
                raise MultiSteeringError(f"Vector file not found: {vector_path}")

            print(f"Loading vector from {vector_path} with weight {weight}")

            try:
                vector_data = torch.load(vector_path, map_location=self.device, weights_only=False)
            except Exception as e:
                raise MultiSteeringError(f"Failed to load vector from {vector_path}: {e}")

            # Extract metadata from loaded vector
            if isinstance(vector_data, dict):
                layer = vector_data.get("layer_index", None)
                steering_vector = vector_data.get("steering_vector", None)

                if steering_vector is None:
                    raise MultiSteeringError(f"No steering vector found in {vector_path}")

                if layer is not None:
                    layers_found.add(layer)

                self.loaded_vectors.append(vector_data)
                self.weights.append(weight)

                print(f"   ✓ Loaded vector from layer {layer}")
            else:
                raise MultiSteeringError(f"Invalid vector format in {vector_path}")

        # Validate compatibility
        if len(layers_found) > 1:
            raise MultiSteeringError(f"Vectors from different layers cannot be combined: {layers_found}")

        if not layers_found:
            raise MultiSteeringError("No layer information found in vectors")

        self.layer = list(layers_found)[0]

        print(f"\nUsing {self.method} method for vector combination")
        print(f"Target layer: {self.layer}")

    def combine_vectors(self, normalize: bool = True) -> torch.Tensor:
        """Combine loaded vectors using SteeringPlan.

        Args:
            normalize: Whether to normalize the combined vector

        Returns:
            Combined steering vector as tensor

        Raises:
            MultiSteeringError: If combination fails
        """
        if not self.loaded_vectors:
            raise MultiSteeringError("No vectors loaded")

        if self.layer is None:
            raise MultiSteeringError("No layer information available")

        print(f"\n🔄 Combining {len(self.loaded_vectors)} vectors")

        # Build RawActivationMap for each vector
        raw_maps: list[RawActivationMap] = []
        for vector_data in self.loaded_vectors:
            steering_vector = vector_data["steering_vector"]

            if not isinstance(steering_vector, torch.Tensor):
                steering_vector = torch.tensor(steering_vector, device=self.device)
            else:
                steering_vector = steering_vector.to(self.device)

            # Create a map with layer name as string (required by SteeringPlan)
            raw_map: RawActivationMap = {str(self.layer): steering_vector}
            raw_maps.append(raw_map)

        # Create SteeringPlan with weighted combination
        # Use sum of weights as scale to preserve steering strength
        # Weights are used as relative proportions, scale controls overall strength
        total_weight = sum(self.weights)
        plan = SteeringPlan.from_raw(
            raw=raw_maps,
            weights=self.weights,
            scale=total_weight,
            normalize=normalize
        )

        # Extract the combined vector from the plan
        layer_str = str(self.layer)
        if layer_str not in plan.layers:
            raise MultiSteeringError(f"Layer {self.layer} not found in steering plan")

        steering_vector = plan.layers[layer_str]
        self.combined_vector = steering_vector.vector
        self.combined_scale = total_weight  # Store the scale for apply_steering

        print(f"   ✓ Combined vector shape: {self.combined_vector.shape}")
        print(f"   ✓ Combined vector norm: {torch.norm(self.combined_vector).item():.4f}")
        print(f"   ✓ Steering scale: {self.combined_scale:.4f}")

        return self.combined_vector

    def apply_steering_stream(
        self,
        model: WisentModel,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = None,
        top_p: float = None,
        enable_thinking: bool = True,
        prompt_is_formatted: bool = False,
        ensure_varied_responses: bool = False,
        phrase_ledger = None
    ) -> Iterable[str]:
        """Apply the combined steering vector to generate text with streaming.

        Args:
            model: WisentModel instance to use for generation
            prompt: Input prompt (either raw text or pre-formatted with chat template)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (uses inference config if None)
            top_p: Top-p sampling parameter (uses inference config if None)
            enable_thinking: If False, disable thinking/reasoning mode (prevents <think> tags for supported models like Qwen)
            prompt_is_formatted: If True, prompt already has chat template applied

        Yields:
            Generated text chunks

        Raises:
            MultiSteeringError: If steering fails
        """
        if self.combined_vector is None:
            raise MultiSteeringError("No combined vector available. Call combine_vectors() first.")

        if self.layer is None:
            raise MultiSteeringError("No layer information available")

        # Get inference config settings, with optional overrides
        gen_kwargs = get_generate_kwargs(max_new_tokens=max_new_tokens)
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        print(f"\n🎯 Applying combined steering vector at layer {self.layer}")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        print(f"Prompt is formatted: {prompt_is_formatted}")
        print("=" * 50)

        # Create SteeringPlan from the combined vector with the stored scale
        raw_map: RawActivationMap = {str(self.layer): self.combined_vector}
        steering_plan = SteeringPlan.from_raw(
            raw=raw_map,
            scale=self.combined_scale,
            normalize=False  # Already normalized in combine_vectors
        )

        # Handle prompt formatting
        if prompt_is_formatted:
            # Prompt already has chat template applied - pass as string directly
            inputs = prompt
        else:
            # Format prompt as chat messages (current behavior)
            messages: list[ChatMessage] = [{"role": "user", "content": prompt}]
            inputs = [messages]

        try:
            # Use WisentModel's generate_stream with steering
            yield from model.generate_stream(
                inputs=inputs,
                **gen_kwargs,
                use_steering=True,
                steering_plan=steering_plan,
                skip_prompt=True,
                skip_special_tokens=True,
                enable_thinking=enable_thinking,
                prompt_is_formatted=prompt_is_formatted,
                ensure_varied_responses=ensure_varied_responses,
                phrase_ledger=phrase_ledger
            )

        except Exception as e:
            import traceback
            error_details = f"{type(e).__name__}: {e}\nTraceback: {traceback.format_exc()}"
            raise MultiSteeringError(f"Failed to apply steering: {error_details}") from e

    def apply_steering(
        self,
        model: WisentModel,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = None,
        top_p: float = None,
        enable_thinking: bool = True,
        prompt_is_formatted: bool = False
    ) -> str:
        """Apply the combined steering vector to generate text (non-streaming).

        Args:
            model: WisentModel instance to use for generation
            prompt: Input prompt (either raw text or pre-formatted with chat template)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (uses inference config if None)
            top_p: Top-p sampling parameter (uses inference config if None)
            enable_thinking: If False, disable thinking/reasoning mode (prevents <think> tags for supported models like Qwen)
            prompt_is_formatted: If True, prompt already has chat template applied

        Returns:
            Generated text

        Raises:
            MultiSteeringError: If steering fails
        """
        if self.combined_vector is None:
            raise MultiSteeringError("No combined vector available. Call combine_vectors() first.")

        if self.layer is None:
            raise MultiSteeringError("No layer information available")

        # Get inference config settings, with optional overrides
        gen_kwargs = get_generate_kwargs(max_new_tokens=max_new_tokens)
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        print(f"\n🎯 Applying combined steering vector at layer {self.layer}")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        print(f"Prompt is formatted: {prompt_is_formatted}")
        print("=" * 50)

        # Create SteeringPlan from the combined vector with the stored scale
        raw_map: RawActivationMap = {str(self.layer): self.combined_vector}
        steering_plan = SteeringPlan.from_raw(
            raw=raw_map,
            scale=self.combined_scale,
            normalize=False  # Already normalized in combine_vectors
        )

        # Handle prompt formatting
        if prompt_is_formatted:
            # Prompt already has chat template applied - pass as string directly
            inputs = prompt
        else:
            # Format prompt as chat messages (current behavior)
            messages: list[ChatMessage] = [{"role": "user", "content": prompt}]
            inputs = [messages]

        try:
            # Use WisentModel's generate with steering
            outputs = model.generate(
                inputs=inputs,
                **gen_kwargs,
                use_steering=True,
                steering_plan=steering_plan,
                enable_thinking=enable_thinking,
                prompt_is_formatted=prompt_is_formatted
            )

            return outputs[0] if outputs else ""

        except Exception as e:
            raise MultiSteeringError(f"Failed to apply steering: {e}")


def run_multi_steer(
    vector_specs: list[str],
    model_name: str,
    prompt: str,
    method: str = "CAA",
    layer: int | None = None,
    max_new_tokens: int = 100,
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
    # Initialize model
    if verbose:
        print(f"\n🚀 Loading model: {model_name}")

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

    # Apply steering (non-streaming) - will use inference config if temperature/top_p are None
    output = multi_steer.apply_steering(
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )

    return output
