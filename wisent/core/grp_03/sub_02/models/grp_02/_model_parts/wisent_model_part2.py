"""WisentModel methods: apply_steering_object, _encode_one, _batch_encode, _extract_assistant_response."""
from __future__ import annotations

import math
import re
from typing import Any

import torch
import torch.nn as nn

from wisent.core.models.layer import extract_token_ids
from wisent.core.models.core.atoms import SteeringVector
from wisent.core.prompts.core.atom import ChatMessage
from wisent.core.errors import ChatTemplateNotAvailableError


def _apply_steering_object(
    self,
    steering_obj: "BaseSteeringObject",
    base_strength: float = 1.0,
    steering_strategy: str = "constant",
    steering_strategy_config: dict | None = None,
    max_new_tokens: int = 128,
) -> None:
    """
    Register forward hooks using a SteeringObject with full method-specific logic.

    Unlike apply_steering() which uses simple vector addition, this method
    uses the SteeringObject's compute_gate() and compute_intensity() methods
    for conditional steering (TETNO, GROM).

    Args:
        steering_obj: A SteeringObject (CAA, Ostrze, MLP, TECZA, TETNO, or GROM)
        base_strength: Base multiplier for steering intensity
        steering_strategy: How to apply steering over tokens:
            - "constant": Same strength throughout generation
            - "initial_only": Only apply at first N tokens
            - "diminishing": Strength decreases over tokens
            - "increasing": Strength increases over tokens
            - "gaussian": Gaussian curve centered at specific token position
        steering_strategy_config: Config for strategy (initial_tokens, rate, gaussian_center, gaussian_width)
        max_new_tokens: Expected max tokens for strategy weight calculation

    Example:
        >>> from wisent.core.steering_methods.steering_object import BaseSteeringObject
        >>> obj = BaseSteeringObject.load("steering.pt")
        >>> wm.apply_steering_object(obj, base_strength=1.0, steering_strategy="diminishing")
        >>> response = wm.generate([...])
        >>> wm.detach()
    """
    from wisent.core.steering_methods.steering_object import BaseSteeringObject

    self.detach()
    if hasattr(steering_obj, 'set_model_weights'):
        steering_obj.set_model_weights(self.hf_model)
    config = steering_strategy_config or {}

    # Token counter shared across all hooks (mutable container)
    token_counter = {"count": 0, "prompt_len": 0}

    def get_steering_weight(strategy: str, token_pos: int, total_tokens: int) -> float:
        """Calculate steering weight based on strategy and token position."""
        position_frac = token_pos / max(total_tokens, 1)

        if strategy == "constant":
            return 1.0
        elif strategy == "initial_only":
            initial_tokens = config.get("initial_tokens", 10)
            return 1.0 if token_pos < initial_tokens else 0.0
        elif strategy == "diminishing":
            rate = config.get("rate", 0.1)
            return math.exp(-rate * token_pos)
        elif strategy == "increasing":
            rate = config.get("rate", 0.1)
            return 1.0 - math.exp(-rate * token_pos)
        elif strategy == "gaussian":
            center = config.get("gaussian_center", 0.5)
            width = config.get("gaussian_width", 0.2)
            return math.exp(-((position_frac - center) ** 2) / (2 * width ** 2))
        else:
            return 1.0

    # Component-targeted steering: dispatch to submodule hooks
    component = getattr(steering_obj.metadata, "extraction_component", "residual_stream")
    if component != "residual_stream":
        from wisent.core.models.component_steering import register_component_steering_hooks
        def _strategy_fn(token_pos: int) -> float:
            return get_steering_weight(steering_strategy, token_pos, max_new_tokens)
        register_component_steering_hooks(
            self.hf_model, steering_obj, self._hook_group,
            base_strength, _strategy_fn,
        )
        return

    name_to_index = {str(i + 1): i for i in range(len(self._layers))}

    for layer_idx in steering_obj.metadata.layers:
        layer_name = str(layer_idx)
        if layer_name not in name_to_index:
            continue

        idx = name_to_index[layer_name]
        layer_module = self._layers[idx]

        def _hook_factory(obj: BaseSteeringObject, layer: int, strength: float, counter: dict, strategy: str, max_tokens: int):
            def _hook(_mod: nn.Module, _inp: tuple, out: torch.Tensor | tuple) -> torch.Tensor | tuple:
                # Get current sequence length to detect new tokens
                if isinstance(out, tuple):
                    hs = out[0]
                else:
                    hs = out

                seq_len = hs.shape[1]

                # On first call, record prompt length
                if counter["prompt_len"] == 0:
                    counter["prompt_len"] = seq_len

                # Calculate token position (0 = first generated token)
                token_pos = seq_len - counter["prompt_len"]
                if token_pos < 0:
                    token_pos = 0

                # Get weight for this token position
                weight = get_steering_weight(strategy, token_pos, max_tokens)
                effective_strength = strength * weight

                # Skip steering if weight is 0
                if effective_strength == 0:
                    return out

                if isinstance(out, tuple):
                    steered = obj.apply_steering(hs, layer=layer, base_strength=effective_strength)
                    steered = steered.to(dtype=hs.dtype)  # enforce dtype match
                    return (steered,) + out[1:]
                else:
                    steered = obj.apply_steering(out, layer=layer, base_strength=effective_strength)
                    return steered.to(dtype=out.dtype)  # enforce dtype match
            return _hook

        handle = layer_module.register_forward_hook(
            _hook_factory(steering_obj, layer_idx, base_strength, token_counter, steering_strategy, max_new_tokens)
        )
        self._hook_group.add(handle)


def _encode_one(
    self,
    message: list[ChatMessage],
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Encode a single input in chat format.

    arguments:
        messages:
            list of {'role': str, 'content': str} dicts (chat messages).
        add_generation_prompt:
            If True, append the model's generation prompt at the end.
        enable_thinking:
            If False, disable thinking/reasoning mode.

    returns:
        dict with 'input_ids' and 'attention_mask' tensors.
    """
    try:
        try:
            result = self.tokenizer.apply_chat_template(
                message, tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
                return_tensors="pt",
            )
        except TypeError:
            result = self.tokenizer.apply_chat_template(
                message, tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_tensors="pt",
            )
        ids = extract_token_ids(result)
    except ValueError as e:
        raise ChatTemplateNotAvailableError(cause=e)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
    }


def _batch_encode(
    self,
    inputs: list[list[ChatMessage]],
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Batch-encode a list of chat messages.

    arguments:
        inputs:
            list of chat messages (each a list of {'role','content'} dicts).
        add_generation_prompt:
            If True, append the model's generation prompt at the end of each.
        enable_thinking:
            If False, disable thinking/reasoning mode.

    returns:
        dict with batched 'input_ids' and 'attention_mask' tensors.
    """
    singles = []
    for item in inputs:
        singles.append(self._encode_one(item, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking))

    batch = self.tokenizer.pad(singles, padding=True, return_tensors="pt")

    batch = {k: v.to(self.device) for k, v in batch.items()}

    return batch


def _extract_assistant_response(self, text: str) -> str:
    """
    Extract only the assistant's response from a decoded chat template output.

    arguments:
        text:
            Full decoded text from tokenizer.batch_decode

    returns:
        Extracted assistant response text
    """
    # Look for the assistant marker in the decoded text
    if "assistant" in text:
        response = text.split("assistant", 1)[1]
        response = response.strip()
    else:
        response = text

    # Remove empty thinking blocks that Qwen adds when enable_thinking=False
    response = re.sub(r'^<think>\s*</think>\s*', '', response)

    # Also remove thinking blocks with content (full thinking output)
    response = re.sub(r'^<think>.*?</think>\s*', '', response, flags=re.DOTALL)

    return response.strip()
