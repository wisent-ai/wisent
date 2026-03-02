"""
Component-targeted steering hook registration.

Enables steering vectors to be injected at specific transformer components
(attention output, MLP output, per-head, etc.) instead of only the residual
stream.  Reuses the existing ExtractionComponent infrastructure for hook
point resolution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from wisent.core.primitives.model_interface.core.activations.strategies.extraction_strategy import (
    ExtractionComponent,
)
from wisent.core.primitives.model_interface.core.activations.component_hooks import (
    _COMPONENT_MAP,
    _PRE_HOOK_COMPONENTS,
    _GLOBAL_COMPONENTS,
    _detect_model_type,
    _get_submodule,
)
from wisent.core.reading.modules.analysis.structure.transformer_analysis import (
    get_component_hook_points,
)

if TYPE_CHECKING:
    from typing import Callable, Optional
    from wisent.core.control.steering_methods.steering_object import BaseSteeringObject
    from wisent.core.primitives.models.core.atoms import HookHandleGroup


def _resolve_extraction_component(value: str) -> ExtractionComponent:
    """Convert a string value to an ExtractionComponent enum member."""
    for member in ExtractionComponent:
        if member.value == value:
            return member
    raise ValueError(
        f"Unknown extraction component: {value}. "
        f"Valid: {ExtractionComponent.list_all()}"
    )


def register_component_steering_hooks(
    model: nn.Module,
    steering_obj: BaseSteeringObject,
    hook_group: HookHandleGroup,
    base_strength: float,
    strategy_fn: Callable[[int], float],
) -> None:
    """
    Register forward/pre-hooks to steer at a specific transformer component.

    Args:
        model: The HuggingFace model (e.g. ``wm.hf_model``).
        steering_obj: Steering object whose ``metadata.extraction_component``
            determines where to inject.
        hook_group: Handle group that tracks removable hooks.
        base_strength: Base multiplier passed to ``apply_steering``.
        strategy_fn: Callable(token_pos) -> weight.  Encapsulates the
            steering strategy (constant, diminishing, etc.).
    """
    component_str = steering_obj.metadata.extraction_component
    component = _resolve_extraction_component(component_str)

    if component == ExtractionComponent.RESIDUAL_STREAM:
        raise ValueError(
            "register_component_steering_hooks should not be called "
            "for RESIDUAL_STREAM; use the default layer-output path."
        )

    tc = _COMPONENT_MAP.get(component)
    if tc is None:
        raise ValueError(
            f"No TransformerComponent mapping for {component.value}"
        )

    use_pre_hook = component in _PRE_HOOK_COMPONENTS
    is_global = component in _GLOBAL_COMPONENTS
    is_attn_scores = component == ExtractionComponent.ATTENTION_SCORES
    model_type = _detect_model_type(model)
    token_counter = {"prompt_len": 0}

    if is_global:
        # Global components: single hook, use first layer key from steering obj
        layer_idx = steering_obj.metadata.layers[0] if steering_obj.metadata.layers else 0
        hook_points = get_component_hook_points(model_type, layer_idx, tc)
        seen = set()
        for point_name in hook_points:
            if point_name in seen:
                continue
            seen.add(point_name)
            submodule = _get_submodule(model, point_name)
            handle = submodule.register_forward_hook(
                _forward_hook_factory(
                    steering_obj, layer_idx, base_strength,
                    token_counter, strategy_fn,
                )
            )
            hook_group.add(handle)
        return

    for layer_idx in steering_obj.metadata.layers:
        hook_points = get_component_hook_points(model_type, layer_idx, tc)
        for point_name in hook_points:
            submodule = _get_submodule(model, point_name)
            if is_attn_scores:
                num_heads = _get_num_heads(model)
                handle = submodule.register_forward_hook(
                    _attn_score_hook_factory(
                        steering_obj, layer_idx, base_strength,
                        token_counter, strategy_fn, num_heads,
                    )
                )
            elif use_pre_hook:
                handle = submodule.register_forward_pre_hook(
                    _pre_hook_factory(
                        steering_obj, layer_idx, base_strength,
                        token_counter, strategy_fn,
                    )
                )
            else:
                handle = submodule.register_forward_hook(
                    _forward_hook_factory(
                        steering_obj, layer_idx, base_strength,
                        token_counter, strategy_fn,
                    )
                )
            hook_group.add(handle)


def _get_num_heads(model: nn.Module) -> int:
    """Get the number of attention heads from model config."""
    config = getattr(model, "config", None)
    if config is not None:
        num_heads = getattr(config, "num_attention_heads", None)
        if num_heads is not None:
            return num_heads
    raise ValueError("model.config missing num_attention_heads")


def _forward_hook_factory(
    obj: BaseSteeringObject,
    layer: int,
    strength: float,
    counter: dict,
    strategy_fn: Callable[[int], float],
):
    """Create a forward hook that adds the steering delta to output."""
    def _hook(_mod, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        seq_len = hs.shape[1] if hs.dim() >= 2 else 1

        if counter["prompt_len"] == 0:
            counter["prompt_len"] = seq_len
        token_pos = max(seq_len - counter["prompt_len"], 0)

        weight = strategy_fn(token_pos)
        effective = strength * weight
        if effective == 0:
            return out

        if isinstance(out, tuple):
            steered = obj.apply_steering(
                hs, layer=layer, base_strength=effective,
            )
            return (steered,) + out[1:]
        return obj.apply_steering(out, layer=layer, base_strength=effective)

    return _hook


def _pre_hook_factory(
    obj: BaseSteeringObject,
    layer: int,
    strength: float,
    counter: dict,
    strategy_fn: Callable[[int], float],
):
    """Create a forward pre-hook that adds the steering delta to input."""
    def _hook(_mod, inp):
        if isinstance(inp, tuple) and len(inp) > 0:
            hs = inp[0]
        else:
            hs = inp

        seq_len = hs.shape[1] if hs.dim() >= 2 else 1

        if counter["prompt_len"] == 0:
            counter["prompt_len"] = seq_len
        token_pos = max(seq_len - counter["prompt_len"], 0)

        weight = strategy_fn(token_pos)
        effective = strength * weight
        if effective == 0:
            return inp

        steered = obj.apply_steering(
            hs, layer=layer, base_strength=effective,
        )

        if isinstance(inp, tuple):
            return (steered,) + inp[1:]
        return steered

    return _hook


def _attn_score_hook_factory(
    obj: BaseSteeringObject,
    layer: int,
    strength: float,
    counter: dict,
    strategy_fn: Callable[[int], float],
    num_heads: int,
):
    """
    Create a forward hook on q_proj that scales Q per-head.

    The steering vector has shape [num_heads].  Each value ``s_h`` scales
    query vectors for head *h* by ``(1 + effective_strength * s_h)``,
    which multiplies the corresponding attention logits by the same factor
    (since scores = Q K^T / sqrt(d)).
    """
    def _hook(_mod, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        seq_len = hs.shape[1] if hs.dim() >= 2 else 1

        if counter["prompt_len"] == 0:
            counter["prompt_len"] = seq_len
        token_pos = max(seq_len - counter["prompt_len"], 0)

        weight = strategy_fn(token_pos)
        effective = strength * weight
        if effective == 0:
            return out

        vec = obj.get_steering_vector(layer).to(hs.device, hs.dtype)
        # vec shape: [num_heads] — per-head scaling factors
        head_dim = hs.shape[-1] // num_heads
        # Reshape Q: [batch, seq, hidden] -> [batch, seq, num_heads, head_dim]
        q = hs.view(*hs.shape[:-1], num_heads, head_dim)
        # scale shape: [num_heads, 1] for broadcast over head_dim
        scale = (1.0 + effective * vec[:num_heads]).unsqueeze(-1)
        q = q * scale
        result = q.reshape(hs.shape)

        if isinstance(out, tuple):
            return (result,) + out[1:]
        return result

    return _hook
