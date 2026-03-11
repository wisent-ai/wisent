"""KV cache activation collection for ExtractionComponent.KV_CACHE.

Extracts activations from past_key_values instead of hidden states.
GQA-safe: uses actual cache shape (num_kv_heads * head_dim), not hidden_dim.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from wisent.core.primitives.model_interface.core.activations.core.atoms import (
    LayerActivations,
    RawActivationMap,
)
from wisent.core.primitives.model_interface.core.activations import extract_activation
from wisent.core.utils.config_tools.constants import (
    INDEX_FIRST, INDEX_SECOND, INDEX_THIRD,
)

if TYPE_CHECKING:
    from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy


def collect_kv_cache(
    model_hf,
    full_enc: dict,
    keep: list[int],
    names_by_idx: list[str],
    strategy: "ExtractionStrategy",
    answer_text: str,
    tokenizer,
    prompt_len: int,
    store_device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> LayerActivations:
    """Collect activations from KV cache (past_key_values).

    Does a single forward pass with use_cache=True, output_hidden_states=False.
    For each layer in *keep*, extracts key_cache, permutes to
    [seq_len, num_kv_heads * head_dim], then applies the token-position
    extraction strategy.

    GQA is handled automatically because the cache shape already reflects
    the actual num_kv_heads (which may be < num_attention_heads).
    """
    out = model_hf(**full_enc, output_hidden_states=False, use_cache=True)
    past_kv = out.past_key_values
    collected: RawActivationMap = {}
    for idx in keep:
        name = names_by_idx[idx]
        # past_kv[layer] = (key_cache, value_cache)
        # key_cache shape: [batch, num_kv_heads, seq_len, head_dim]
        key_cache = past_kv[idx][INDEX_FIRST]
        # Squeeze batch dim, giving [num_kv_heads, seq_len, head_dim]
        kc = key_cache.squeeze(INDEX_FIRST)
        # Permute to [seq_len, num_kv_heads, head_dim]
        kc = kc.permute(INDEX_SECOND, INDEX_FIRST, INDEX_THIRD).contiguous()
        # Flatten to [seq_len, num_kv_heads * head_dim]
        seq_len = kc.shape[INDEX_FIRST]
        kc = kc.view(seq_len, -INDEX_SECOND)
        value = extract_activation(strategy, kc, answer_text, tokenizer, prompt_len)
        value = value.to(store_device)
        if dtype is not None:
            value = value.to(dtype)
        collected[name] = value
    return LayerActivations(collected)
