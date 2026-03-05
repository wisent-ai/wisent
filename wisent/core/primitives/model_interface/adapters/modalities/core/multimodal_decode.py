"""Multimodal adapter: decode and greedy decode methods."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from wisent.core.primitives.models.config import get_generate_kwargs

__all__ = ["decode_llava", "decode_qwen_vl", "decode_idefics", "decode_generic", "greedy_decode"]


def decode_llava(model: nn.Module, latent: torch.Tensor, processor: Any, *, display_token_limit: int) -> str:
    """Decode for LLaVA models."""
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "lm_head"):
            logits = lm.lm_head(latent)
            return greedy_decode(logits, processor, display_token_limit=display_token_limit)
    return decode_generic(model, latent, processor, display_token_limit=display_token_limit)


def decode_qwen_vl(model: nn.Module, latent: torch.Tensor, processor: Any, *, display_token_limit: int) -> str:
    """Decode for Qwen-VL models."""
    if hasattr(model, "lm_head"):
        logits = model.lm_head(latent)
        return greedy_decode(logits, processor, display_token_limit=display_token_limit)
    return decode_generic(model, latent, processor, display_token_limit=display_token_limit)


def decode_idefics(model: nn.Module, latent: torch.Tensor, processor: Any, *, display_token_limit: int) -> str:
    """Decode for IDEFICS models."""
    if hasattr(model, "embed_out"):
        logits = model.embed_out(latent)
        return greedy_decode(logits, processor, display_token_limit=display_token_limit)
    elif hasattr(model, "lm_head"):
        logits = model.lm_head(latent)
        return greedy_decode(logits, processor, display_token_limit=display_token_limit)
    return decode_generic(model, latent, processor, display_token_limit=display_token_limit)


def decode_generic(model: nn.Module, latent: torch.Tensor, processor: Any, *, display_token_limit: int) -> str:
    """Generic decoding fallback."""
    lm_head = None
    for attr in ["lm_head", "embed_out", "output_projection", "head"]:
        if hasattr(model, attr):
            lm_head = getattr(model, attr)
            break
        if hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, attr):
                lm_head = getattr(lm, attr)
                break
    if lm_head is not None:
        logits = lm_head(latent)
        return greedy_decode(logits, processor, display_token_limit=display_token_limit)
    return f"[Latent decoded: shape={latent.shape}, mean={latent.mean().item():.4f}]"


def greedy_decode(logits: torch.Tensor, processor: Any, *, display_token_limit: int, max_length: int | None = None) -> str:
    """Perform greedy decoding from logits."""
    if logits.dim() == 3:
        next_token_logits = logits[:, -1, :]
    else:
        next_token_logits = logits
    predicted_ids = torch.argmax(next_token_logits, dim=-1)
    if predicted_ids.dim() == 0:
        predicted_ids = predicted_ids.unsqueeze(0)
    if predicted_ids.dim() == 1:
        predicted_ids = predicted_ids.unsqueeze(0)
    try:
        text = processor.decode(predicted_ids[0], skip_special_tokens=True)
    except Exception:
        if hasattr(processor, "tokenizer"):
            text = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        else:
            text = f"[Token IDs: {predicted_ids[0].tolist()[:display_token_limit]}...]"
    return text
