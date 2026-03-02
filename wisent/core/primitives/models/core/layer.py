"""Layer representation and tokenizer utilities for model extraction."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Layer:
    """Represents a model layer for activation extraction."""

    index: int
    type: str = "transformer"
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = f"layer_{self.index}"


def extract_token_ids(result) -> torch.Tensor:
    """Extract a 1-D LongTensor of token IDs from apply_chat_template.

    Handles: torch.Tensor, BatchEncoding/dict, tokenizers.Encoding,
    and plain list[int] return types across transformers versions.
    """
    if isinstance(result, torch.Tensor):
        ids = result[0] if result.dim() > 1 else result
        return ids.long()
    if hasattr(result, "input_ids") or (
        isinstance(result, dict) and "input_ids" in result
    ):
        raw_ids = result["input_ids"]
        if isinstance(raw_ids, torch.Tensor):
            return (raw_ids[0] if raw_ids.dim() > 1 else raw_ids).long()
        if isinstance(raw_ids, list):
            inner = raw_ids[0] if raw_ids else raw_ids
            if isinstance(inner, torch.Tensor):
                return inner.long()
            if isinstance(inner, list):
                return torch.tensor(inner, dtype=torch.long)
            if hasattr(inner, "ids"):
                return torch.tensor(inner.ids, dtype=torch.long)
            return torch.tensor(list(inner), dtype=torch.long)
        if hasattr(raw_ids, "ids"):
            return torch.tensor(raw_ids.ids, dtype=torch.long)
        return torch.tensor(list(raw_ids), dtype=torch.long)
    if hasattr(result, "ids"):
        return torch.tensor(result.ids, dtype=torch.long)
    if isinstance(result, list):
        return torch.tensor(result, dtype=torch.long)
    return torch.tensor(list(result), dtype=torch.long)
