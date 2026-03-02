"""Helper classes for model core atoms: HookHandleGroup, TopLogits, GenerationStats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

__all__ = [
    "HookHandleGroup",
    "TopLogits",
    "GenerationStats",
]


class HookHandleGroup:
    """
    Manage a set of torch hooks to ensure clean detach.
    """
    def __init__(self) -> None:
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def add(self, handle: torch.utils.hooks.RemovableHandle) -> None:
        self._handles.append(handle)

    def remove_all(self) -> None:
        while self._handles:
            h = self._handles.pop()
            try:
                h.remove()
            except Exception:
                pass


@dataclass(slots=True)
class TopLogits:
    """
    Info for a generated step.

    attributes:
        token_id:
            chosen token id at this step.
        logit:
            raw logit for that token.
        prob:
            softmax probability for that token.
        topk_ids/topk_probs:
            optional top-k for analysis/visualization.
    """
    token_id: int
    logit: float
    prob: float
    topk_ids: list[int] | None = None
    topk_probs: list[float] | None = None


@dataclass(slots=True)
class GenerationStats:
    """
    Per-sequence stats for a generation call.

    attributes:
        tokens:
            the generated token ids (excluding the prompt).
        per_step:
            optional list of TopLogits, one per generated step.
    """
    tokens: list[int]
    per_step: list[TopLogits] | None = None
