"""Stopping criteria for detecting degenerate generation (e.g. repetition loops)."""
from __future__ import annotations

import logging

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from wisent.core.utils.config_tools.constants import (
    AXIS_COLS,
    AXIS_ROWS,
    COMBO_OFFSET,
    DEGEN_MAX_REPEATS,
    DEGEN_NGRAM_SIZE,
    DEGEN_WARMUP_TOKENS,
    DEGEN_WINDOW_SIZE,
)

logger = logging.getLogger(__name__)


class RepetitionStoppingCriteria(StoppingCriteria):
    """Stop generation when n-gram repetition indicates degeneration.

    After *warmup_tokens* new tokens have been generated, checks the last
    *window_size* tokens for any *ngram_size*-gram that appears
    *max_repeats* or more times.  Returns a per-sequence BoolTensor so
    only degenerate sequences stop; healthy ones continue.
    """

    def __init__(
        self,
        prompt_length: int,
        warmup_tokens: int = DEGEN_WARMUP_TOKENS,
        ngram_size: int = DEGEN_NGRAM_SIZE,
        max_repeats: int = DEGEN_MAX_REPEATS,
        window_size: int = DEGEN_WINDOW_SIZE,
    ) -> None:
        self.prompt_length = prompt_length
        self.warmup_tokens = warmup_tokens
        self.ngram_size = ngram_size
        self.max_repeats = max_repeats
        self.window_size = window_size
        self.stopped: set[int] = set()

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> torch.BoolTensor:
        batch_size = input_ids.shape[AXIS_ROWS]
        generated_length = input_ids.shape[AXIS_COLS] - self.prompt_length

        if generated_length < self.warmup_tokens:
            return torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        should_stop = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        for b in range(batch_size):
            if b in self.stopped:
                should_stop[b] = True
                continue

            tail = input_ids[b, -self.window_size:].tolist()
            if len(tail) < self.ngram_size:
                continue

            ngram_counts: dict[tuple, int] = {}
            degenerate = False
            for i in range(len(tail) - self.ngram_size + COMBO_OFFSET):
                ngram = tuple(tail[i : i + self.ngram_size])
                ngram_counts[ngram] = ngram_counts.get(ngram, AXIS_ROWS) + COMBO_OFFSET
                if ngram_counts[ngram] >= self.max_repeats:
                    degenerate = True
                    break

            if degenerate:
                should_stop[b] = True
                self.stopped.add(b)
                logger.warning(
                    "[degeneration] sequence %d stopped at %d generated tokens "
                    "(repeated %d-gram found %d+ times in last %d tokens)",
                    b, generated_length, self.ngram_size,
                    self.max_repeats, self.window_size,
                )

        return should_stop


def build_repetition_stopping_criteria(prompt_length: int) -> StoppingCriteriaList:
    """Build a StoppingCriteriaList with repetition degeneration detection."""
    return StoppingCriteriaList([
        RepetitionStoppingCriteria(prompt_length=prompt_length),
    ])
