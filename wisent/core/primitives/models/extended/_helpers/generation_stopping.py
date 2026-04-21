"""Stopping criteria for detecting degenerate generation (e.g. repetition loops, gibberish)."""
from __future__ import annotations

import logging

import torch
from transformers import PreTrainedTokenizer, StoppingCriteria, StoppingCriteriaList

from wisent.core.reading.evaluators.core.text_quality import (
    DEFAULT_NONSENSE_MIN_TOKENS,
    _is_gibberish,
)
from wisent.core.utils.config_tools.constants import (
    AXIS_COLS,
    AXIS_ROWS,
    COMBO_OFFSET,
    DEGEN_DIVERSITY_THRESHOLD,
    DEGEN_MAX_REPEATS,
    DEGEN_NGRAM_SIZE,
    DEGEN_WARMUP_TOKENS,
    DEGEN_WINDOW_SIZE,
)

# Gibberish check is more expensive than n-gram repetition (requires decoding),
# so we run it less often and only after a larger warmup to avoid false trips
# on short natural-language prefixes.
GIBBERISH_CHECK_WARMUP_TOKENS = 32
GIBBERISH_CHECK_INTERVAL = 16

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
        diversity_threshold: float = DEGEN_DIVERSITY_THRESHOLD,
    ) -> None:
        self.prompt_length = prompt_length
        self.warmup_tokens = warmup_tokens
        self.ngram_size = ngram_size
        self.max_repeats = max_repeats
        self.window_size = window_size
        self.diversity_threshold = diversity_threshold
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
            reason = ""
            for i in range(len(tail) - self.ngram_size + COMBO_OFFSET):
                ngram = tuple(tail[i : i + self.ngram_size])
                ngram_counts[ngram] = ngram_counts.get(ngram, AXIS_ROWS) + COMBO_OFFSET
                if ngram_counts[ngram] >= self.max_repeats:
                    degenerate = True
                    reason = (
                        f"repeated {self.ngram_size}-gram found "
                        f"{self.max_repeats}+ times in last {self.window_size} tokens"
                    )
                    break

            if not degenerate:
                unique_tokens = len(set(tail))
                diversity_ratio = unique_tokens / len(tail)
                if diversity_ratio < self.diversity_threshold:
                    degenerate = True
                    reason = (
                        f"low token diversity {diversity_ratio:.2f} "
                        f"({unique_tokens}/{len(tail)} unique) "
                        f"< threshold {self.diversity_threshold}"
                    )

            if degenerate:
                should_stop[b] = True
                self.stopped.add(b)
                logger.warning(
                    "[degeneration] sequence %d stopped at %d generated tokens (%s)",
                    b, generated_length, reason,
                )

        return should_stop


class GibberishStoppingCriteria(StoppingCriteria):
    """Stop generation when the decoded tail is gibberish.

    Complements `RepetitionStoppingCriteria`: repetition works on token-id
    n-grams and catches "the the the" style loops, while this criterion
    decodes the generated text and runs the same `_is_gibberish` heuristics
    used by the evaluator-side coherence gate. This catches failures the
    n-gram check misses — CamelCase concatenation, non-English character
    salads, tokenizer-fragmentation nonsense, low function-word ratio.

    Decoding is not free, so we wait until *warmup_tokens* have been
    produced and only re-check every *check_interval* new tokens. Each
    sequence stops independently; once stopped it stays stopped.
    """

    def __init__(
        self,
        prompt_length: int,
        tokenizer: "PreTrainedTokenizer",
        warmup_tokens: int = GIBBERISH_CHECK_WARMUP_TOKENS,
        check_interval: int = GIBBERISH_CHECK_INTERVAL,
        nonsense_min_tokens: int = DEFAULT_NONSENSE_MIN_TOKENS,
    ) -> None:
        self.prompt_length = prompt_length
        self.tokenizer = tokenizer
        self.warmup_tokens = warmup_tokens
        self.check_interval = check_interval
        self.nonsense_min_tokens = nonsense_min_tokens
        self.stopped: set[int] = set()
        self._last_check_at: int = 0

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> torch.BoolTensor:
        batch_size = input_ids.shape[AXIS_ROWS]
        generated_length = input_ids.shape[AXIS_COLS] - self.prompt_length
        should_stop = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        # Carry forward sequences already marked stopped.
        for b in self.stopped:
            if b < batch_size:
                should_stop[b] = True

        if generated_length < self.warmup_tokens:
            return should_stop
        if generated_length - self._last_check_at < self.check_interval:
            return should_stop
        self._last_check_at = generated_length

        for b in range(batch_size):
            if b in self.stopped:
                continue
            generated_ids = input_ids[b, self.prompt_length:].tolist()
            decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            if _is_gibberish(decoded, nonsense_min_tokens=self.nonsense_min_tokens):
                should_stop[b] = True
                self.stopped.add(b)
                logger.warning(
                    "[gibberish] sequence %d stopped at %d generated tokens "
                    "(decoded prefix: %r)",
                    b, generated_length, decoded[:80],
                )

        return should_stop


def build_degeneration_stopping_criteria(
    prompt_length: int,
    tokenizer: "PreTrainedTokenizer",
) -> StoppingCriteriaList:
    """Build a StoppingCriteriaList with repetition + gibberish detection.

    Repetition catches token-id n-gram loops; gibberish decodes the tail and
    runs the text_quality heuristics (CamelCase salads, low function-word
    ratio, tokenizer-fragmentation nonsense words).
    """
    return StoppingCriteriaList([
        RepetitionStoppingCriteria(prompt_length=prompt_length),
        GibberishStoppingCriteria(prompt_length=prompt_length, tokenizer=tokenizer),
    ])
