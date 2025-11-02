"""
Coherence/quality evaluator for personalization steering.

Evaluates response quality, coherence, and freedom from repetition.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    import torch

__all__ = ["evaluate_quality"]


def evaluate_quality(
    response: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: "torch.device",
) -> float:
    """
    Evaluate response quality using heuristic checks on a scale of 1-100.

    Checks for common quality issues:
    - Empty or too short responses
    - Repetitive tokens (single token appearing too frequently)
    - Repeated phrases (same bigram appearing multiple times)
    - Nonsensical patterns (excessive special characters)
    - Character repetition (same character repeated many times)

    Args:
        response: The response to evaluate
        model: The model (not used, kept for API compatibility)
        tokenizer: The tokenizer (not used, kept for API compatibility)
        device: Device (not used, kept for API compatibility)

    Returns:
        Quality score between 1 and 100
        - 100 = Perfect quality (no issues detected)
        - 1 = Very poor quality (multiple severe issues)
    """
    score = 1.0  # Start with perfect score (0-1 scale)

    # Check 1: Empty or too short
    if len(response.strip()) < 10:
        score *= 0.1
        # Scale to 1-100 and return early
        return max(1.0, score * 99.0 + 1.0)

    tokens = response.lower().split()

    # Check 2: Repetitive tokens
    if len(tokens) > 0:
        token_counts = Counter(tokens)
        most_common_count = token_counts.most_common(1)[0][1]
        repetition_ratio = most_common_count / len(tokens)

        # Penalize if any token appears more than 30% of the time
        if repetition_ratio > 0.3:
            score *= 1.0 - (repetition_ratio - 0.3)

    # Check 3: Repeated n-grams (phrases)
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    if bigrams:
        bigram_counts = Counter(bigrams)
        most_common_bigram_count = bigram_counts.most_common(1)[0][1]
        if most_common_bigram_count > 2:  # Same phrase repeated 3+ times
            score *= 0.5

    # Check 4: Nonsensical patterns (too many special chars)
    special_char_ratio = len(re.findall(r"[^a-zA-Z0-9\s.,!?']", response)) / max(
        len(response), 1
    )
    if special_char_ratio > 0.2:
        score *= 0.7

    # Check 5: Single repeated character
    if re.search(r"(.)\1{10,}", response):  # Same char 10+ times in a row
        score *= 0.3

    # Ensure score is non-negative
    score = max(0.0, score)

    # Scale from 0-1 to 1-100
    quality_score = score * 99.0 + 1.0

    return quality_score
