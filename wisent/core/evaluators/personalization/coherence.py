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


def _is_gibberish(text: str) -> bool:
    """
    Detect if text is gibberish/nonsensical.

    Checks for:
    - Insufficient spacing (words concatenated together)
    - Too many long tokens (concatenated words)
    - CamelCase patterns indicating concatenated words
    - Repeated word fragments within tokens
    - Too few valid English words
    """
    if not text or len(text.strip()) < 10:
        return False  # Too short to evaluate, let other checks handle

    # Check 1: Spacing ratio - normal English has ~15-20% spaces
    space_ratio = text.count(' ') / len(text)
    if len(text) > 50 and space_ratio < 0.08:
        return True

    tokens = text.split()
    if not tokens:
        return False

    # Check 2: Long tokens (concatenated words)
    long_tokens = sum(1 for t in tokens if len(t) > 25)
    if long_tokens / len(tokens) > 0.1:
        return True

    # Check 3: CamelCase patterns (e.g., "hisHandsThatDelight", "HewalksAway")
    # This catches concatenated words even if spacing ratio is ok
    camel_pattern = re.compile(r'[a-z]{2,}[A-Z][a-z]{2,}')
    camel_count = sum(1 for t in tokens if camel_pattern.search(t))
    if camel_count >= 2:  # Multiple camelCase tokens is suspicious
        return True

    # Check 4: Repeated fragments within tokens (e.g., "thethethe", "forforfor")
    if re.search(r'(\w{2,6})\1{2,}', text.lower()):
        return True

    # Check 5: Word validity - check if tokens look like real words
    # Common English words for quick validation
    common_words = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "is", "are", "was", "were", "been", "has", "had", "does", "did", "here",
        "where", "why", "how", "very", "more", "some", "any", "also", "than", "then",
    }

    if len(tokens) >= 5:
        # Check what fraction of tokens are recognizable
        valid_count = 0
        for token in tokens:
            clean_token = re.sub(r'[^a-zA-Z]', '', token.lower())
            if not clean_token:
                continue
            # Token is valid if it's a common word OR has vowels and reasonable length
            if clean_token in common_words:
                valid_count += 1
            elif len(clean_token) <= 15 and re.search(r'[aeiou]', clean_token):
                valid_count += 1

        validity_ratio = valid_count / len(tokens)
        if validity_ratio < 0.3:
            return True

    return False


def evaluate_quality(
    response: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: "torch.device",
) -> float:
    """
    Evaluate response quality using heuristic checks on a scale of 1-100.

    Checks for common quality issues:
    - Gibberish/nonsensical text (immediate zero)
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
        - 0 = Gibberish detected
    """
    # Check for gibberish first - immediate zero if detected
    if _is_gibberish(response):
        return 0.0

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
