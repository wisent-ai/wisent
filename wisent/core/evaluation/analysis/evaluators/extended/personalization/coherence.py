"""
Coherence/quality evaluator for personalization steering.

Evaluates response quality, coherence, and freedom from repetition.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

from wisent.core.constants import (
    COHERENCE_FUNCTION_WORD_THRESHOLD, COHERENCE_SPACE_RATIO_MIN,
    COHERENCE_LONG_TOKEN_LEN, COHERENCE_LONG_TOKEN_RATIO,
    COHERENCE_VALIDITY_RATIO_MIN, COHERENCE_QUALITY_MIN_TOKENS,
    COHERENCE_RATIO_THRESHOLD,
    MIN_TOKENS_TRIGRAM, NONSENSE_MIN_TOKENS,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    import torch

# Global tokenizer cache
_tokenizer_cache = {}

# Function words - the glue words of English that appear in natural text
# Real sentences need these; gibberish often lacks them
FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "only", "own", "same", "than", "too", "very", "just", "also",
    "now", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "few", "more", "most", "other", "some", "such", "no",
    "any", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their", "this", "that",
    "these", "those", "what", "which", "who", "whom", "whose",
}


def _has_low_function_word_ratio(text: str, threshold: float = COHERENCE_FUNCTION_WORD_THRESHOLD) -> bool:
    """Check if text has suspiciously low ratio of function words.

    Natural English text typically has 30-50% function words.
    Gibberish made of strung-together nouns/jargon has very few.

    Args:
        text: Text to check
        threshold: Minimum ratio of function words (default 0.15)

    Returns:
        True if text has too few function words (likely gibberish)
    """
    tokens = re.findall(r'\b\w+\b', text.lower())
    if len(tokens) < MIN_TOKENS_TRIGRAM:
        return False  # Too short to judge

    function_count = sum(1 for t in tokens if t in FUNCTION_WORDS)
    ratio = function_count / len(tokens)

    return ratio < threshold


def _get_tokenizer():
    """Get a cached tokenizer for nonsense word detection."""
    if "tokenizer" not in _tokenizer_cache:
        try:
            from transformers import AutoTokenizer
            _tokenizer_cache["tokenizer"] = AutoTokenizer.from_pretrained(
                "gpt2", use_fast=True
            )
        except Exception:
            _tokenizer_cache["tokenizer"] = None
    return _tokenizer_cache["tokenizer"]


def _is_nonsense_word(word: str, tokenizer) -> bool:
    """Check if a word is nonsense by counting subword tokens.

    Real words tokenize into fewer tokens. Nonsense words get split
    into many small fragments.

    Args:
        word: The word to check
        tokenizer: A tokenizer instance

    Returns:
        True if the word appears to be nonsense
    """
    if not tokenizer or len(word) < NONSENSE_MIN_TOKENS:
        return False

    # Skip non-ASCII words (likely other languages)
    if not word.isascii():
        return False

    # Tokenize the word
    tokens = tokenizer.encode(word, add_special_tokens=False)

    # Calculate ratio of tokens to characters
    # Real words: ~1 token per 3-4 chars
    # Nonsense: ~1 token per 1-2 chars
    ratio = len(tokens) / len(word)

    # If more than 1 token per 2 characters AND at least 4 tokens, likely nonsense
    if ratio > COHERENCE_RATIO_THRESHOLD and len(tokens) >= COHERENCE_QUALITY_MIN_TOKENS:
        return True

    return False


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
    if len(text) > 50 and space_ratio < COHERENCE_SPACE_RATIO_MIN:
        return True

    tokens = text.split()
    if not tokens:
        return False

    # Check 2: Long tokens (concatenated words)
    long_tokens = sum(1 for t in tokens if len(t) > COHERENCE_LONG_TOKEN_LEN)
    if long_tokens / len(tokens) > COHERENCE_LONG_TOKEN_RATIO:
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
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "is", "are", "was", "were", "been",
        "has", "had", "does", "did", "here", "where", "why", "how", "very",
        "more", "some", "any", "also", "than", "then",
    }

    if len(tokens) >= NONSENSE_MIN_TOKENS:
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
        if validity_ratio < COHERENCE_VALIDITY_RATIO_MIN:
            return True

    # Check 6: Function word ratio - real English has ~30-50% function words
    # Gibberish made of strung-together nouns/jargon has very few
    if _has_low_function_word_ratio(text, threshold=COHERENCE_FUNCTION_WORD_THRESHOLD):
        return True

    return False


# Import evaluate_quality and _is_incoherent from helpers
# (split to meet 300-line limit)
from wisent.core.evaluators.personalization._coherence_helpers import (  # noqa: E402
    evaluate_quality,
    _is_incoherent,
)

__all__ = ["evaluate_quality"]
