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


def _has_low_function_word_ratio(text: str, threshold: float = 0.15) -> bool:
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
    if len(tokens) < 6:
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
    if not tokenizer or len(word) < 5:
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
    if ratio > 0.5 and len(tokens) >= 4:
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

    # Check 6: Function word ratio - real English has ~30-50% function words
    # Gibberish made of strung-together nouns/jargon has very few
    if _has_low_function_word_ratio(text, threshold=0.15):
        return True

    return False


def _is_incoherent(text: str) -> bool:
    """
    Detect if text is semantically incoherent or unhelpful.

    Catches issues that pass gibberish detection but are still low quality:
    - Too short to be helpful (< 20 chars for a response)
    - Repeated phrases/sentences
    - Circular/meaningless statements
    - Refusals or non-answers disguised as responses
    """
    if not text:
        return True

    text = text.strip()

    # Check 1: Too short to be a helpful response
    if len(text) < 20:
        return True

    # Check 2: Single word or very few words (unhelpful)
    tokens = text.split()
    if len(tokens) < 4:
        return True

    # Check 3: Consecutive duplicate words (e.g., "policymakers policymakers")
    tokens_lower = [t.lower().strip('.,!?"\'-') for t in tokens]
    for i in range(len(tokens_lower) - 1):
        if tokens_lower[i] == tokens_lower[i + 1] and len(tokens_lower[i]) > 2:
            return True

    # Check 4: Repeated sentences - split by sentence endings
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip().lower() for s in sentences if s.strip()]
    if len(sentences) >= 2:
        unique_sentences = set(sentences)
        # If more than half the sentences are duplicates, it's repetitive
        if len(unique_sentences) < len(sentences) * 0.5:
            return True

    # Check 5: Repeated phrases (3+ word sequences appearing multiple times)
    if len(tokens) >= 6:
        trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens) - 2)]
        trigram_counts = Counter(trigrams)
        most_common_count = trigram_counts.most_common(1)[0][1] if trigrams else 0
        # If any trigram appears 3+ times, it's repetitive
        if most_common_count >= 3:
            return True

    # Check 6: Circular statements that don't add information
    # e.g., "Football, football, and the beautiful game are intertwined, intertwined, intertwined"
    unique_tokens = set(t.lower().strip('.,!?"\'-') for t in tokens)
    # If unique words are less than 40% of total words, very repetitive
    if len(tokens) >= 5 and len(unique_tokens) / len(tokens) < 0.4:
        return True

    # Check 7: Non-answer patterns
    non_answer_patterns = [
        r'^"?no"?\.?$',  # Just "No" or "No."
        r'^therefore\s+there\s+(are|is)\s+no',  # "Therefore there are no..."
        r'^you\s+didn\'?t\s+tell\s+me',  # "You didn't tell me..."
        r'^i\'?ve?\s+got\s+a\s+few\s+of\s+you',  # Nonsensical "I've got a few of you"
    ]
    text_lower = text.lower()
    for pattern in non_answer_patterns:
        if re.match(pattern, text_lower):
            return True

    # Check 8: Excessive repetition of the same word (3+ times for content words)
    content_words = [t.lower().strip('.,!?"\'-') for t in tokens if len(t) > 3]
    if content_words:
        word_counts = Counter(content_words)
        for word, count in word_counts.items():
            # Skip common filler words
            if word in {'that', 'this', 'have', 'been', 'with', 'from', 'they', 'would', 'could', 'should'}:
                continue
            # If any content word appears more than 3 times in a short response, flag it
            if count >= 3 and count / len(content_words) > 0.15:
                return True

    # Check 9: Nonsense words (using tokenizer fragmentation)
    tokenizer = _get_tokenizer()
    if tokenizer and len(tokens) >= 5:
        nonsense_count = 0
        for token in tokens_lower:
            if len(token) >= 4 and _is_nonsense_word(token, tokenizer):
                nonsense_count += 1
        # If more than 15% of words are nonsense, flag it
        if nonsense_count / len(tokens) > 0.15:
            return True

    return False


def evaluate_quality(
    response: "str | list[str]",
    model: "PreTrainedModel | None" = None,
    tokenizer: "PreTrainedTokenizer | None" = None,
    device: "torch.device | None" = None,
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
        response: The response to evaluate (string or list of strings)
        model: The model (not used, kept for API compatibility)
        tokenizer: The tokenizer (not used, kept for API compatibility)
        device: Device (not used, kept for API compatibility)

    Returns:
        Quality score between 1 and 100
        - 100 = Perfect quality (no issues detected)
        - 1 = Very poor quality (multiple severe issues)
        - 0 = Gibberish detected
    """
    # Handle list inputs - compute average quality
    if isinstance(response, list):
        if not response:
            return 50.0  # Default if empty
        scores = [evaluate_quality(r) for r in response]
        return sum(scores) / len(scores)
    
    # Check for gibberish first - immediate zero if detected
    if _is_gibberish(response):
        return 0.0

    # Check for incoherent/unhelpful responses - immediate zero if detected
    if _is_incoherent(response):
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
