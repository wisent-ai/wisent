"""Extracted from base_dedupers.py - SimHashDeduper internal methods."""

from typing import Mapping


def hamming_distance_impl(a: int, b: int) -> int:
    """Compute the Hamming distance between two integers.

    XOR the two integers; the number of set bits in the result is
    the Hamming distance. For example, let word_1 = "hause" and
    word_2 = "mause", then:
        a = hash64("hause") = 0b110100101011... (64 bits)
        b = hash64("mause") = 0b110100111011... (64 bits)
        a ^ b = 0b000000110000... (64 bits)
    The number of 1s in a ^ b is the Hamming distance, so here it is 2.

    Args:
        a: First hash integer
        b: Second hash integer

    Returns:
        Hamming distance (number of differing bits)
    """
    x = a ^ b
    return x.bit_count() if hasattr(int, "bit_count") else bin(x).count("1")


def exact_key_impl(item: Mapping[str, str], exact_keys: list) -> tuple:
    """Compute exact dedup key from item fields.

    Args:
        item: Mapping of field names to values
        exact_keys: List of field names to use for exact key

    Returns:
        Tuple of sorted (key, value) pairs
    """
    kv = [(k, item.get(k, "")) for k in exact_keys]
    return tuple(sorted(kv))


def default_stopwords() -> set:
    """Return the default English stopword set for SimHash.

    Returns:
        Set of common English stopwords
    """
    return {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
        "if", "in", "into", "is", "it", "no", "not", "of", "on", "or",
        "such", "that", "the", "their", "then", "there", "these", "they",
        "this", "to", "was", "will", "with", "i", "you", "he", "she", "we",
    }
