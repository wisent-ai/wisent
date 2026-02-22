"""
Backward-compatibility shim.

All code has moved to wisent.core.activations.cache subpackage.
This file re-exports everything for existing imports.
"""

from wisent.core.activations.cache import (
    get_strategy_text_family,
    CachedActivations,
    RawPairData,
    RawCachedActivations,
    RawActivationCache,
    ActivationCache,
    collect_and_cache_activations,
    collect_and_cache_raw_activations,
)

__all__ = [
    "get_strategy_text_family",
    "CachedActivations",
    "RawPairData",
    "RawCachedActivations",
    "RawActivationCache",
    "ActivationCache",
    "collect_and_cache_activations",
    "collect_and_cache_raw_activations",
]
