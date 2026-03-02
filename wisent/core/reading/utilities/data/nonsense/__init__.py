"""Nonsense baseline comparison utilities."""

from .nonsense_baseline import (
    generate_nonsense_activations,
    compute_nonsense_baseline,
    analyze_with_nonsense_baseline,
)
from .nonsense_cache import (
    get_cached_nonsense_from_db,
    cache_nonsense_to_db,
    list_cached_nonsense,
)

__all__ = [
    "generate_nonsense_activations",
    "compute_nonsense_baseline",
    "analyze_with_nonsense_baseline",
    "get_cached_nonsense_from_db",
    "cache_nonsense_to_db",
    "list_cached_nonsense",
]
