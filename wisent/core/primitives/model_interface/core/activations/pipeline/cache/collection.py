"""
High-level collect-and-cache functions.

Orchestrate activation collection and caching in one step.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from wisent.core.activations import ExtractionStrategy
from wisent.core.constants import PROGRESS_REPORT_INTERVAL
from wisent.core.contrastive_pairs.pair import ContrastivePair
from .cached_activations import CachedActivations, get_strategy_text_family
from .raw_cached_activations import RawCachedActivations
from .disk_caches import ActivationCache, RawActivationCache

if TYPE_CHECKING:
    from wisent.core.models.wisent_model import WisentModel


def collect_and_cache_activations(
    model: "WisentModel",
    pairs: List[ContrastivePair],
    benchmark: str,
    strategy: ExtractionStrategy,
    cache: Optional[ActivationCache] = None,
    cache_dir: Optional[str] = None,
    show_progress: bool = True,
) -> CachedActivations:
    """
    Collect activations for all pairs and all layers, then cache.

    Args:
        model: WisentModel instance
        pairs: List of contrastive pairs
        benchmark: Benchmark name
        strategy: Extraction strategy
        cache: Optional existing cache to use
        cache_dir: Cache directory (used if cache not provided)
        show_progress: Print progress

    Returns:
        CachedActivations with all layers for all pairs
    """
    from wisent.core.activations.activations_collector import ActivationCollector

    if cache is None and cache_dir:
        cache = ActivationCache(cache_dir)

    if cache and cache.has(model.model_name, benchmark, strategy):
        if show_progress:
            print(f"Loading cached activations for {benchmark}/{strategy.value}")
        return cache.get(model.model_name, benchmark, strategy)

    collector = ActivationCollector(model=model)

    cached = CachedActivations(
        benchmark=benchmark,
        strategy=strategy,
        model_name=model.model_name,
        num_layers=model.num_layers,
    )

    for i, pair in enumerate(pairs):
        if show_progress and i % PROGRESS_REPORT_INTERVAL == 0:
            print(f"Collecting activations: {i+1}/{len(pairs)}", end="\r", flush=True)

        updated = collector.collect(pair, strategy=strategy, layers=None)
        cached.add_pair(
            updated.positive_response.layers_activations,
            updated.negative_response.layers_activations,
        )

    if show_progress:
        print(f"Collected activations: {len(pairs)}/{len(pairs)} pairs, {cached.num_layers} layers")

    if cache:
        cache.put(cached)
        if show_progress:
            print(f"Cached to {cache.cache_dir}")

    return cached


def collect_and_cache_raw_activations(
    model: "WisentModel",
    pairs: List[ContrastivePair],
    benchmark: str,
    strategy: ExtractionStrategy,
    cache: Optional[RawActivationCache] = None,
    cache_dir: Optional[str] = None,
    show_progress: bool = True,
) -> RawCachedActivations:
    """
    Collect RAW hidden states for all pairs and all layers, then cache.

    Unlike collect_and_cache_activations, this stores full sequences
    allowing extraction with any strategy in the same text family.
    """
    from wisent.core.activations.activations_collector import ActivationCollector

    text_family = get_strategy_text_family(strategy)

    if cache is None and cache_dir:
        cache = RawActivationCache(cache_dir)

    if cache and cache.has(model.model_name, benchmark, text_family):
        if show_progress:
            print(f"Loading cached raw activations for {benchmark}/{text_family}")
        return cache.get(model.model_name, benchmark, text_family)

    collector = ActivationCollector(model=model)

    cached = RawCachedActivations(
        benchmark=benchmark,
        text_family=text_family,
        model_name=model.model_name,
        num_layers=model.num_layers,
    )

    for i, pair in enumerate(pairs):
        if show_progress and i % PROGRESS_REPORT_INTERVAL == 0:
            print(f"Collecting raw activations: {i+1}/{len(pairs)}", end="\r", flush=True)

        raw_data = collector.collect_raw(pair, strategy=strategy, layers=None)
        cached.add_pair(
            pos_hidden_states=raw_data["pos_hidden_states"],
            neg_hidden_states=raw_data["neg_hidden_states"],
            pos_answer_text=raw_data["pos_answer_text"],
            neg_answer_text=raw_data["neg_answer_text"],
            pos_prompt_len=raw_data["pos_prompt_len"],
            neg_prompt_len=raw_data["neg_prompt_len"],
        )

    if show_progress:
        print(f"Collected raw activations: {len(pairs)}/{len(pairs)} pairs, {cached.num_layers} layers")

    if cache:
        cache.put(cached)
        if show_progress:
            print(f"Cached raw to {cache.cache_dir}")

    return cached
