"""Exhaustive layer combination search functions."""

from __future__ import annotations

import heapq
from itertools import combinations as itertools_combinations
from math import comb
from typing import Dict, List, Optional, Tuple, Callable

import torch

from ..geometry import (
    StructureType,
    GeometryAnalysisConfig,
    detect_geometry_structure,
)
from ..multi_layer.helpers import combine_layer_activations
from .types import ExhaustiveCombinationResult, ExhaustiveGeometryAnalysisResult
from .helpers import analyze_combination_patterns, generate_exhaustive_recommendation


__all__ = [
    "detect_geometry_exhaustive",
    "detect_geometry_limited",
    "detect_geometry_contiguous",
    "detect_geometry_smart",
]


def _run_combination_test(
    combo: Tuple[int, ...],
    pos_by_layer: Dict[int, torch.Tensor],
    neg_by_layer: Dict[int, torch.Tensor],
    combination_method: str,
    geo_cfg: GeometryAnalysisConfig,
) -> ExhaustiveCombinationResult:
    """Test a single layer combination."""
    if len(combo) == 1:
        combined_pos = pos_by_layer[combo[0]]
        combined_neg = neg_by_layer[combo[0]]
    else:
        combined_pos, combined_neg = combine_layer_activations(
            pos_by_layer, neg_by_layer, list(combo), combination_method
        )
    result = detect_geometry_structure(combined_pos, combined_neg, geo_cfg)
    all_scores = {name: score.score for name, score in result.all_scores.items()}
    return ExhaustiveCombinationResult(
        layers=combo, best_structure=result.best_structure,
        best_score=result.best_score, all_scores=all_scores,
    )


def _finalize_results(
    top_results_heap: List[Tuple[float, ExhaustiveCombinationResult]],
    single_layer_results: List[ExhaustiveCombinationResult],
    layers: List[int],
    total_combinations: int,
) -> ExhaustiveGeometryAnalysisResult:
    """Finalize and package exhaustive search results."""
    all_results = [r for _, r in sorted(top_results_heap, key=lambda x: -x[0])]
    best_result = all_results[0] if all_results else None
    best_combination = best_result.layers if best_result else ()
    best_score = best_result.best_score if best_result else 0.0
    best_structure = best_result.best_structure if best_result else StructureType.UNKNOWN

    if single_layer_results:
        single_layer_results.sort(key=lambda x: x.best_score, reverse=True)
        single_layer_best = single_layer_results[0].layers[0]
        single_layer_best_score = single_layer_results[0].best_score
    else:
        single_layer_best, single_layer_best_score = layers[0], 0.0

    combination_beats_single = best_score > single_layer_best_score
    improvement_over_single = best_score - single_layer_best_score
    patterns = analyze_combination_patterns(all_results, layers, top_k=min(50, len(all_results)))
    recommendation = generate_exhaustive_recommendation(
        best_combination, best_score, best_structure, single_layer_best,
        single_layer_best_score, combination_beats_single, improvement_over_single,
        patterns, total_combinations
    )

    return ExhaustiveGeometryAnalysisResult(
        total_combinations=total_combinations, all_results=all_results,
        best_combination=best_combination, best_score=best_score,
        best_structure=best_structure, top_10=all_results[:10],
        single_layer_best=single_layer_best, single_layer_best_score=single_layer_best_score,
        combination_beats_single=combination_beats_single,
        improvement_over_single=improvement_over_single,
        patterns=patterns, recommendation=recommendation,
    )


def detect_geometry_exhaustive(
    pos_activations_by_layer: Dict[int, torch.Tensor],
    neg_activations_by_layer: Dict[int, torch.Tensor],
    max_layers: int = 16,
    combination_method: str = "concat",
    num_components: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    top_k: int = 100,
) -> ExhaustiveGeometryAnalysisResult:
    """Exhaustively test all 2^N - 1 layer combinations."""
    layers = sorted(pos_activations_by_layer.keys())[:max_layers]
    if not layers:
        raise ValueError("No layers provided")

    geo_cfg = GeometryAnalysisConfig(num_components=num_components, optimization_steps=50)
    total_combinations = (1 << len(layers)) - 1
    top_results_heap: List[Tuple[float, ExhaustiveCombinationResult]] = []
    single_layer_results: List[ExhaustiveCombinationResult] = []

    idx = 0
    for r in range(1, len(layers) + 1):
        for combo in itertools_combinations(layers, r):
            idx += 1
            if progress_callback:
                progress_callback(idx, total_combinations)

            combo_result = _run_combination_test(
                combo, pos_activations_by_layer, neg_activations_by_layer,
                combination_method, geo_cfg
            )
            if len(combo) == 1:
                single_layer_results.append(combo_result)

            if len(top_results_heap) < top_k:
                heapq.heappush(top_results_heap, (combo_result.best_score, combo_result))
            elif combo_result.best_score > top_results_heap[0][0]:
                heapq.heapreplace(top_results_heap, (combo_result.best_score, combo_result))

    return _finalize_results(top_results_heap, single_layer_results, layers, total_combinations)


def detect_geometry_limited(
    pos_activations_by_layer: Dict[int, torch.Tensor],
    neg_activations_by_layer: Dict[int, torch.Tensor],
    max_combo_size: int = 3,
    combination_method: str = "concat",
    num_components: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    top_k: int = 100,
) -> ExhaustiveGeometryAnalysisResult:
    """Test 1,2,3-layer combinations plus all layers. O(N^3) complexity."""
    layers = sorted(pos_activations_by_layer.keys())
    if not layers:
        raise ValueError("No layers provided")

    geo_cfg = GeometryAnalysisConfig(num_components=num_components, optimization_steps=50)
    n = len(layers)
    total = sum(comb(n, r) for r in range(1, min(max_combo_size, n) + 1))
    if max_combo_size < n:
        total += 1

    top_results_heap: List[Tuple[float, ExhaustiveCombinationResult]] = []
    single_layer_results: List[ExhaustiveCombinationResult] = []

    def combo_gen():
        for r in range(1, min(max_combo_size, n) + 1):
            for c in itertools_combinations(layers, r):
                yield c
        if max_combo_size < n:
            yield tuple(layers)

    for idx, combo in enumerate(combo_gen(), 1):
        if progress_callback:
            progress_callback(idx, total)
        combo_result = _run_combination_test(
            combo, pos_activations_by_layer, neg_activations_by_layer,
            combination_method, geo_cfg
        )
        if len(combo) == 1:
            single_layer_results.append(combo_result)
        if len(top_results_heap) < top_k:
            heapq.heappush(top_results_heap, (combo_result.best_score, combo_result))
        elif combo_result.best_score > top_results_heap[0][0]:
            heapq.heapreplace(top_results_heap, (combo_result.best_score, combo_result))

    return _finalize_results(top_results_heap, single_layer_results, layers, total)


def detect_geometry_contiguous(
    pos_activations_by_layer: Dict[int, torch.Tensor],
    neg_activations_by_layer: Dict[int, torch.Tensor],
    combination_method: str = "concat",
    num_components: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    top_k: int = 100,
) -> ExhaustiveGeometryAnalysisResult:
    """Test contiguous layer combinations only. O(N^2) complexity."""
    layers = sorted(pos_activations_by_layer.keys())
    if not layers:
        raise ValueError("No layers provided")

    geo_cfg = GeometryAnalysisConfig(num_components=num_components, optimization_steps=50)
    n = len(layers)
    total = n * (n + 1) // 2

    top_results_heap: List[Tuple[float, ExhaustiveCombinationResult]] = []
    single_layer_results: List[ExhaustiveCombinationResult] = []

    idx = 0
    for start in range(n):
        for end in range(start, n):
            idx += 1
            if progress_callback:
                progress_callback(idx, total)
            combo = tuple(layers[start:end + 1])
            combo_result = _run_combination_test(
                combo, pos_activations_by_layer, neg_activations_by_layer,
                combination_method, geo_cfg
            )
            if len(combo) == 1:
                single_layer_results.append(combo_result)
            if len(top_results_heap) < top_k:
                heapq.heappush(top_results_heap, (combo_result.best_score, combo_result))
            elif combo_result.best_score > top_results_heap[0][0]:
                heapq.heapreplace(top_results_heap, (combo_result.best_score, combo_result))

    return _finalize_results(top_results_heap, single_layer_results, layers, total)


def detect_geometry_smart(
    pos_activations_by_layer: Dict[int, torch.Tensor],
    neg_activations_by_layer: Dict[int, torch.Tensor],
    max_combo_size: int = 3,
    combination_method: str = "concat",
    num_components: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    top_k: int = 100,
) -> ExhaustiveGeometryAnalysisResult:
    """Smart search: contiguous + limited (1,2,3-layer) with deduplication."""
    layers = sorted(pos_activations_by_layer.keys())
    if not layers:
        raise ValueError("No layers provided")

    geo_cfg = GeometryAnalysisConfig(num_components=num_components, optimization_steps=50)
    n = len(layers)

    all_combos_set: set = set()
    for start in range(n):
        for end in range(start, n):
            all_combos_set.add(tuple(layers[start:end + 1]))
    for r in range(1, min(max_combo_size, n) + 1):
        for c in itertools_combinations(layers, r):
            all_combos_set.add(c)

    all_combos = sorted(all_combos_set, key=lambda x: (len(x), x))
    total = len(all_combos)

    top_results_heap: List[Tuple[float, ExhaustiveCombinationResult]] = []
    single_layer_results: List[ExhaustiveCombinationResult] = []

    for idx, combo in enumerate(all_combos, 1):
        if progress_callback:
            progress_callback(idx, total)
        combo_result = _run_combination_test(
            combo, pos_activations_by_layer, neg_activations_by_layer,
            combination_method, geo_cfg
        )
        if len(combo) == 1:
            single_layer_results.append(combo_result)
        if len(top_results_heap) < top_k:
            heapq.heappush(top_results_heap, (combo_result.best_score, combo_result))
        elif combo_result.best_score > top_results_heap[0][0]:
            heapq.heapreplace(top_results_heap, (combo_result.best_score, combo_result))

    return _finalize_results(top_results_heap, single_layer_results, layers, total)
