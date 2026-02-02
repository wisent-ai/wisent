"""Helper functions for exhaustive layer combination analysis."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple, Any

from ..geometry import StructureType
from .types import ExhaustiveCombinationResult


__all__ = ["analyze_combination_patterns", "generate_exhaustive_recommendation"]


def analyze_combination_patterns(
    all_results: List[ExhaustiveCombinationResult],
    layers: List[int],
    top_k: int = 50,
) -> Dict[str, Any]:
    """Analyze patterns in top combinations."""
    top_results = all_results[:top_k]

    # Layer frequency in top combinations
    layer_freq = Counter()
    for r in top_results:
        for layer in r.layers:
            layer_freq[layer] += 1

    # Combination size distribution in top results
    size_dist = Counter(len(r.layers) for r in top_results)

    # Best score by combination size
    size_to_best: Dict[int, float] = {}
    for r in all_results:
        size = len(r.layers)
        if size not in size_to_best or r.best_score > size_to_best[size]:
            size_to_best[size] = r.best_score

    # Structure frequency in top combinations
    structure_freq = Counter(r.best_structure for r in top_results)

    # Adjacent layer pairs in top combinations
    adjacent_count = 0
    for r in top_results:
        if len(r.layers) >= 2:
            sorted_layers = sorted(r.layers)
            for i in range(len(sorted_layers) - 1):
                if sorted_layers[i + 1] - sorted_layers[i] == 1:
                    adjacent_count += 1
                    break

    # Layer position analysis (early vs late layers)
    mid_layer = layers[len(layers) // 2] if layers else 0
    early_in_top = sum(1 for r in top_results for l in r.layers if l < mid_layer)
    late_in_top = sum(1 for r in top_results for l in r.layers if l >= mid_layer)

    return {
        "layer_frequency_in_top": dict(layer_freq.most_common()),
        "most_important_layers": [l for l, _ in layer_freq.most_common(5)],
        "size_distribution_in_top": dict(size_dist),
        "best_score_by_size": size_to_best,
        "optimal_combination_size": max(size_to_best.keys(), key=lambda k: size_to_best[k]) if size_to_best else 1,
        "structure_frequency_in_top": {s.value: c for s, c in structure_freq.most_common()},
        "dominant_structure": structure_freq.most_common(1)[0][0].value if structure_freq else "unknown",
        "adjacent_pairs_in_top": adjacent_count,
        "early_vs_late_ratio": early_in_top / late_in_top if late_in_top > 0 else float('inf'),
    }


def generate_exhaustive_recommendation(
    best_combination: Tuple[int, ...],
    best_score: float,
    best_structure: StructureType,
    single_layer_best: int,
    single_layer_best_score: float,
    combination_beats_single: bool,
    improvement_over_single: float,
    patterns: Dict[str, Any],
    total_combinations: int,
) -> str:
    """Generate recommendation from exhaustive analysis."""
    parts = []

    parts.append(f"Tested {total_combinations} layer combinations.")

    if combination_beats_single and improvement_over_single > 0.05:
        layers_str = "+".join(f"L{l}" for l in best_combination)
        parts.append(
            f"BEST: {layers_str} ({best_structure.value}: {best_score:.3f}), "
            f"+{improvement_over_single:.3f} over single layer L{single_layer_best}."
        )
    else:
        parts.append(
            f"BEST: Single layer L{single_layer_best} ({best_score:.3f}). "
            f"Multi-layer combinations don't significantly improve."
        )

    # Pattern insights
    opt_size = patterns.get("optimal_combination_size", 1)
    if opt_size > 1:
        parts.append(f"Optimal combination size: {opt_size} layers.")

    important_layers = patterns.get("most_important_layers", [])
    if important_layers:
        layers_str = ", ".join(f"L{l}" for l in important_layers[:3])
        parts.append(f"Most important layers: {layers_str}.")

    dominant = patterns.get("dominant_structure", "unknown")
    parts.append(f"Dominant structure: {dominant}.")

    return " ".join(parts)
