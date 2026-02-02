"""Helper functions for multi-layer geometry analysis."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from ..geometry import (
    GeometryAnalysisConfig,
    GeometryAnalysisResult,
    detect_geometry_structure,
)


__all__ = [
    "combine_layer_activations",
    "analyze_subsets",
    "analyze_pairs",
    "analyze_adjacent",
    "analyze_skip",
    "analyze_custom",
    "compare_combined_vs_single",
    "generate_recommendation",
]


def combine_layer_activations(
    pos_by_layer: Dict[int, torch.Tensor],
    neg_by_layer: Dict[int, torch.Tensor],
    layers: List[int],
    method: str = "concat",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine activations from multiple layers."""
    pos_acts = [pos_by_layer[l] for l in layers if l in pos_by_layer]
    neg_acts = [neg_by_layer[l] for l in layers if l in neg_by_layer]
    if not pos_acts or not neg_acts:
        raise ValueError("No activations found for specified layers")

    if method == "concat":
        return torch.cat(pos_acts, dim=-1), torch.cat(neg_acts, dim=-1)
    elif method == "mean":
        return torch.stack(pos_acts, dim=0).mean(dim=0), torch.stack(neg_acts, dim=0).mean(dim=0)
    elif method == "weighted":
        weights = torch.linspace(0.5, 1.5, len(pos_acts))
        weights = weights / weights.sum()
        combined_pos = sum(w * a for w, a in zip(weights, pos_acts))
        combined_neg = sum(w * a for w, a in zip(weights, neg_acts))
        return combined_pos, combined_neg
    raise ValueError(f"Unknown combination method: {method}")


def analyze_subsets(pos_by_layer, neg_by_layer, layers, cfg, geo_cfg, all_combo_results):
    """Analyze layer subsets (early/middle/late/halves)."""
    results: Dict[str, GeometryAnalysisResult] = {}
    if not cfg.analyze_subsets or len(layers) < 3:
        return results

    n = len(layers)
    third, half = n // 3, n // 2
    subsets = [
        ("early", layers[:third] if third > 0 else layers[:1]),
        ("middle", layers[third:2*third] if third > 0 else layers[1:2]),
        ("late", layers[2*third:] if third > 0 else layers[-1:]),
        ("first_half", layers[:half] if half > 0 else layers[:1]),
        ("second_half", layers[half:] if half > 0 else layers[-1:]),
    ]
    for name, subset in subsets:
        if len(subset) >= 1:
            pos, neg = combine_layer_activations(pos_by_layer, neg_by_layer, subset, cfg.combination_method)
            result = detect_geometry_structure(pos, neg, geo_cfg)
            results[name] = result
            all_combo_results[name] = result
    return results


def analyze_pairs(pos_by_layer, neg_by_layer, layers, cfg, geo_cfg, all_combo_results):
    """Analyze layer pairs (all combinations of 2 layers)."""
    from itertools import combinations
    results: Dict[str, GeometryAnalysisResult] = {}
    if not cfg.analyze_pairs or len(layers) < 2:
        return results

    for i, (l1, l2) in enumerate(combinations(layers, 2)):
        if i >= cfg.max_pair_combinations:
            break
        name = f"L{l1}+L{l2}"
        pos, neg = combine_layer_activations(pos_by_layer, neg_by_layer, [l1, l2], cfg.combination_method)
        result = detect_geometry_structure(pos, neg, geo_cfg)
        results[name] = result
        all_combo_results[name] = result
    return results


def analyze_adjacent(pos_by_layer, neg_by_layer, layers, cfg, geo_cfg, all_combo_results):
    """Analyze adjacent layer pairs (L1+L2, L2+L3, etc.)."""
    results: Dict[str, GeometryAnalysisResult] = {}
    if not cfg.analyze_adjacent or len(layers) < 2:
        return results

    for i in range(len(layers) - 1):
        l1, l2 = layers[i], layers[i + 1]
        name = f"adj_L{l1}+L{l2}"
        pos, neg = combine_layer_activations(pos_by_layer, neg_by_layer, [l1, l2], cfg.combination_method)
        result = detect_geometry_structure(pos, neg, geo_cfg)
        results[name] = result
        all_combo_results[name] = result
    return results


def analyze_skip(pos_by_layer, neg_by_layer, layers, cfg, geo_cfg, all_combo_results):
    """Analyze skip patterns (every 2nd, every 3rd, first/last)."""
    results: Dict[str, GeometryAnalysisResult] = {}
    if not cfg.analyze_skip or len(layers) < 4:
        return results

    patterns = [("every_2nd", layers[::2]), ("first_last", [layers[0], layers[-1]])]
    if len(layers) >= 6:
        patterns.append(("every_3rd", layers[::3]))
    if len(layers) >= 3:
        patterns.append(("first_mid_last", [layers[0], layers[len(layers)//2], layers[-1]]))

    for name, subset in patterns:
        if len(subset) >= 2:
            pos, neg = combine_layer_activations(pos_by_layer, neg_by_layer, subset, cfg.combination_method)
            result = detect_geometry_structure(pos, neg, geo_cfg)
            results[name] = result
            all_combo_results[name] = result
    return results


def analyze_custom(pos_by_layer, neg_by_layer, layers, cfg, geo_cfg, all_combo_results):
    """Analyze custom layer combinations."""
    results: Dict[str, GeometryAnalysisResult] = {}
    if not cfg.analyze_custom:
        return results

    for i, custom_layers in enumerate(cfg.analyze_custom):
        valid = [l for l in custom_layers if l in layers]
        if valid:
            name = f"custom_{i}_L" + "+L".join(map(str, valid))
            pos, neg = combine_layer_activations(pos_by_layer, neg_by_layer, valid, cfg.combination_method)
            result = detect_geometry_structure(pos, neg, geo_cfg)
            results[name] = result
            all_combo_results[name] = result
    return results


def compare_combined_vs_single(combined_result, best_layer, best_score):
    """Compare combined vs single layer performance."""
    if not combined_result:
        return "No comparison available"
    if combined_result.best_score > best_score + 0.1:
        return f"Combined ({combined_result.best_score:.2f}) better than single ({best_score:.2f})"
    elif best_score > combined_result.best_score + 0.1:
        return f"Single L{best_layer} ({best_score:.2f}) better than combined"
    return f"Similar: combined={combined_result.best_score:.2f}, single={best_score:.2f}"


def generate_recommendation(
    per_layer_results, layer_subset_results, skip_results, layer_pair_results,
    best_single_layer, best_single_layer_structure, best_single_layer_score,
    best_combination, best_combination_score, best_combination_structure,
    layer_agreement, all_combinations_ranked
):
    """Generate comprehensive recommendation based on multi-layer analysis."""
    parts = []
    if layer_agreement > 0.8:
        parts.append(f"High agreement ({layer_agreement:.0%}): consistent structure.")
    elif layer_agreement < 0.4:
        parts.append(f"Low agreement ({layer_agreement:.0%}): varies by depth.")
    else:
        parts.append(f"Moderate agreement ({layer_agreement:.0%}).")

    if best_combination and best_combination_score > best_single_layer_score + 0.05:
        improvement = best_combination_score - best_single_layer_score
        parts.append(
            f"BEST: '{best_combination}' ({best_combination_structure.value}: "
            f"{best_combination_score:.2f}) +{improvement:.2f} over L{best_single_layer}."
        )
    else:
        parts.append(
            f"BEST: L{best_single_layer} ({best_single_layer_structure.value}: "
            f"{best_single_layer_score:.2f}). Multi-layer doesn't improve."
        )

    if len(all_combinations_ranked) >= 3:
        top3 = ", ".join([f"{n}={s:.2f}" for n, s, _ in all_combinations_ranked[:3]])
        parts.append(f"Top 3: {top3}.")

    return " ".join(parts)
