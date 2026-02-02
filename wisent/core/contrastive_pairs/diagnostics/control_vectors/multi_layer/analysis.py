"""Multi-layer geometry analysis functions."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ..geometry import (
    StructureType,
    GeometryAnalysisConfig,
    GeometryAnalysisResult,
    detect_geometry_structure,
)
from .types import (
    MultiLayerGeometryConfig,
    LayerGeometryResult,
    MultiLayerGeometryResult,
)
from .helpers import (
    combine_layer_activations,
    analyze_subsets,
    analyze_pairs,
    analyze_adjacent,
    analyze_skip,
    analyze_custom,
    compare_combined_vs_single,
    generate_recommendation,
)


__all__ = ["detect_geometry_multi_layer", "detect_geometry_all_layers"]


def detect_geometry_multi_layer(
    pos_activations_by_layer: Dict[int, torch.Tensor],
    neg_activations_by_layer: Dict[int, torch.Tensor],
    config: MultiLayerGeometryConfig | None = None,
) -> MultiLayerGeometryResult:
    """Detect geometric structure across multiple layers."""
    cfg = config or MultiLayerGeometryConfig()
    geo_cfg = GeometryAnalysisConfig(
        num_components=cfg.num_components, optimization_steps=cfg.optimization_steps
    )

    layers = sorted(pos_activations_by_layer.keys())
    if not layers:
        raise ValueError("No layers provided")

    all_combo_results: Dict[str, GeometryAnalysisResult] = {}
    per_layer_results: Dict[int, LayerGeometryResult] = {}
    structure_by_depth: Dict[str, List[float]] = {
        "linear": [], "cone": [], "cluster": [], "manifold": [],
        "sparse": [], "bimodal": [], "orthogonal": []
    }

    # 1. Analyze each layer individually
    if cfg.analyze_per_layer:
        for layer in layers:
            pos_acts = pos_activations_by_layer[layer]
            neg_acts = neg_activations_by_layer[layer]
            result = detect_geometry_structure(pos_acts, neg_acts, geo_cfg)
            all_scores = {name: score.score for name, score in result.all_scores.items()}
            per_layer_results[layer] = LayerGeometryResult(
                layer=layer, best_structure=result.best_structure,
                best_score=result.best_score, all_scores=all_scores,
            )
            all_combo_results[f"L{layer}"] = result
            for struct_name, score in all_scores.items():
                if struct_name in structure_by_depth:
                    structure_by_depth[struct_name].append(score)

    # 2. Find best single layer
    if per_layer_results:
        best_layer = max(per_layer_results.keys(), key=lambda l: per_layer_results[l].best_score)
        best_single_layer = best_layer
        best_single_layer_structure = per_layer_results[best_layer].best_structure
        best_single_layer_score = per_layer_results[best_layer].best_score
    else:
        best_single_layer, best_single_layer_structure = layers[0], StructureType.UNKNOWN
        best_single_layer_score = 0.0

    # 3. Analyze all layers combined
    combined_result = None
    if cfg.analyze_combined and len(layers) > 1:
        combined_pos, combined_neg = combine_layer_activations(
            pos_activations_by_layer, neg_activations_by_layer, layers, cfg.combination_method
        )
        combined_result = detect_geometry_structure(combined_pos, combined_neg, geo_cfg)
        all_combo_results["all_layers"] = combined_result

    # 4-8. Analyze various layer combinations
    layer_subset_results = analyze_subsets(
        pos_activations_by_layer, neg_activations_by_layer, layers, cfg, geo_cfg, all_combo_results
    )
    layer_pair_results = analyze_pairs(
        pos_activations_by_layer, neg_activations_by_layer, layers, cfg, geo_cfg, all_combo_results
    )
    adjacent_pair_results = analyze_adjacent(
        pos_activations_by_layer, neg_activations_by_layer, layers, cfg, geo_cfg, all_combo_results
    )
    skip_results = analyze_skip(
        pos_activations_by_layer, neg_activations_by_layer, layers, cfg, geo_cfg, all_combo_results
    )
    custom_results = analyze_custom(
        pos_activations_by_layer, neg_activations_by_layer, layers, cfg, geo_cfg, all_combo_results
    )

    # 9. Compute layer agreement
    if per_layer_results:
        structures = [r.best_structure for r in per_layer_results.values()]
        most_common = max(set(structures), key=structures.count)
        layer_agreement = structures.count(most_common) / len(structures)
    else:
        layer_agreement = 0.0

    # 10. Rank all combinations
    all_combinations_ranked = sorted(
        [(name, r.best_score, r.best_structure) for name, r in all_combo_results.items()],
        key=lambda x: x[1], reverse=True
    )

    # Determine best combination
    if all_combinations_ranked:
        best_combo_name, best_combo_score, best_combo_structure = all_combinations_ranked[0]
        if best_combo_score > best_single_layer_score:
            best_combination = best_combo_name
            best_combination_score = best_combo_score
            best_combination_structure = best_combo_structure
        else:
            best_combination = None
            best_combination_score = best_single_layer_score
            best_combination_structure = best_single_layer_structure
    else:
        best_combination = None
        best_combination_score = best_single_layer_score
        best_combination_structure = best_single_layer_structure

    combined_vs_single = compare_combined_vs_single(
        combined_result, best_single_layer, best_single_layer_score
    )
    recommendation = generate_recommendation(
        per_layer_results, layer_subset_results, skip_results, layer_pair_results,
        best_single_layer, best_single_layer_structure, best_single_layer_score,
        best_combination, best_combination_score, best_combination_structure,
        layer_agreement, all_combinations_ranked
    )

    return MultiLayerGeometryResult(
        per_layer_results=per_layer_results, combined_result=combined_result,
        layer_subset_results=layer_subset_results, layer_pair_results=layer_pair_results,
        adjacent_pair_results=adjacent_pair_results, skip_results=skip_results,
        custom_results=custom_results, best_single_layer=best_single_layer,
        best_single_layer_structure=best_single_layer_structure,
        best_single_layer_score=best_single_layer_score,
        best_combination=best_combination, best_combination_score=best_combination_score,
        best_combination_structure=best_combination_structure,
        combined_vs_single=combined_vs_single, layer_agreement=layer_agreement,
        structure_by_depth=structure_by_depth,
        all_combinations_ranked=all_combinations_ranked, recommendation=recommendation,
    )


def detect_geometry_all_layers(
    pairs_with_activations: List,
    layers: Optional[List[int]] = None,
    config: MultiLayerGeometryConfig | None = None,
) -> MultiLayerGeometryResult:
    """Convenience function to detect geometry from pairs with activations."""
    if not pairs_with_activations:
        raise ValueError("No pairs provided")

    pos_by_layer: Dict[int, List[torch.Tensor]] = {}
    neg_by_layer: Dict[int, List[torch.Tensor]] = {}

    for pair in pairs_with_activations:
        pos_acts = pair.positive_response.layers_activations
        neg_acts = pair.negative_response.layers_activations
        for layer_key, act in pos_acts.items():
            layer = int(layer_key)
            if layers is None or layer in layers:
                pos_by_layer.setdefault(layer, []).append(act.float() if act is not None else None)
        for layer_key, act in neg_acts.items():
            layer = int(layer_key)
            if layers is None or layer in layers:
                neg_by_layer.setdefault(layer, []).append(act.float() if act is not None else None)

    pos_tensors, neg_tensors = {}, {}
    for layer in pos_by_layer:
        valid_pos = [a for a in pos_by_layer[layer] if a is not None]
        valid_neg = [a for a in neg_by_layer.get(layer, []) if a is not None]
        if valid_pos and valid_neg:
            pos_tensors[layer] = torch.stack(valid_pos)
            neg_tensors[layer] = torch.stack(valid_neg)

    return detect_geometry_multi_layer(pos_tensors, neg_tensors, config)
