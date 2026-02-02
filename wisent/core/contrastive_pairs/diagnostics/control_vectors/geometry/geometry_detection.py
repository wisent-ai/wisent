"""Main geometry structure detection entry point."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from .geometry_types import (
    StructureType,
    StructureScore,
    GeometryAnalysisConfig,
    GeometryAnalysisResult,
)
from .geometry_detectors import (
    detect_linear_structure,
    detect_cone_structure_score,
    detect_cluster_structure,
    detect_manifold_structure,
    detect_sparse_structure,
    detect_bimodal_structure,
    detect_orthogonal_structure,
)

__all__ = ["detect_geometry_structure"]


def detect_geometry_structure(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    config: GeometryAnalysisConfig | None = None,
) -> GeometryAnalysisResult:
    """Detect the geometric structure of activation differences."""
    cfg = config or GeometryAnalysisConfig()

    pos_tensor = pos_activations.detach().float()
    neg_tensor = neg_activations.detach().float()

    if pos_tensor.dim() == 1:
        pos_tensor = pos_tensor.unsqueeze(0)
    if neg_tensor.dim() == 1:
        neg_tensor = neg_tensor.unsqueeze(0)

    n_pairs = min(pos_tensor.shape[0], neg_tensor.shape[0])
    diff_vectors = pos_tensor[:n_pairs] - neg_tensor[:n_pairs]

    raw_scores: Dict[str, StructureScore] = {}
    raw_scores["linear"] = detect_linear_structure(pos_tensor, neg_tensor, diff_vectors, cfg)
    raw_scores["cone"] = detect_cone_structure_score(pos_tensor, neg_tensor, cfg)
    raw_scores["cluster"] = detect_cluster_structure(pos_tensor, neg_tensor, diff_vectors, cfg)
    raw_scores["manifold"] = detect_manifold_structure(pos_tensor, neg_tensor, diff_vectors, cfg)
    raw_scores["sparse"] = detect_sparse_structure(pos_tensor, neg_tensor, diff_vectors, cfg)
    raw_scores["bimodal"] = detect_bimodal_structure(pos_tensor, neg_tensor, diff_vectors, cfg)
    raw_scores["orthogonal"] = detect_orthogonal_structure(pos_tensor, neg_tensor, diff_vectors, cfg)

    best_structure, best_score = _find_most_specific_structure(raw_scores)
    recommendation = _generate_recommendation(best_structure, raw_scores)

    return GeometryAnalysisResult(
        best_structure=best_structure,
        best_score=best_score,
        all_scores=raw_scores,
        recommendation=recommendation,
        details={
            "config": cfg.__dict__,
            "n_positive": pos_tensor.shape[0],
            "n_negative": neg_tensor.shape[0],
            "hidden_dim": pos_tensor.shape[1],
        }
    )


def _find_most_specific_structure(scores: Dict[str, StructureScore]) -> Tuple[StructureType, float]:
    """Find the most specific structure that fits the data well."""
    THRESHOLDS = {
        "linear": 0.5, "cone": 0.5, "orthogonal": 0.5, "cluster": 0.6,
        "sparse": 0.7, "bimodal": 0.5, "manifold": 0.3,
    }
    specificity_order = ["linear", "cone", "orthogonal", "cluster", "sparse", "bimodal", "manifold"]

    for struct_name in specificity_order:
        if struct_name in scores:
            if scores[struct_name].score >= THRESHOLDS.get(struct_name, 0.5):
                return scores[struct_name].structure_type, scores[struct_name].score

    best_key = max(scores.keys(), key=lambda k: scores[k].score)
    return scores[best_key].structure_type, scores[best_key].score


def _generate_recommendation(best_structure: StructureType, all_scores: Dict[str, StructureScore]) -> str:
    """Generate steering method recommendation based on detected geometry."""
    recommendations = {
        StructureType.LINEAR: "Use CAA - single direction steering.",
        StructureType.CONE: "Use PRISM - multi-directional steering.",
        StructureType.CLUSTER: "Use cluster-based steering.",
        StructureType.MANIFOLD: "Use TITAN with learned gating.",
        StructureType.SPARSE: "Use SAE-based steering.",
        StructureType.BIMODAL: "Use PULSE with conditional gating.",
        StructureType.ORTHOGONAL: "Use multiple independent CAA vectors.",
        StructureType.UNKNOWN: "Start with CAA and evaluate.",
    }
    return recommendations.get(best_structure, recommendations[StructureType.UNKNOWN])
