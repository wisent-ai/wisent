"""
Two-cloud relationship metrics and analysis orchestrator.

Functions for analyzing geometric relationships between positive and
negative activation clouds, plus the top-level analysis function.

Extracted from activation_structure.py to keep files under 300 lines.
"""

import torch
import numpy as np
from typing import Dict, Any

from wisent.core.utils.config_tools.constants import NORM_EPS, MIN_CLOUD_POINTS
from wisent.core.reading.modules.analysis.structure._cloud_metrics import (
    compute_cloud_shape,
    compute_cone_fit,
    compute_sphere_fit,
    compute_manifold_dimension,
    compute_cluster_structure,
    compute_density_structure,
)


def compute_two_cloud_relationship(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, Any]:
    """Analyze the geometric relationship between pos and neg clouds."""
    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()
    n_pos, d = pos.shape
    n_neg = neg.shape[0]
    if n_pos < MIN_CLOUD_POINTS or n_neg < MIN_CLOUD_POINTS:
        return {"error": "need at least 3 points in each cloud"}
    pos_centroid = pos.mean(axis=0)
    neg_centroid = neg.mean(axis=0)
    centroid_distance = float(np.linalg.norm(pos_centroid - neg_centroid))
    centroid_direction = pos_centroid - neg_centroid
    centroid_dir_norm = np.linalg.norm(centroid_direction)
    if centroid_dir_norm > NORM_EPS:
        centroid_direction = centroid_direction / centroid_dir_norm
    pos_spread = float(np.linalg.norm(pos - pos_centroid, axis=1).mean())
    neg_spread = float(np.linalg.norm(neg - neg_centroid, axis=1).mean())
    avg_spread = (pos_spread + neg_spread) / 2
    separation_ratio = centroid_distance / (avg_spread + NORM_EPS)
    pos_to_pos_centroid = np.linalg.norm(pos - pos_centroid, axis=1)
    pos_to_neg_centroid = np.linalg.norm(pos - neg_centroid, axis=1)
    pos_overlap = float((pos_to_neg_centroid < pos_to_pos_centroid).mean())
    neg_to_pos_centroid = np.linalg.norm(neg - pos_centroid, axis=1)
    neg_to_neg_centroid = np.linalg.norm(neg - neg_centroid, axis=1)
    neg_overlap = float((neg_to_pos_centroid < neg_to_neg_centroid).mean())
    pos_centered = pos - pos_centroid
    neg_centered = neg - neg_centroid
    try:
        _, _, Vh_pos = np.linalg.svd(pos_centered, full_matrices=False)
        _, _, Vh_neg = np.linalg.svd(neg_centered, full_matrices=False)
        pc1_alignment = float(abs(np.dot(Vh_pos[0], Vh_neg[0])))
        pc2_alignment = float(abs(np.dot(Vh_pos[1], Vh_neg[1]))) if len(Vh_pos) > 1 else 0
    except:
        pc1_alignment = 0
        pc2_alignment = 0
    return {
        "centroid_distance": centroid_distance,
        "separation_ratio": separation_ratio,
        "pos_spread": pos_spread, "neg_spread": neg_spread,
        "pos_overlap_fraction": pos_overlap,
        "neg_overlap_fraction": neg_overlap,
        "pc1_alignment": pc1_alignment,
        "pc2_alignment": pc2_alignment,
    }


def compute_relative_position(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, Any]:
    """Analyze relative position: is neg a shifted/rotated version of pos?"""
    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()
    n = min(len(pos), len(neg))
    pos, neg = pos[:n], neg[:n]
    pos_centroid = pos.mean(axis=0)
    neg_centroid = neg.mean(axis=0)
    shift = pos_centroid - neg_centroid
    shift_norm = np.linalg.norm(shift)
    neg_shifted = neg + shift
    residuals = pos - neg_shifted
    residual_norms = np.linalg.norm(residuals, axis=1)
    orig_diffs = pos - neg
    orig_diff_norms = np.linalg.norm(orig_diffs, axis=1)
    shift_explains = 1 - (residual_norms.mean() / (orig_diff_norms.mean() + NORM_EPS))
    diff_normalized = orig_diffs / (orig_diff_norms[:, np.newaxis] + NORM_EPS)
    mean_diff_dir = diff_normalized.mean(axis=0)
    mean_diff_dir = mean_diff_dir / (np.linalg.norm(mean_diff_dir) + NORM_EPS)
    translation_consistency = float((diff_normalized @ mean_diff_dir).mean())
    return {
        "shift_vector_norm": float(shift_norm),
        "shift_explains_fraction": float(shift_explains),
        "translation_consistency": translation_consistency,
        "residual_after_shift": float(residual_norms.mean()),
    }


def analyze_activation_structure(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, Any]:
    """
    Comprehensive analysis of activation space structure.

    Analyzes individual cloud shapes, specific structure tests,
    and relationships between pos and neg clouds.
    """
    from wisent.core.reading.modules.analysis.structure._cloud_metrics import (
        compute_topology_indicators,
    )

    results = {}
    results["pos_shape"] = compute_cloud_shape(pos_activations)
    results["neg_shape"] = compute_cloud_shape(neg_activations)
    results["pos_cone"] = compute_cone_fit(pos_activations)
    results["neg_cone"] = compute_cone_fit(neg_activations)
    results["pos_sphere"] = compute_sphere_fit(pos_activations)
    results["neg_sphere"] = compute_sphere_fit(neg_activations)
    results["pos_manifold"] = compute_manifold_dimension(pos_activations)
    results["neg_manifold"] = compute_manifold_dimension(neg_activations)
    results["pos_clusters"] = compute_cluster_structure(pos_activations)
    results["neg_clusters"] = compute_cluster_structure(neg_activations)
    results["pos_density"] = compute_density_structure(pos_activations)
    results["neg_density"] = compute_density_structure(neg_activations)
    results["relationship"] = compute_two_cloud_relationship(pos_activations, neg_activations)
    results["relative_position"] = compute_relative_position(pos_activations, neg_activations)
    return results
