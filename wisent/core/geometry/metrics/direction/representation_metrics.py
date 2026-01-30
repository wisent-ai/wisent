"""
High-level representation analysis functions.

Orchestrates all representation metrics for comprehensive analysis.
These are the main entry points that call the metric functions
from the representation subdirectory.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Import basic metric functions from the representation subdirectory
from ..representation import (
    compute_magnitude_metrics,
    compute_sparsity_metrics,
    compute_pair_quality_metrics,
    compute_cross_layer_consistency,
    compute_token_position_metrics,
    compute_manifold_metrics,
    compute_direction_overlap_metrics,
    compute_noise_baseline_comparison,
)


def analyze_representation_geometry(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, Any]:
    """
    Comprehensive geometric analysis of a representation.

    Analyzes the shape of the representation and returns raw metrics
    describing its geometry. Does NOT make recommendations.

    Returns metrics describing:
    - Linear separability (can a hyperplane separate pos/neg?)
    - Cone structure (do diff vectors point same direction?)
    - Intrinsic dimensionality (how many dims does concept use?)
    - Curvature (is the manifold flat or curved?)
    - Sparsity (is signal concentrated or distributed?)
    - Noise comparison (is this real signal?)

    The geometry determines which steering method is appropriate:
    - TIGHT CONE + FLAT + LINEAR -> CAA
    - LINEAR but LOW ALIGNMENT -> Hyperplane
    - MULTI-DIRECTIONAL -> PRISM
    - CURVED/NONLINEAR -> MLP-based
    - HIGH CURVATURE -> TITAN/PULSE
    """
    from ...probe.probe_metrics import (
        compute_linear_probe_accuracy,
        compute_mlp_probe_accuracy,
        compute_knn_accuracy,
    )
    from ....steering.steerability import compute_steerability_metrics

    results = {}

    # 1. Linear separability
    linear_acc = compute_linear_probe_accuracy(pos_activations, neg_activations)
    mlp_acc = compute_mlp_probe_accuracy(pos_activations, neg_activations)
    knn_acc = compute_knn_accuracy(pos_activations, neg_activations)

    results["separability"] = {
        "linear_probe_accuracy": linear_acc,
        "mlp_probe_accuracy": mlp_acc,
        "knn_accuracy": knn_acc,
        "nonlinearity_gap": mlp_acc - linear_acc,
    }

    # 2. Cone structure (direction consistency)
    pair_quality = compute_pair_quality_metrics(pos_activations, neg_activations)
    steerability = compute_steerability_metrics(pos_activations, neg_activations)

    results["cone_structure"] = {
        "alignment_mean": pair_quality.get("alignment_mean"),
        "alignment_std": pair_quality.get("alignment_std"),
        "alignment_min": pair_quality.get("alignment_min"),
        "pct_positive_alignment": steerability.get("pct_positive_alignment"),
        "outlier_fraction": pair_quality.get("outlier_fraction"),
    }

    # 3. Intrinsic dimensionality
    manifold = compute_manifold_metrics(pos_activations, neg_activations)

    results["dimensionality"] = {
        "variance_pc1": manifold.get("variance_pc1"),
        "variance_top5": manifold.get("variance_top5"),
        "dims_for_50pct_variance": manifold.get("dims_for_50pct_variance"),
        "dims_for_90pct_variance": manifold.get("dims_for_90pct_variance"),
        "participation_ratio": manifold.get("participation_ratio"),
    }

    # 4. Curvature
    results["curvature"] = {
        "local_linearity_mean": manifold.get("local_linearity_mean"),
        "local_linearity_std": manifold.get("local_linearity_std"),
        "curvature_proxy": manifold.get("curvature_proxy"),
    }

    # 5. Sparsity
    sparsity = compute_sparsity_metrics(pos_activations, neg_activations)

    results["sparsity"] = {
        "neurons_for_50pct": sparsity.get("neurons_for_50pct"),
        "neurons_for_90pct": sparsity.get("neurons_for_90pct"),
        "neurons_for_50pct_fraction": sparsity.get("neurons_for_50pct_fraction"),
        "diff_gini": sparsity.get("diff_gini"),
        "top_10_contribution_fraction": sparsity.get("top_10_contribution_fraction"),
    }

    # 6. Magnitude
    magnitude = compute_magnitude_metrics(pos_activations, neg_activations)

    results["magnitude"] = {
        "activation_norm_mean": magnitude.get("all_norm_mean"),
        "diff_norm_mean": magnitude.get("diff_norm_mean"),
        "steering_vector_norm": magnitude.get("steering_vector_norm"),
        "steering_to_activation_ratio": magnitude.get("steering_to_activation_ratio"),
        "steering_to_diff_ratio": magnitude.get("steering_to_diff_ratio"),
    }

    # 7. Noise comparison
    noise = compute_noise_baseline_comparison(pos_activations, neg_activations)

    results["noise_comparison"] = {
        "actual_linear_probe": noise.get("actual", {}).get("linear_probe"),
        "noise_linear_probe": noise.get("noise_baseline", {}).get("linear_probe"),
        "actual_alignment": noise.get("actual", {}).get("alignment_mean"),
        "noise_alignment": noise.get("noise_baseline", {}).get("alignment_mean"),
        "alignment_vs_noise": noise.get("vs_noise", {}).get("alignment_mean"),
        "variance_vs_noise": noise.get("vs_noise", {}).get("variance_pc1"),
    }

    # Summary classification (no thresholds, just extracted key values)
    results["summary"] = {
        "is_linearly_separable": linear_acc,
        "is_tight_cone": pair_quality.get("alignment_mean"),
        "is_flat": 1 - (manifold.get("curvature_proxy") or 0),
        "is_low_dimensional": 1 / (manifold.get("dims_for_90pct_variance") or 1),
        "signal_above_noise": noise.get("vs_noise", {}).get("alignment_mean"),
    }

    return results


def compute_all_representation_metrics(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    activations_by_layer: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    pos_activations_by_position: Optional[Dict[int, torch.Tensor]] = None,
    neg_activations_by_position: Optional[Dict[int, torch.Tensor]] = None,
    other_directions: Optional[Dict[str, np.ndarray]] = None,
    include_noise_baseline: bool = True,
) -> Dict[str, Any]:
    """
    Compute all representation metrics.

    This is the main entry point for comprehensive representation description.
    """
    results = {}

    # Always compute these
    results["magnitude"] = compute_magnitude_metrics(pos_activations, neg_activations)
    results["sparsity"] = compute_sparsity_metrics(pos_activations, neg_activations)
    results["pair_quality"] = compute_pair_quality_metrics(pos_activations, neg_activations)
    results["manifold"] = compute_manifold_metrics(pos_activations, neg_activations)

    # Noise baseline comparison (detect if data has semantic content vs noise)
    if include_noise_baseline:
        results["noise_comparison"] = compute_noise_baseline_comparison(pos_activations, neg_activations)

    # Optional: cross-layer consistency
    if activations_by_layer is not None:
        results["cross_layer"] = compute_cross_layer_consistency(activations_by_layer)

    # Optional: token position dependence
    if pos_activations_by_position is not None and neg_activations_by_position is not None:
        results["token_position"] = compute_token_position_metrics(
            pos_activations_by_position, neg_activations_by_position
        )

    # Optional: direction overlap
    results["direction_overlap"] = compute_direction_overlap_metrics(
        pos_activations, neg_activations, other_directions
    )

    return results
