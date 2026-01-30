"""
Core geometry metrics computation.

This module contains the main compute_geometry_metrics function that computes
all geometry metrics for activation representations.
"""

import os
os.environ["NUMBA_NUM_THREADS"] = "1"

from typing import Dict, Optional, Any
import torch

from ..probe.probe_metrics import (
    compute_signal_strength, compute_linear_probe_accuracy,
    compute_mlp_probe_accuracy, compute_knn_accuracy, compute_knn_pca_accuracy,
)
from ..distribution.distribution_metrics import (
    compute_mmd_rbf, compute_density_ratio, compute_fisher_per_dimension,
)
from ...analysis.intrinsic_dim import compute_local_intrinsic_dims
from ..direction.direction_metrics import (
    compute_direction_stability,
    compute_pairwise_diff_consistency,
)
from ..direction.multi_direction import compute_multi_direction_accuracy
from ...steering.analysis.steerability import compute_steerability_metrics
from ...steering.analysis.steering_recommendation import compute_steering_recommendation
from ...utils.icd import compute_icd
from ...concepts import detect_multiple_concepts, compute_concept_coherence
from ...analysis.signal_analysis import compute_signal_to_noise
from ...analysis.activation_structure import (
    compute_two_cloud_relationship, compute_relative_position, compute_cluster_structure,
)
from ..representation import (
    compute_magnitude_metrics, compute_sparsity_metrics, compute_pair_quality_metrics,
    compute_manifold_metrics, compute_noise_baseline_comparison,
)
from ...data.nonsense_baseline import analyze_with_nonsense_baseline
from ...analysis.transformer_analysis import compare_components_for_benchmark
from .metrics_viz import generate_metrics_visualizations


def compute_geometry_metrics(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_folds: int = 5,
    model=None,
    tokenizer=None,
    layer: Optional[int] = None,
    device: str = "cuda",
    pos_activations_by_component: Optional[Dict[str, torch.Tensor]] = None,
    neg_activations_by_component: Optional[Dict[str, torch.Tensor]] = None,
    generate_visualizations: bool = False,
) -> Dict[str, Any]:
    """Compute comprehensive geometry metrics for activations."""
    metrics = {}

    # Basic probe metrics
    metrics["signal_strength"] = compute_signal_strength(pos_activations, neg_activations, n_folds)
    metrics["linear_probe_accuracy"] = compute_linear_probe_accuracy(pos_activations, neg_activations, n_folds)
    metrics["mlp_probe_accuracy"] = compute_mlp_probe_accuracy(pos_activations, neg_activations, n_folds=n_folds)

    # ICD
    icd_result = compute_icd(pos_activations, neg_activations)
    metrics.update({f"icd_{k}": v for k, v in icd_result.items()})

    # Direction metrics
    stability = compute_direction_stability(pos_activations, neg_activations)
    metrics.update({f"direction_{k}": v for k, v in stability.items()})
    consistency = compute_pairwise_diff_consistency(pos_activations, neg_activations)
    metrics.update({f"consistency_{k}": v for k, v in consistency.items()})

    # Steerability
    steerability = compute_steerability_metrics(pos_activations, neg_activations)
    metrics.update({f"steer_{k}": v for k, v in steerability.items()})

    # Concept analysis
    metrics["concept_coherence"] = compute_concept_coherence(pos_activations, neg_activations)
    concept_detection = detect_multiple_concepts(pos_activations, neg_activations)
    metrics["n_concepts"] = concept_detection.get("n_concepts", 1)
    metrics["best_silhouette"] = concept_detection.get("best_silhouette", 0)

    # Magnitude metrics
    magnitude = compute_magnitude_metrics(pos_activations, neg_activations)
    metrics.update({f"magnitude_{k}": v for k, v in magnitude.items()})

    # Pair quality metrics
    pair_quality = compute_pair_quality_metrics(pos_activations, neg_activations)
    if "error" not in pair_quality:
        metrics["pair_alignment_mean"] = pair_quality.get("alignment_mean")
        metrics["pair_alignment_std"] = pair_quality.get("alignment_std")
        metrics["pair_outlier_fraction"] = pair_quality.get("outlier_fraction")
        metrics["pair_high_quality_fraction"] = pair_quality.get("high_quality_fraction")

    # Two-cloud relationship
    relationship = compute_two_cloud_relationship(pos_activations, neg_activations)
    if "error" not in relationship:
        metrics["cloud_centroid_distance"] = relationship.get("centroid_distance")
        metrics["cloud_separation_ratio"] = relationship.get("separation_ratio")
        metrics["cloud_pos_overlap"] = relationship.get("pos_overlap_fraction")
        metrics["cloud_neg_overlap"] = relationship.get("neg_overlap_fraction")
        metrics["cloud_pc1_alignment"] = relationship.get("pc1_alignment")

    # Relative position (translation analysis)
    rel_position = compute_relative_position(pos_activations, neg_activations)
    metrics["shift_explains_fraction"] = rel_position.get("shift_explains_fraction")
    metrics["translation_consistency"] = rel_position.get("translation_consistency")

    # Distribution metrics
    metrics["mmd_rbf"] = compute_mmd_rbf(pos_activations, neg_activations)
    metrics["density_ratio"] = compute_density_ratio(pos_activations, neg_activations)
    fisher = compute_fisher_per_dimension(pos_activations, neg_activations)
    metrics.update({f"fisher_{k}": v for k, v in fisher.items()})

    # Intrinsic dim
    dim_pos, dim_neg, dim_ratio = compute_local_intrinsic_dims(pos_activations, neg_activations)
    metrics["intrinsic_dim_pos"] = dim_pos
    metrics["intrinsic_dim_neg"] = dim_neg
    metrics["intrinsic_dim_ratio"] = dim_ratio

    # Multi-direction
    multi_dir = compute_multi_direction_accuracy(pos_activations, neg_activations)
    metrics["multi_dir_saturation_k"] = multi_dir.get("saturation_k", 1)
    metrics["multi_dir_gain"] = multi_dir.get("gain_from_multi", 0.0)

    # k-NN
    metrics["knn_accuracy"] = compute_knn_accuracy(pos_activations, neg_activations)
    metrics["knn_pca_accuracy"] = compute_knn_pca_accuracy(pos_activations, neg_activations)

    # Signal to noise
    metrics["signal_to_noise"] = compute_signal_to_noise(pos_activations, neg_activations)

    # Sparsity metrics
    sparsity = compute_sparsity_metrics(pos_activations, neg_activations)
    metrics["sparsity_neurons_for_50pct"] = sparsity.get("neurons_for_50pct")
    metrics["sparsity_neurons_for_90pct"] = sparsity.get("neurons_for_90pct")
    metrics["sparsity_diff_gini"] = sparsity.get("diff_gini")
    metrics["sparsity_top_10_contribution"] = sparsity.get("top_10_contribution_fraction")

    # Manifold metrics
    manifold = compute_manifold_metrics(pos_activations, neg_activations)
    metrics["manifold_variance_pc1"] = manifold.get("variance_pc1")
    metrics["manifold_dims_for_90pct"] = manifold.get("dims_for_90pct_variance")
    metrics["manifold_participation_ratio"] = manifold.get("participation_ratio")
    metrics["manifold_local_linearity"] = manifold.get("local_linearity_mean")
    metrics["manifold_curvature"] = manifold.get("curvature_proxy")

    # Cluster structure
    pos_clusters = compute_cluster_structure(pos_activations)
    neg_clusters = compute_cluster_structure(neg_activations)
    if "error" not in pos_clusters:
        metrics["pos_best_k_clusters"] = pos_clusters.get("best_k")
        metrics["pos_best_silhouette"] = pos_clusters.get("best_silhouette")
    if "error" not in neg_clusters:
        metrics["neg_best_k_clusters"] = neg_clusters.get("best_k")
        metrics["neg_best_silhouette"] = neg_clusters.get("best_silhouette")

    # Noise baseline comparison
    noise_comparison = compute_noise_baseline_comparison(pos_activations, neg_activations)
    if "error" not in noise_comparison:
        vs_noise = noise_comparison.get("vs_noise", {})
        metrics["noise_alignment_above_baseline"] = vs_noise.get("alignment_mean")
        metrics["noise_linear_probe_above_baseline"] = vs_noise.get("linear_probe")
        stds_above = noise_comparison.get("stds_above_noise", {})
        metrics["noise_alignment_z_score"] = stds_above.get("alignment_mean")
        metrics["noise_linear_z_score"] = stds_above.get("linear_probe")

    # Nonsense baseline (requires model/tokenizer)
    if model is not None and tokenizer is not None:
        nonsense_result = analyze_with_nonsense_baseline(
            pos_activations, neg_activations, model=model,
            tokenizer=tokenizer, layer=layer, device=device,
        )
        if "error" not in nonsense_result:
            metrics["nonsense_baseline"] = nonsense_result
            if nonsense_result.get("z_scores"):
                metrics["nonsense_linear_z"] = nonsense_result["z_scores"].get("linear_probe_z")
                metrics["nonsense_signal_z"] = nonsense_result["z_scores"].get("signal_strength_z")
            metrics["is_real_signal"] = nonsense_result.get("is_real_signal", False)

    # Transformer component analysis
    if pos_activations_by_component is not None and neg_activations_by_component is not None:
        component_result = compare_components_for_benchmark(
            model, tokenizer, pos_activations_by_component, neg_activations_by_component,
        )
        metrics["component_analysis"] = component_result
        if "best_component" in component_result:
            metrics["best_component"] = component_result["best_component"]

    # Visualizations
    if generate_visualizations:
        visualizations = generate_metrics_visualizations(pos_activations, neg_activations, metrics)
        if visualizations:
            metrics["visualizations"] = visualizations

    # Generate recommendation
    recommendation = compute_steering_recommendation(metrics)
    metrics["recommended_method"] = recommendation.get("recommended_method", "CAA")
    metrics["recommendation_confidence"] = recommendation.get("confidence", 0.5)
    metrics["recommendation_reasoning"] = recommendation.get("reasoning", [])
    metrics["method_scores"] = recommendation.get("method_scores", {})

    return metrics
