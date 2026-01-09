"""
Geometry analysis module for activation representations.

This module provides metrics and analysis tools for understanding
the geometric structure of activation spaces.
"""

from .probe_metrics import (
    compute_signal_strength,
    compute_linear_probe_accuracy,
    compute_mlp_probe_accuracy,
    compute_knn_accuracy,
    compute_knn_pca_accuracy,
    compute_knn_umap_accuracy,
    compute_knn_pacmap_accuracy,
)

from .distribution_metrics import (
    compute_mmd_rbf,
    compute_density_ratio,
    compute_fisher_per_dimension,
)

from .intrinsic_dim import (
    estimate_local_intrinsic_dim,
    compute_local_intrinsic_dims,
    compute_diff_intrinsic_dim,
)

from .direction_metrics import (
    compute_direction_from_pairs,
    compute_direction_stability,
    compute_multi_direction_accuracy,
    compute_pairwise_diff_consistency,
)

from .steerability import (
    compute_steerability_metrics,
    compute_linearity_score,
    compute_recommendation,
    compute_adaptive_recommendation,
    compute_robust_recommendation,
)

from .nonsense_baseline import (
    generate_nonsense_activations,
    compute_nonsense_baseline,
    analyze_with_nonsense_baseline,
)

from .concept_analysis import (
    detect_multiple_concepts,
    split_by_concepts,
    analyze_concept_independence,
    compute_concept_coherence,
    compute_concept_stability,
    decompose_into_concepts,
    find_mixed_pairs,
    get_pure_concept_pairs,
    recommend_per_concept_steering,
)

from .signal_analysis import (
    compute_signal_to_noise,
    compute_null_distribution,
    compare_to_null,
    validate_concept,
    compute_bootstrap_signal_estimate,
    compute_saturation_check,
    find_optimal_pair_count,
)

from .icd import (
    compute_icd,
)

from .runner import (
    run_full_repscan,
    run_full_repscan_with_layer_search,
    run_full_repscan_with_steering_eval,
    evaluate_steering_effectiveness,
    evaluate_activation_regions,
    compute_geometry_metrics,
)

from .transformer_analysis import (
    TransformerComponent,
    analyze_transformer_components,
    get_component_hook_points,
    compare_components_for_benchmark,
    compare_concept_granularity,
)

__all__ = [
    # Probe metrics
    "compute_signal_strength",
    "compute_linear_probe_accuracy",
    "compute_mlp_probe_accuracy",
    "compute_knn_accuracy",
    "compute_knn_pca_accuracy",
    "compute_knn_umap_accuracy",
    "compute_knn_pacmap_accuracy",
    # Distribution metrics
    "compute_mmd_rbf",
    "compute_density_ratio",
    "compute_fisher_per_dimension",
    # Intrinsic dimension
    "estimate_local_intrinsic_dim",
    "compute_local_intrinsic_dims",
    "compute_diff_intrinsic_dim",
    # Direction metrics
    "compute_direction_from_pairs",
    "compute_direction_stability",
    "compute_multi_direction_accuracy",
    "compute_pairwise_diff_consistency",
    # Steerability
    "compute_steerability_metrics",
    "compute_linearity_score",
    "compute_recommendation",
    "compute_adaptive_recommendation",
    "compute_robust_recommendation",
    # Nonsense baseline
    "generate_nonsense_activations",
    "compute_nonsense_baseline",
    "analyze_with_nonsense_baseline",
    # Concept analysis
    "detect_multiple_concepts",
    "split_by_concepts",
    "analyze_concept_independence",
    "compute_concept_coherence",
    "compute_concept_stability",
    "decompose_into_concepts",
    "find_mixed_pairs",
    "get_pure_concept_pairs",
    "recommend_per_concept_steering",
    # Signal analysis
    "compute_signal_to_noise",
    "compute_null_distribution",
    "compare_to_null",
    "validate_concept",
    "compute_bootstrap_signal_estimate",
    "compute_saturation_check",
    "find_optimal_pair_count",
    # ICD
    "compute_icd",
    # Runner
    "run_full_repscan",
    "run_full_repscan_with_layer_search",
    "run_full_repscan_with_steering_eval",
    "evaluate_steering_effectiveness",
    "evaluate_activation_regions",
    "compute_geometry_metrics",
    # Transformer analysis
    "TransformerComponent",
    "analyze_transformer_components",
    "get_component_hook_points",
    "compare_components_for_benchmark",
    "compare_concept_granularity",
]
