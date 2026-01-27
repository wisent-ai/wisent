"""
Geometry analysis module for activation representations.

This module provides metrics and analysis tools for understanding
the geometric structure of activation spaces.
"""

# Configure numba threading before any other imports (prevents hangs on macOS)
from . import numba_config  # noqa: F401

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
    compute_final_steering_prescription,
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
    analyze_concept_structure,
    get_pair_concept_assignments,
    compute_concept_linear_separability,
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

from .representation_metrics import (
    compute_magnitude_metrics,
    compute_sparsity_metrics,
    compute_pair_quality_metrics,
    compute_cross_layer_consistency,
    compute_manifold_metrics,
    compute_token_position_metrics,
    compute_direction_overlap_metrics,
    compute_noise_baseline_comparison,
    compute_all_representation_metrics,
    analyze_representation_geometry,
)

from .activation_structure import (
    compute_cloud_shape,
    compute_cone_fit,
    compute_sphere_fit,
    compute_manifold_dimension,
    compute_cluster_structure,
    compute_density_structure,
    compute_topology_indicators,
    compute_two_cloud_relationship,
    compute_relative_position,
    analyze_activation_structure,
)

from .visualizations import (
    plot_pca_projection,
    plot_diff_vectors,
    plot_norm_distribution,
    plot_alignment_distribution,
    plot_eigenvalue_spectrum,
    plot_pairwise_distances,
    plot_tsne_projection,
    plot_umap_projection,
    plot_cone_visualization,
    plot_layer_comparison,
    create_summary_figure,
    render_matplotlib_figure,
)

from .pacmap_alt import (
    plot_pacmap_alt,
    pacmap_embedding,
)

from .steering_recommendation import (
    SteeringThresholds,
    compute_steering_recommendation,
    compute_per_layer_recommendation,
    get_method_description,
    get_method_requirements,
)

from .concept_naming import (
    decompose_and_name_concepts,
    name_concepts,
)

from .concept_visualizations import (
    create_concept_overview_figure,
    create_per_concept_figure,
    create_all_concept_figures,
)

from .repscan_with_concepts import (
    run_repscan_with_concept_naming,
    extract_pair_texts_from_enriched_pairs,
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
    "compute_final_steering_prescription",
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
    "analyze_concept_structure",
    "get_pair_concept_assignments",
    "compute_concept_linear_separability",
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
    # Representation metrics
    "compute_magnitude_metrics",
    "compute_sparsity_metrics",
    "compute_pair_quality_metrics",
    "compute_cross_layer_consistency",
    "compute_manifold_metrics",
    "compute_token_position_metrics",
    "compute_direction_overlap_metrics",
    "compute_noise_baseline_comparison",
    "compute_all_representation_metrics",
    "analyze_representation_geometry",
    # Activation structure
    "compute_cloud_shape",
    "compute_cone_fit",
    "compute_sphere_fit",
    "compute_manifold_dimension",
    "compute_cluster_structure",
    "compute_density_structure",
    "compute_topology_indicators",
    "compute_two_cloud_relationship",
    "compute_relative_position",
    "analyze_activation_structure",
    # Visualizations
    "plot_pca_projection",
    "plot_diff_vectors",
    "plot_norm_distribution",
    "plot_alignment_distribution",
    "plot_eigenvalue_spectrum",
    "plot_pairwise_distances",
    "plot_tsne_projection",
    "plot_umap_projection",
    "plot_cone_visualization",
    "plot_layer_comparison",
    "create_summary_figure",
    "render_matplotlib_figure",
    # PaCMAP alternative
    "plot_pacmap_alt",
    "pacmap_embedding",
    # Steering recommendation
    "SteeringThresholds",
    "compute_steering_recommendation",
    "compute_per_layer_recommendation",
    "get_method_description",
    "get_method_requirements",
    # Concept naming
    "decompose_and_name_concepts",
    "name_concepts",
    "run_repscan_with_concept_naming",
    "extract_pair_texts_from_enriched_pairs",
    # Concept visualizations
    "create_concept_overview_figure",
    "create_per_concept_figure",
    "create_all_concept_figures",
]
