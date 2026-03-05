import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)

"""
Geometry analysis module for activation representations.

This module provides metrics and analysis tools for understanding
the geometric structure of activation spaces.
"""

# Configure numba threading before any other imports (prevents hangs on macOS)
from .modules.geo_utils.config import numba_config  # noqa: F401

from .utilities.metrics.probe.probe_metrics import (
    compute_signal_strength,
    compute_linear_probe_accuracy,
    compute_mlp_probe_accuracy,
    compute_knn_accuracy,
    compute_knn_pca_accuracy,
    compute_knn_umap_accuracy,
    compute_knn_pacmap_accuracy,
)

from .utilities.metrics.distribution.distribution_metrics import (
    compute_mmd_rbf,
    compute_density_ratio,
    compute_fisher_per_dimension,
)

from .utilities.signal_analysis.intrinsic_dim import (
    estimate_local_intrinsic_dim,
    compute_local_intrinsic_dims,
    compute_diff_intrinsic_dim,
    participation_ratio,
    effective_rank,
    stable_rank,
    two_nn_dimension,
    pca_variance_dimensions,
    compute_effective_dimensions,
)

from .modules.validation.null_tests import (
    compute_null_effective_dimensions,
    compute_effective_dimensions_vs_null,
    compute_cone_null,
    compute_sphere_null,
    compute_cluster_null,
    compute_translation_null,
    compute_geometry_vs_null,
)

from .utilities.metrics.direction.direction_metrics import (
    compute_direction_from_pairs,
    compute_direction_stability,
    compute_pairwise_diff_consistency,
)
from .utilities.metrics.direction.multi_direction import compute_multi_direction_accuracy

from .modules.steering.analysis.steerability import (
    compute_steerability_metrics,
    compute_linearity_score,
    compute_recommendation,
    compute_adaptive_recommendation,
    compute_robust_recommendation,
    compute_final_steering_prescription,
)

from .utilities.data.sources.nonsense import (
    generate_nonsense_activations,
    compute_nonsense_baseline,
    analyze_with_nonsense_baseline,
)

from .utilities.concepts import (
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

from .utilities.signal_analysis.signal_analysis import (
    compute_signal_to_noise,
    compute_null_distribution,
    compare_to_null,
    validate_concept,
    compute_bootstrap_signal_estimate,
    compute_saturation_check,
    find_optimal_pair_count,
)

from .modules.geo_utils.icd import (
    compute_icd,
)

from .runner import (
    run_full_zwiad,
    run_full_zwiad_with_layer_search,
    run_full_zwiad_with_steering_eval,
    evaluate_steering_effectiveness,
    evaluate_activation_regions,
    compute_geometry_metrics,
)

from .utilities.signal_analysis.structure import (
    TransformerComponent,
    analyze_transformer_components,
    get_component_hook_points,
    compare_components_for_benchmark,
    compare_concept_granularity,
)

from .utilities.metrics.representation import (
    compute_magnitude_metrics,
    compute_sparsity_metrics,
    compute_pair_quality_metrics,
    compute_cross_layer_consistency,
    compute_token_position_metrics,
    compute_manifold_metrics,
    compute_direction_overlap_metrics,
    compute_noise_baseline_comparison,
)
from .utilities.metrics.direction.representation_metrics import (
    compute_all_representation_metrics,
    analyze_representation_geometry,
)

from .utilities.signal_analysis.structure import (
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

from wisent.core.utils.visualization.geometry.public.visualizations import (
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

from .modules.geo_utils.config.pacmap_alt import (
    plot_pacmap_alt,
    pacmap_embedding,
)

from .utilities.concepts import (
    decompose_and_name_concepts,
    name_concepts,
    create_concept_overview_figure,
    create_per_concept_figure,
    create_all_concept_figures,
)

from .modules.zwiad.zwiad_with_concepts import (
    run_zwiad_with_concept_naming,
    extract_pair_texts_from_enriched_pairs,
)

from .modules.zwiad.zwiad_config import ZwiadProtocolConfig
from .modules.zwiad.zwiad_protocol import run_zwiad_protocol

from .intervention_selection import (
    RigorousInterventionResult,
    rigorous_select_intervention,
)

from .modules.steering.analysis.steering_validation import (
    compute_steering_effect_size,
    validate_steering_effectiveness,
    run_full_validation,
)

from wisent.core.utils.visualization.geometry.public.direction_visualization import (
    compute_direction_angles,
    plot_direction_similarity_matrix,
    plot_directions_in_pca_space,
    compute_per_concept_directions,
    visualize_concept_directions,
)

from .modules.validation.behavioral_validation import (
    BehavioralValidationResult,
    compute_activation_movement,
    compute_behavioral_change,
    validate_steering_behavioral,
    run_behavioral_validation,
)

from .geometry_search_space import (
    GeometrySearchSpace,
    GeometrySearchConfig,
)

# Re-export database loaders for backward compatibility
from .utilities.data.database_loaders import (
    load_activations_from_database,
    load_pair_texts_from_database,
    load_available_layers_from_database,
)

from .utilities.data.cache import get_cache_path, get_cached_layers

# Re-export steering utilities for backward compatibility
from wisent.core.utils.visualization.steering.steering_visualizations import create_steering_effect_figure
from wisent.core.utils.visualization.steering.steering_viz_utils import (
    create_steering_object_from_pairs,
    extract_activations_from_responses,
    load_reference_activations,
    train_classifier_and_predict,
    save_viz_summary,
)
