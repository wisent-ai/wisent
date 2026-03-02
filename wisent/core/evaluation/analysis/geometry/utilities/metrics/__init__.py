"""Geometry metrics package - re-exports from subdirectories."""
from .core.metrics_core import compute_geometry_metrics
from .core.metrics_viz import generate_metrics_visualizations
from .probe.probe_metrics import (
    compute_signal_strength,
    compute_linear_probe_accuracy,
    compute_mlp_probe_accuracy,
    compute_knn_accuracy,
    compute_knn_pca_accuracy,
)
from .probe.signal_metrics import compute_signal_metrics
from .distribution.distribution_metrics import (
    compute_mmd_rbf,
    compute_density_ratio,
    compute_fisher_per_dimension,
)
from .distribution.decomposition_metrics import compute_decomposition_metrics
from .direction.direction_metrics import (
    compute_direction_stability,
    compute_pairwise_diff_consistency,
)
from .direction.multi_direction import compute_multi_direction_accuracy
from .representation import (
    compute_magnitude_metrics,
    compute_sparsity_metrics,
    compute_pair_quality_metrics,
    compute_manifold_metrics,
    compute_noise_baseline_comparison,
    compute_cross_layer_consistency,
    compute_token_position_metrics,
    compute_direction_overlap_metrics,
)
from .direction.representation_metrics import (
    analyze_representation_geometry,
    compute_all_representation_metrics,
)
from .direction.geometry_metrics import compute_geometry_summary

__all__ = [
    "compute_geometry_metrics",
    "generate_metrics_visualizations",
    "compute_signal_strength",
    "compute_linear_probe_accuracy",
    "compute_mlp_probe_accuracy",
    "compute_knn_accuracy",
    "compute_knn_pca_accuracy",
    "compute_signal_metrics",
    "compute_mmd_rbf",
    "compute_density_ratio",
    "compute_fisher_per_dimension",
    "compute_decomposition_metrics",
    "compute_direction_stability",
    "compute_multi_direction_accuracy",
    "compute_pairwise_diff_consistency",
    "compute_magnitude_metrics",
    "compute_sparsity_metrics",
    "compute_pair_quality_metrics",
    "compute_manifold_metrics",
    "compute_noise_baseline_comparison",
    "compute_cross_layer_consistency",
    "compute_token_position_metrics",
    "compute_direction_overlap_metrics",
    "analyze_representation_geometry",
    "compute_all_representation_metrics",
    "compute_geometry_summary",
]
