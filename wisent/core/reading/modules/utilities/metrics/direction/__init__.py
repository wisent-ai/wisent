"""Direction-based metrics."""
from .direction_metrics import (
    compute_direction_stability,
    compute_pairwise_diff_consistency,
)
from .multi_direction import compute_multi_direction_accuracy
from .geometry_metrics import compute_geometry_summary
from .representation_metrics import (
    analyze_representation_geometry,
    compute_all_representation_metrics,
)

# Re-export from representation subdirectory for backward compatibility
from ..representation import (
    compute_magnitude_metrics,
    compute_sparsity_metrics,
    compute_pair_quality_metrics,
    compute_manifold_metrics,
    compute_noise_baseline_comparison,
    compute_cross_layer_consistency,
    compute_token_position_metrics,
    compute_direction_overlap_metrics,
)

__all__ = [
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
