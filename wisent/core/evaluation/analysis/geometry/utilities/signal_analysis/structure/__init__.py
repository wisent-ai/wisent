"""Activation structure analysis."""

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
from .transformer_analysis import (
    TransformerComponent,
    analyze_transformer_components,
    get_component_hook_points,
    compare_components_for_benchmark,
    compare_concept_granularity,
)

__all__ = [
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
    # Transformer analysis
    "TransformerComponent",
    "analyze_transformer_components",
    "get_component_hook_points",
    "compare_components_for_benchmark",
    "compare_concept_granularity",
]
