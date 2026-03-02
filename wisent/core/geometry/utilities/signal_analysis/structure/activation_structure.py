"""
Activation space structure analysis.

Comprehensive analysis of activation point clouds: shape, cone/sphere fit,
manifold dimension, clustering, density, topology, two-cloud relationships.

Implementation split into _cloud_metrics.py and _relationship_metrics.py
to keep files under 300 lines. This module re-exports all public functions.
"""

from ._cloud_metrics import (
    compute_cloud_shape,
    compute_cone_fit,
    compute_sphere_fit,
    compute_manifold_dimension,
    compute_cluster_structure,
    compute_density_structure,
    compute_topology_indicators,
)
from ._relationship_metrics import (
    compute_two_cloud_relationship,
    compute_relative_position,
    analyze_activation_structure,
)

__all__ = [
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
]
