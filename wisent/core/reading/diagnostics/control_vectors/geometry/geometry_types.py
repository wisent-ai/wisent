"""Geometry detection type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Tuple

from wisent.core.utils.config_tools.constants import (
    DIAG_NUM_COMPONENTS,
    DEFAULT_OPTIMIZATION_STEPS,
    GEO_DIAG_LINEAR_VARIANCE,
    GEO_DIAG_CONE_THRESHOLD,
    MAX_CLUSTERS,
    GEO_DIAG_CLUSTER_SILHOUETTE,
    MANIFOLD_NEIGHBORS,
    GEO_DIAG_MANIFOLD_THRESHOLD,
    GEO_DIAG_SPARSE_THRESHOLD,
    GEO_DIAG_BIMODAL_DIP,
    GEO_DIAG_ORTHOGONAL_THRESHOLD,
)

__all__ = [
    "StructureType",
    "StructureScore",
    "GeometryAnalysisConfig",
    "GeometryAnalysisResult",
]


class StructureType(Enum):
    """Types of geometric structures in activation space."""
    LINEAR = "linear"
    CONE = "cone"
    CLUSTER = "cluster"
    MANIFOLD = "manifold"
    SPARSE = "sparse"
    BIMODAL = "bimodal"
    ORTHOGONAL = "orthogonal"
    UNKNOWN = "unknown"


@dataclass
class StructureScore:
    """Score for a single structure type."""
    structure_type: StructureType
    score: float
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeometryAnalysisConfig:
    """Configuration for comprehensive geometry analysis."""

    num_components: int = DIAG_NUM_COMPONENTS
    """Number of components/directions to analyze."""

    optimization_steps: int = DEFAULT_OPTIMIZATION_STEPS
    """Steps for optimization-based methods."""

    linear_variance_threshold: float = GEO_DIAG_LINEAR_VARIANCE
    """Variance explained threshold for linear structure."""

    cone_threshold: float = GEO_DIAG_CONE_THRESHOLD
    """Cone score threshold."""

    max_clusters: int = MAX_CLUSTERS
    """Maximum number of clusters to try."""

    cluster_silhouette_threshold: float = GEO_DIAG_CLUSTER_SILHOUETTE
    """Silhouette score threshold for cluster detection."""

    manifold_neighbors: int = MANIFOLD_NEIGHBORS
    """Number of neighbors for manifold analysis."""

    manifold_threshold: float = GEO_DIAG_MANIFOLD_THRESHOLD
    """Score threshold for manifold structure."""

    sparse_threshold: float = GEO_DIAG_SPARSE_THRESHOLD
    """Fraction of active dimensions threshold."""

    bimodal_dip_threshold: float = GEO_DIAG_BIMODAL_DIP
    """P-value threshold for dip test."""

    orthogonal_threshold: float = GEO_DIAG_ORTHOGONAL_THRESHOLD
    """Max correlation for orthogonal subspaces."""

    use_universal_thresholds: bool = True
    """Whether to use thresholds tuned for universal subspace theory."""


@dataclass
class GeometryAnalysisResult:
    """Results from comprehensive geometry analysis."""

    best_structure: StructureType
    """The structure type that best fits the data."""

    best_score: float
    """Score of the best-fitting structure."""

    all_scores: Dict[str, StructureScore]
    """Scores for all analyzed structure types."""

    recommendation: str
    """Recommended steering method based on geometry."""

    details: Dict[str, Any] = field(default_factory=dict)
    """Additional analysis details."""

    def get_ranking(self) -> List[Tuple[StructureType, float]]:
        """Get structures ranked by score."""
        return sorted(
            [(s.structure_type, s.score) for s in self.all_scores.values()],
            key=lambda x: x[1],
            reverse=True
        )
