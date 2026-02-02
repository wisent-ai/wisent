"""Geometry detection type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Tuple

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

    num_components: int = 5
    """Number of components/directions to analyze."""

    optimization_steps: int = 100
    """Steps for optimization-based methods."""

    linear_variance_threshold: float = 0.85
    """Variance explained threshold for linear structure."""

    cone_threshold: float = 0.65
    """Cone score threshold."""

    max_clusters: int = 5
    """Maximum number of clusters to try."""

    cluster_silhouette_threshold: float = 0.55
    """Silhouette score threshold for cluster detection."""

    manifold_neighbors: int = 10
    """Number of neighbors for manifold analysis."""

    manifold_threshold: float = 0.70
    """Score threshold for manifold structure."""

    sparse_threshold: float = 0.1
    """Fraction of active dimensions threshold."""

    bimodal_dip_threshold: float = 0.05
    """P-value threshold for dip test."""

    orthogonal_threshold: float = 0.12
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
