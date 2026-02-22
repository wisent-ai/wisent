"""Multi-layer geometry analysis type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from ..geometry import StructureType, GeometryAnalysisResult


@dataclass
class MultiLayerGeometryConfig:
    """Configuration for multi-layer geometry analysis."""

    num_components: int = 5
    optimization_steps: int = 50
    combination_method: str = "concat"  # "concat", "mean", "weighted"
    analyze_per_layer: bool = True
    analyze_combined: bool = True
    analyze_subsets: bool = True  # early/middle/late
    analyze_pairs: bool = True  # all pairs of layers
    analyze_adjacent: bool = True  # adjacent layer pairs
    analyze_skip: bool = True  # every other layer, every third, etc.
    analyze_custom: Optional[List[List[int]]] = None  # custom layer combinations
    max_pair_combinations: int = 50  # limit number of pair combinations to analyze


@dataclass
class LayerGeometryResult:
    """Geometry result for a single layer."""
    layer: int
    best_structure: StructureType
    best_score: float
    all_scores: Dict[str, float]


@dataclass
class MultiLayerGeometryResult:
    """Results from multi-layer geometry analysis."""

    per_layer_results: Dict[int, LayerGeometryResult]
    """Geometry analysis for each individual layer."""

    combined_result: Optional[GeometryAnalysisResult]
    """Geometry analysis for all layers combined."""

    layer_subset_results: Dict[str, GeometryAnalysisResult]
    """Geometry analysis for layer subsets (e.g., 'early', 'middle', 'late')."""

    layer_pair_results: Dict[str, GeometryAnalysisResult]
    """Geometry analysis for pairs of layers (e.g., 'L1+L5', 'L2+L8')."""

    adjacent_pair_results: Dict[str, GeometryAnalysisResult]
    """Geometry analysis for adjacent layer pairs (e.g., 'L1+L2', 'L2+L3')."""

    skip_results: Dict[str, GeometryAnalysisResult]
    """Geometry analysis for skip patterns (e.g., 'every_2nd', 'every_3rd')."""

    custom_results: Dict[str, GeometryAnalysisResult]
    """Geometry analysis for custom layer combinations."""

    best_single_layer: int
    """Layer with strongest structure detection."""

    best_single_layer_structure: StructureType
    """Structure type detected at best single layer."""

    best_single_layer_score: float
    """Score at best single layer."""

    best_combination: Optional[str]
    """Best performing layer combination (if better than single layer)."""

    best_combination_score: float
    """Score of best combination."""

    best_combination_structure: Optional[StructureType]
    """Structure type detected at best combination."""

    combined_vs_single: str
    """Whether combined layers improve over single layer."""

    layer_agreement: float
    """How much layers agree on structure type (0-1)."""

    structure_by_depth: Dict[str, List[float]]
    """How each structure score varies by layer depth."""

    all_combinations_ranked: List[Tuple[str, float, StructureType]]
    """All combinations ranked by score: (name, score, structure)."""

    recommendation: str
    """Recommendation based on multi-layer analysis."""


__all__ = [
    "MultiLayerGeometryConfig",
    "LayerGeometryResult",
    "MultiLayerGeometryResult",
]
