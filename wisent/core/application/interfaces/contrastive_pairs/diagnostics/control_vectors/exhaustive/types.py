"""Exhaustive layer combination analysis type definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from ..geometry import StructureType


@dataclass
class ExhaustiveCombinationResult:
    """Result for a single layer combination."""
    layers: Tuple[int, ...]
    best_structure: StructureType
    best_score: float
    all_scores: Dict[str, float]


@dataclass
class ExhaustiveGeometryAnalysisResult:
    """Results from exhaustive layer combination analysis."""

    total_combinations: int
    """Total number of combinations tested."""

    all_results: List[ExhaustiveCombinationResult]
    """All results, sorted by best_score descending."""

    best_combination: Tuple[int, ...]
    """Layer combination with highest score."""

    best_score: float
    """Highest score achieved."""

    best_structure: StructureType
    """Structure type at best combination."""

    top_10: List[ExhaustiveCombinationResult]
    """Top 10 combinations."""

    single_layer_best: int
    """Best single layer."""

    single_layer_best_score: float
    """Score of best single layer."""

    combination_beats_single: bool
    """Whether any multi-layer combination beats best single layer."""

    improvement_over_single: float
    """How much best combination improves over best single layer."""

    patterns: Dict[str, Any]
    """Discovered patterns (layer frequency in top combinations, etc.)."""

    recommendation: str
    """Final recommendation."""


__all__ = [
    "ExhaustiveCombinationResult",
    "ExhaustiveGeometryAnalysisResult",
]
