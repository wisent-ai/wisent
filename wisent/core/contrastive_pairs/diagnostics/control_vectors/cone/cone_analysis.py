"""Cone structure analysis for activation spaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List

import torch

from .cone_helpers import (
    compute_pca_directions,
    discover_cone_directions,
    compute_cone_explained_variance,
    check_half_space_consistency,
    test_positive_combinations,
    compute_cosine_similarity_matrix,
    compute_avg_off_diagonal,
    compute_separation_scores,
    compute_cone_score,
)

__all__ = [
    "ConeAnalysisConfig",
    "ConeAnalysisResult",
    "check_cone_structure",
]


@dataclass
class ConeAnalysisConfig:
    """Configuration for cone structure analysis."""

    num_directions: int = 5
    """Number of directions to discover in the cone."""

    optimization_steps: int = 100
    """Gradient steps for cone direction optimization."""

    learning_rate: float = 0.01
    """Learning rate for optimization."""

    min_cosine_similarity: float = 0.2
    """Minimum cosine similarity between cone directions."""

    max_cosine_similarity: float = 0.95
    """Maximum cosine similarity (avoid redundant directions)."""

    pca_components: int = 5
    """Number of PCA components to compare against."""

    cone_threshold: float = 0.7
    """Threshold for cone_score to declare cone structure exists."""


@dataclass
class ConeAnalysisResult:
    """Results from cone structure analysis."""

    has_cone_structure: bool
    """Whether a cone structure was detected."""

    cone_score: float
    """Score from 0-1 indicating cone-ness (1 = perfect cone)."""

    pca_explained_variance: float
    """Variance explained by PCA directions."""

    cone_explained_variance: float
    """Variance explained by cone directions."""

    num_directions_found: int
    """Number of valid cone directions discovered."""

    direction_cosine_similarities: List[List[float]]
    """Pairwise cosine similarities between discovered directions."""

    avg_cosine_similarity: float
    """Average pairwise cosine similarity (high = more cone-like)."""

    half_space_consistency: float
    """Fraction of directions in same half-space as primary."""

    separation_scores: List[float]
    """Per-direction separation between positive and negative activations."""

    positive_combination_score: float
    """How well positive activations can be represented as positive combinations."""

    details: Dict[str, Any] = field(default_factory=dict)
    """Additional diagnostic details."""


def check_cone_structure(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    config: ConeAnalysisConfig | None = None,
) -> ConeAnalysisResult:
    """
    Analyze whether activations form a cone structure vs linear subspace.

    A cone structure implies:
    1. Multiple directions mediate the behavior (not just one)
    2. These directions are positively correlated (same half-space)
    3. The behavior can be achieved by positive combinations of directions
    4. Cone explains variance better than or comparable to PCA
    """
    cfg = config or ConeAnalysisConfig()

    pos_tensor = pos_activations.detach().float()
    neg_tensor = neg_activations.detach().float()

    if pos_tensor.dim() == 1:
        pos_tensor = pos_tensor.unsqueeze(0)
    if neg_tensor.dim() == 1:
        neg_tensor = neg_tensor.unsqueeze(0)

    pca_directions, pca_explained = compute_pca_directions(
        pos_tensor, neg_tensor, n_components=cfg.pca_components
    )

    cone_directions, cone_metadata = discover_cone_directions(
        pos_tensor, neg_tensor,
        num_directions=cfg.num_directions,
        optimization_steps=cfg.optimization_steps,
        learning_rate=cfg.learning_rate,
        min_cos_sim=cfg.min_cosine_similarity,
        max_cos_sim=cfg.max_cosine_similarity,
    )

    cone_explained = compute_cone_explained_variance(
        pos_tensor, neg_tensor, cone_directions
    )
    half_space_score = check_half_space_consistency(cone_directions)
    pos_combo_score = test_positive_combinations(
        pos_tensor, neg_tensor, cone_directions
    )
    cos_sim_matrix = compute_cosine_similarity_matrix(cone_directions)
    avg_cos_sim = compute_avg_off_diagonal(cos_sim_matrix)
    separation_scores = compute_separation_scores(
        pos_tensor, neg_tensor, cone_directions
    )
    final_score = compute_cone_score(
        pca_explained=pca_explained,
        cone_explained=cone_explained,
        half_space_score=half_space_score,
        avg_cos_sim=avg_cos_sim,
        pos_combo_score=pos_combo_score,
        separation_scores=separation_scores,
    )

    return ConeAnalysisResult(
        has_cone_structure=final_score >= cfg.cone_threshold,
        cone_score=final_score,
        pca_explained_variance=pca_explained,
        cone_explained_variance=cone_explained,
        num_directions_found=cone_directions.shape[0],
        direction_cosine_similarities=cos_sim_matrix.tolist(),
        avg_cosine_similarity=avg_cos_sim,
        half_space_consistency=half_space_score,
        separation_scores=separation_scores,
        positive_combination_score=pos_combo_score,
        details={
            "config": cfg.__dict__,
            "cone_metadata": cone_metadata,
            "pca_directions_shape": list(pca_directions.shape),
            "cone_directions_shape": list(cone_directions.shape),
        }
    )
