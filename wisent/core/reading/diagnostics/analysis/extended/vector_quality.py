"""Quality diagnostics for steering vectors.

Analyzes steering vector quality using multiple metrics:
1. Convergence - Is the vector stable with different subsets of pairs?
2. Pair alignment - Which pairs contribute well vs poorly?
3. Signal-to-noise ratio - How clean is the signal?
4. PCA dominance - Is there a clear dominant direction?
5. Cross-validation - Does the vector generalize to held-out pairs?
6. Clustering separation - How well do positive/negative activations separate?
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np

from ..base import DiagnosticsIssue, DiagnosticsReport, MetricReport
"""Formerly imported CV_FOLDS is now a required parameter."""

__all__ = [
    "VectorQualityConfig",
    "VectorQualityReport",
    "run_vector_quality_diagnostics",
]


@dataclass(slots=True)
class VectorQualityConfig:
    """Configuration for vector quality diagnostics thresholds."""

    vq_convergence_critical: float = None
    vq_convergence_warning: float = None
    vq_cv_score_critical: float = None
    vq_cv_score_warning: float = None
    vq_snr_critical: float = None
    vq_snr_warning: float = None
    vq_pca_variance_critical: float = None
    vq_pca_variance_warning: float = None
    vq_alignment_std_warning: float = None
    vq_alignment_min_critical: float = None
    vq_silhouette_critical: float = None
    vq_silhouette_warning: float = None
    vq_min_pairs: int = None
    vq_warnings_fair_threshold: int = None

    def __post_init__(self):
        """Validate all fields are provided."""
        for name in self.__slots__:
            if getattr(self, name) is None:
                raise ValueError(f"{name} is required in VectorQualityConfig")


@dataclass
class VectorQualityReport:
    """Detailed quality report for a steering vector."""
    
    # Convergence
    convergence_score: Optional[float] = None
    convergence_50_vs_100: Optional[float] = None
    convergence_75_vs_100: Optional[float] = None
    
    # Cross-validation
    cv_score_mean: Optional[float] = None
    cv_score_std: Optional[float] = None
    cv_scores: List[float] = field(default_factory=list)
    
    # Signal-to-noise
    snr: Optional[float] = None
    signal: Optional[float] = None
    noise: Optional[float] = None
    
    # PCA
    pca_pc1_variance: Optional[float] = None
    pca_pc2_variance: Optional[float] = None
    pca_cumulative_3: Optional[float] = None
    
    # Pair alignment
    alignment_mean: Optional[float] = None
    alignment_std: Optional[float] = None
    alignment_min: Optional[float] = None
    alignment_max: Optional[float] = None
    outlier_pairs: List[Tuple[int, float, str]] = field(default_factory=list)
    
    # Clustering
    silhouette_score: Optional[float] = None
    
    # Held-out transfer (train on 80%, test alignment on 20%)
    held_out_transfer: Optional[float] = None
    
    # CV classification (classification accuracy, not just alignment)
    cv_classification_accuracy: Optional[float] = None
    
    # Cohen's d (effect size)
    cohens_d: Optional[float] = None
    
    # Summary
    overall_quality: Optional[str] = None
    issues: List[DiagnosticsIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()


def _create_vector_from_diffs(diffs: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Create steering vector by averaging difference vectors."""
    from wisent.core.utils.config_tools.constants import NORM_EPS, PARSER_DEFAULT_LAYER_START
    vec = diffs.mean(dim=PARSER_DEFAULT_LAYER_START)
    if normalize and torch.norm(vec) > NORM_EPS:
        vec = vec / torch.norm(vec)
    return vec


def _compute_convergence(
    difference_vectors: torch.Tensor,
    config: VectorQualityConfig
) -> Tuple[float, float, float, List[DiagnosticsIssue]]:
    """Check if vector converges as more pairs are added."""
    from wisent.core.utils.config_tools.constants import COMBO_OFFSET, COMBO_BASE
    issues = []
    n = len(difference_vectors)

    if n < config.vq_min_pairs:
        return None, None, None, issues

    # Create vectors from half, three-quarter, and full sets of pairs
    n_half = max(COMBO_OFFSET, n // COMBO_BASE)
    n_three_quarter = max(COMBO_OFFSET, int(n * config.vq_convergence_warning))

    vec_half = _create_vector_from_diffs(difference_vectors[:n_half])
    vec_three_quarter = _create_vector_from_diffs(difference_vectors[:n_three_quarter])
    vec_full = _create_vector_from_diffs(difference_vectors)

    sim_half_full = _cosine_similarity(vec_half, vec_full)
    sim_three_quarter_full = _cosine_similarity(vec_three_quarter, vec_full)
    convergence = sim_half_full  # Primary metric

    if convergence < config.vq_convergence_critical:
        issues.append(DiagnosticsIssue(
            metric="convergence",
            severity="critical",
            message=f"Vector not converged: half vs full similarity = {convergence:.3f} (< {config.vq_convergence_critical})",
            details={"sim_half_full": sim_half_full, "sim_three_quarter_full": sim_three_quarter_full}
        ))
    elif convergence < config.vq_convergence_warning:
        issues.append(DiagnosticsIssue(
            metric="convergence",
            severity="warning",
            message=f"Vector may not be fully converged: half vs full similarity = {convergence:.3f}",
            details={"sim_half_full": sim_half_full, "sim_three_quarter_full": sim_three_quarter_full}
        ))

    return convergence, sim_half_full, sim_three_quarter_full, issues


def _compute_cross_validation(
    difference_vectors: torch.Tensor,
    config: VectorQualityConfig,
    default_score: float,
    *,
    cv_folds: int,
) -> Tuple[float, float, List[float], List[DiagnosticsIssue]]:
    """Cross-validation to test generalization."""
    from wisent.core.utils.config_tools.constants import COMBO_OFFSET
    issues = []
    n = len(difference_vectors)

    if n < config.vq_min_pairs:
        return None, None, [], issues

    n_folds = min(cv_folds, n)
    fold_size = n // n_folds

    if fold_size < COMBO_OFFSET:
        return None, None, [], issues

    cv_scores = []
    indices = list(range(n))

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - COMBO_OFFSET else n
        test_indices = indices[test_start:test_end]
        train_indices = [i for i in indices if i not in test_indices]

        if not train_indices or not test_indices:
            continue

        train_vec = _create_vector_from_diffs(difference_vectors[train_indices])
        test_sims = [_cosine_similarity(difference_vectors[i], train_vec) for i in test_indices]
        cv_scores.append(statistics.mean(test_sims))

    if not cv_scores:
        return None, None, [], issues

    cv_mean = statistics.mean(cv_scores)
    cv_std = statistics.stdev(cv_scores) if len(cv_scores) > COMBO_OFFSET else default_score

    if cv_mean < config.vq_cv_score_critical:
        issues.append(DiagnosticsIssue(
            metric="cross_validation",
            severity="critical",
            message=f"Poor cross-validation score: {cv_mean:.3f} (< {config.vq_cv_score_critical}). Vector does not generalize.",
            details={"cv_mean": cv_mean, "cv_std": cv_std}
        ))
    elif cv_mean < config.vq_cv_score_warning:
        issues.append(DiagnosticsIssue(
            metric="cross_validation",
            severity="warning",
            message=f"Low cross-validation score: {cv_mean:.3f}. Vector may not generalize well.",
            details={"cv_mean": cv_mean, "cv_std": cv_std}
        ))

    return cv_mean, cv_std, cv_scores, issues




# Re-exports from split modules
from wisent.core.reading.diagnostics.analysis._vector_quality_helpers import (
    _compute_snr,
    _compute_pca,
    _compute_pair_alignment,
    _compute_clustering,
    _compute_held_out_transfer,
    _compute_cv_classification,
    _compute_cohens_d,
)
from wisent.core.reading.diagnostics.analysis._vector_quality_runner import (
    run_vector_quality_diagnostics,
)
