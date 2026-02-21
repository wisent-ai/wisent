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

__all__ = [
    "VectorQualityConfig",
    "VectorQualityReport",
    "run_vector_quality_diagnostics",
]


@dataclass(slots=True)
class VectorQualityConfig:
    """Thresholds for vector quality diagnostics.
    
    Based on empirical analysis with Italian/English steering vectors:
    - Good vectors typically show: convergence > 0.9, CV > 0.7, SNR > 50, PC1 > 40%
    - Thresholds are set to catch clearly problematic vectors while allowing reasonable ones
    """
    
    # Convergence thresholds (similarity between 50% and 100% of pairs)
    convergence_critical: float = 0.5
    convergence_warning: float = 0.75
    
    # Cross-validation score thresholds
    cv_score_critical: float = 0.3
    cv_score_warning: float = 0.6
    
    # Signal-to-noise ratio thresholds
    snr_critical: float = 10.0
    snr_warning: float = 30.0
    
    # PCA PC1 variance explained thresholds (in high-dim space, 10-20% can be good)
    pca_variance_critical: float = 0.05
    pca_variance_warning: float = 0.15
    
    # Pair alignment thresholds (std of alignments)
    alignment_std_warning: float = 0.30
    alignment_min_critical: float = 0.2
    
    # Clustering silhouette thresholds (can be low in high-dim space)
    silhouette_critical: float = -0.1
    silhouette_warning: float = 0.05
    
    # Minimum pairs required for quality analysis
    min_pairs_for_analysis: int = 5
    
    # Number of CV folds
    cv_folds: int = 5


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
    overall_quality: str = "unknown"
    issues: List[DiagnosticsIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()


def _create_vector_from_diffs(diffs: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Create steering vector by averaging difference vectors."""
    vec = diffs.mean(dim=0)
    if normalize and torch.norm(vec) > 1e-8:
        vec = vec / torch.norm(vec)
    return vec


def _compute_convergence(
    difference_vectors: torch.Tensor,
    config: VectorQualityConfig
) -> Tuple[float, float, float, List[DiagnosticsIssue]]:
    """Check if vector converges as more pairs are added."""
    issues = []
    n = len(difference_vectors)
    
    if n < config.min_pairs_for_analysis:
        return None, None, None, issues
    
    # Create vectors from 50%, 75%, and 100% of pairs
    n_50 = max(1, n // 2)
    n_75 = max(1, int(n * 0.75))
    
    vec_50 = _create_vector_from_diffs(difference_vectors[:n_50])
    vec_75 = _create_vector_from_diffs(difference_vectors[:n_75])
    vec_100 = _create_vector_from_diffs(difference_vectors)
    
    sim_50_100 = _cosine_similarity(vec_50, vec_100)
    sim_75_100 = _cosine_similarity(vec_75, vec_100)
    convergence = sim_50_100  # Primary metric
    
    if convergence < config.convergence_critical:
        issues.append(DiagnosticsIssue(
            metric="convergence",
            severity="critical",
            message=f"Vector not converged: 50% vs 100% similarity = {convergence:.3f} (< {config.convergence_critical})",
            details={"sim_50_100": sim_50_100, "sim_75_100": sim_75_100}
        ))
    elif convergence < config.convergence_warning:
        issues.append(DiagnosticsIssue(
            metric="convergence",
            severity="warning",
            message=f"Vector may not be fully converged: 50% vs 100% similarity = {convergence:.3f}",
            details={"sim_50_100": sim_50_100, "sim_75_100": sim_75_100}
        ))
    
    return convergence, sim_50_100, sim_75_100, issues


def _compute_cross_validation(
    difference_vectors: torch.Tensor,
    config: VectorQualityConfig
) -> Tuple[float, float, List[float], List[DiagnosticsIssue]]:
    """5-fold cross-validation to test generalization."""
    issues = []
    n = len(difference_vectors)
    
    if n < config.min_pairs_for_analysis:
        return None, None, [], issues
    
    n_folds = min(config.cv_folds, n)
    fold_size = n // n_folds
    
    if fold_size < 1:
        return None, None, [], issues
    
    cv_scores = []
    indices = list(range(n))
    
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
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
    cv_std = statistics.stdev(cv_scores) if len(cv_scores) > 1 else 0.0
    
    if cv_mean < config.cv_score_critical:
        issues.append(DiagnosticsIssue(
            metric="cross_validation",
            severity="critical",
            message=f"Poor cross-validation score: {cv_mean:.3f} (< {config.cv_score_critical}). Vector does not generalize.",
            details={"cv_mean": cv_mean, "cv_std": cv_std}
        ))
    elif cv_mean < config.cv_score_warning:
        issues.append(DiagnosticsIssue(
            metric="cross_validation",
            severity="warning",
            message=f"Low cross-validation score: {cv_mean:.3f}. Vector may not generalize well.",
            details={"cv_mean": cv_mean, "cv_std": cv_std}
        ))
    
    return cv_mean, cv_std, cv_scores, issues




# Re-exports from split modules
from wisent.core.contrastive_pairs.diagnostics.analysis._vector_quality_helpers import (
    _compute_snr,
    _compute_pca,
    _compute_pair_alignment,
    _compute_clustering,
    _compute_held_out_transfer,
    _compute_cv_classification,
    _compute_cohens_d,
)
from wisent.core.contrastive_pairs.diagnostics.analysis._vector_quality_runner import (
    run_vector_quality_diagnostics,
)
