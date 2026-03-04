"""Main runner for vector quality diagnostics."""
from __future__ import annotations

import statistics
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np

from ..base import DiagnosticsIssue, DiagnosticsReport, MetricReport
from wisent.core.utils.config_tools.constants import (
    PARSER_DEFAULT_LAYER_START, COMBO_OFFSET,
)
from wisent.core.reading.diagnostics.analysis.vector_quality import (
    VectorQualityConfig,
    VectorQualityReport,
    _cosine_similarity,
    _create_vector_from_diffs,
    _compute_convergence,
    _compute_cross_validation,
)
from wisent.core.reading.diagnostics.analysis._vector_quality_helpers import (
    _compute_snr,
    _compute_pca,
    _compute_pair_alignment,
    _compute_clustering,
    _compute_held_out_transfer,
    _compute_cv_classification,
    _compute_cohens_d,
)

def run_vector_quality_diagnostics(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    min_clusters: int,
    *, adaptive_cv_min_folds: int,
    pair_prompts: Optional[List[str]] = None,
    config: Optional[VectorQualityConfig] = None,
) -> Tuple[VectorQualityReport, DiagnosticsReport]:
    """
    Run comprehensive quality diagnostics on steering vector training data.
    
    Args:
        positive_activations: Tensor of shape [n_pairs, hidden_dim] for positive examples
        negative_activations: Tensor of shape [n_pairs, hidden_dim] for negative examples
        pair_prompts: Optional list of prompts for each pair (for reporting outliers)
        config: Quality thresholds configuration
        
    Returns:
        Tuple of (VectorQualityReport, DiagnosticsReport)
    """
    cfg = config or VectorQualityConfig()
    report = VectorQualityReport()
    all_issues: List[DiagnosticsIssue] = []
    
    # Compute difference vectors
    difference_vectors = positive_activations - negative_activations
    final_vector = _create_vector_from_diffs(difference_vectors)
    n_pairs = len(difference_vectors)
    
    # 1. Convergence
    conv, sim_50, sim_75, issues = _compute_convergence(difference_vectors, cfg)
    report.convergence_score = conv
    report.convergence_50_vs_100 = sim_50
    report.convergence_75_vs_100 = sim_75
    all_issues.extend(issues)
    
    # 2. Cross-validation
    cv_mean, cv_std, cv_scores, issues = _compute_cross_validation(difference_vectors, cfg)
    report.cv_score_mean = cv_mean
    report.cv_score_std = cv_std
    report.cv_scores = cv_scores
    all_issues.extend(issues)
    
    # 3. Signal-to-noise ratio
    snr, signal, noise, issues = _compute_snr(positive_activations, negative_activations, cfg)
    report.snr = snr
    report.signal = signal
    report.noise = noise
    all_issues.extend(issues)
    
    # 4. PCA
    pc1, pc2, cum3, issues = _compute_pca(difference_vectors, cfg)
    report.pca_pc1_variance = pc1
    report.pca_pc2_variance = pc2
    report.pca_cumulative_3 = cum3
    all_issues.extend(issues)
    
    # 5. Pair alignment
    mean_a, std_a, min_a, max_a, outliers, issues = _compute_pair_alignment(
        difference_vectors, final_vector, pair_prompts, cfg
    )
    report.alignment_mean = mean_a
    report.alignment_std = std_a
    report.alignment_min = min_a
    report.alignment_max = max_a
    report.outlier_pairs = outliers
    all_issues.extend(issues)
    
    # 6. Clustering
    silhouette, issues = _compute_clustering(positive_activations, negative_activations, cfg, min_clusters=min_clusters)
    report.silhouette_score = silhouette
    all_issues.extend(issues)
    
    # 7. Held-out transfer
    report.held_out_transfer = _compute_held_out_transfer(difference_vectors, cfg)
    
    # 8. CV classification accuracy
    report.cv_classification_accuracy = _compute_cv_classification(
        positive_activations, negative_activations, cfg, adaptive_cv_min_folds=adaptive_cv_min_folds
    )
    
    # 9. Cohen's d
    report.cohens_d = _compute_cohens_d(positive_activations, negative_activations, min_clusters=min_clusters)
    
    # Determine overall quality
    critical_count = sum(COMBO_OFFSET for i in all_issues if i.severity == "critical")
    warning_count = sum(COMBO_OFFSET for i in all_issues if i.severity == "warning")

    if critical_count > PARSER_DEFAULT_LAYER_START:
        report.overall_quality = "poor"
    elif warning_count >= cfg.vq_warnings_fair_threshold:
        report.overall_quality = "fair"
    elif warning_count > PARSER_DEFAULT_LAYER_START:
        report.overall_quality = "good"
    else:
        report.overall_quality = "excellent"
    
    # Generate recommendations
    if report.outlier_pairs:
        report.recommendations.append(
            f"Remove {len(report.outlier_pairs)} outlier pair(s) with low alignment and retrain."
        )
    if report.convergence_score and report.convergence_score < cfg.vq_convergence_warning:
        report.recommendations.append(
            "Add more contrastive pairs to improve vector stability."
        )
    if report.snr and report.snr < cfg.vq_snr_warning:
        report.recommendations.append(
            "Use more distinct positive/negative examples to increase signal-to-noise ratio."
        )
    if report.pca_pc1_variance and report.pca_pc1_variance < cfg.vq_pca_variance_warning:
        report.recommendations.append(
            "Pairs may capture multiple concepts. Focus pairs on a single, clear distinction."
        )
    
    report.issues = all_issues
    
    # Create DiagnosticsReport
    summary = {
        "num_pairs": n_pairs,
        "overall_quality": report.overall_quality,
        "convergence_score": report.convergence_score,
        "cv_score": report.cv_score_mean,
        "snr": report.snr,
        "pca_pc1_variance": report.pca_pc1_variance,
        "silhouette": report.silhouette_score,
        "held_out_transfer": report.held_out_transfer,
        "cv_classification_accuracy": report.cv_classification_accuracy,
        "cohens_d": report.cohens_d,
        "num_outliers": len(report.outlier_pairs),
    }
    
    metric_report = MetricReport(name="vector_quality", summary=summary, issues=all_issues)
    diagnostics_report = DiagnosticsReport.from_metrics([metric_report])
    
    return report, diagnostics_report
