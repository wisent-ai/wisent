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

from .base import DiagnosticsIssue, DiagnosticsReport, MetricReport

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


def _compute_snr(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    config: VectorQualityConfig
) -> Tuple[float, float, float, List[DiagnosticsIssue]]:
    """Compute signal-to-noise ratio."""
    issues = []
    
    mean_pos = positive_activations.mean(dim=0)
    mean_neg = negative_activations.mean(dim=0)
    signal = torch.norm(mean_pos - mean_neg).item()
    
    noise_pos = positive_activations.std(dim=0).mean().item()
    noise_neg = negative_activations.std(dim=0).mean().item()
    noise = (noise_pos + noise_neg) / 2
    
    snr = signal / (noise + 1e-8)
    
    if snr < config.snr_critical:
        issues.append(DiagnosticsIssue(
            metric="snr",
            severity="critical",
            message=f"Very low signal-to-noise ratio: {snr:.2f} (< {config.snr_critical}). Pairs may not capture distinct concept.",
            details={"snr": snr, "signal": signal, "noise": noise}
        ))
    elif snr < config.snr_warning:
        issues.append(DiagnosticsIssue(
            metric="snr",
            severity="warning",
            message=f"Low signal-to-noise ratio: {snr:.2f}. Consider using cleaner pairs.",
            details={"snr": snr, "signal": signal, "noise": noise}
        ))
    
    return snr, signal, noise, issues


def _compute_pca(
    difference_vectors: torch.Tensor,
    config: VectorQualityConfig
) -> Tuple[float, float, float, List[DiagnosticsIssue]]:
    """Compute PCA variance explained by top components."""
    issues = []
    n = len(difference_vectors)
    
    if n < 3:
        return None, None, None, issues
    
    try:
        from sklearn.decomposition import PCA
        
        n_components = min(5, n - 1)
        pca = PCA(n_components=n_components)
        # Convert to float32 for sklearn compatibility (BFloat16 not supported)
        pca.fit(difference_vectors.float().numpy())
        
        pc1_var = pca.explained_variance_ratio_[0]
        pc2_var = pca.explained_variance_ratio_[1] if n_components > 1 else 0.0
        cumulative_3 = sum(pca.explained_variance_ratio_[:min(3, n_components)])
        
        if pc1_var < config.pca_variance_critical:
            issues.append(DiagnosticsIssue(
                metric="pca",
                severity="critical",
                message=f"No dominant direction: PC1 explains only {pc1_var*100:.1f}% variance (< {config.pca_variance_critical*100}%).",
                details={"pc1_variance": pc1_var, "pc2_variance": pc2_var}
            ))
        elif pc1_var < config.pca_variance_warning:
            issues.append(DiagnosticsIssue(
                metric="pca",
                severity="warning",
                message=f"Weak dominant direction: PC1 explains {pc1_var*100:.1f}% variance.",
                details={"pc1_variance": pc1_var, "pc2_variance": pc2_var}
            ))
        
        return pc1_var, pc2_var, cumulative_3, issues
    except ImportError:
        return None, None, None, issues


def _compute_pair_alignment(
    difference_vectors: torch.Tensor,
    final_vector: torch.Tensor,
    pair_prompts: Optional[List[str]],
    config: VectorQualityConfig
) -> Tuple[float, float, float, float, List[Tuple[int, float, str]], List[DiagnosticsIssue]]:
    """Analyze how well each pair aligns with the final vector."""
    issues = []
    outliers = []
    
    n = len(difference_vectors)
    if n < 2:
        return None, None, None, None, [], issues
    
    alignments = []
    for i, diff in enumerate(difference_vectors):
        alignment = _cosine_similarity(diff, final_vector)
        alignments.append(alignment)
        
        prompt = pair_prompts[i] if pair_prompts and i < len(pair_prompts) else f"pair_{i}"
        
        if alignment < config.alignment_min_critical:
            outliers.append((i, alignment, prompt))
    
    mean_align = statistics.mean(alignments)
    std_align = statistics.stdev(alignments) if len(alignments) > 1 else 0.0
    min_align = min(alignments)
    max_align = max(alignments)
    
    if outliers:
        issues.append(DiagnosticsIssue(
            metric="pair_alignment",
            severity="warning",
            message=f"{len(outliers)} outlier pair(s) with alignment < {config.alignment_min_critical}. Consider removing them.",
            details={"outliers": [(i, a, p) for i, a, p in outliers]}
        ))
    
    if std_align > config.alignment_std_warning:
        issues.append(DiagnosticsIssue(
            metric="pair_alignment",
            severity="warning",
            message=f"High variance in pair alignments (std={std_align:.3f}). Pairs may capture different concepts.",
            details={"mean": mean_align, "std": std_align}
        ))
    
    return mean_align, std_align, min_align, max_align, outliers, issues


def _compute_clustering(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    config: VectorQualityConfig
) -> Tuple[float, List[DiagnosticsIssue]]:
    """Compute clustering separation using silhouette score."""
    issues = []
    
    n_pos = len(positive_activations)
    n_neg = len(negative_activations)
    
    if n_pos < 2 or n_neg < 2:
        return None, issues
    
    try:
        from sklearn.metrics import silhouette_score
        
        all_activations = torch.cat([positive_activations, negative_activations], dim=0).float().numpy()
        labels = [0] * n_pos + [1] * n_neg
        
        silhouette = silhouette_score(all_activations, labels)
        
        if silhouette < config.silhouette_critical:
            issues.append(DiagnosticsIssue(
                metric="clustering",
                severity="critical",
                message=f"Very poor class separation: silhouette={silhouette:.3f} (< {config.silhouette_critical}). Positive/negative activations overlap significantly.",
                details={"silhouette": silhouette}
            ))
        elif silhouette < config.silhouette_warning:
            issues.append(DiagnosticsIssue(
                metric="clustering",
                severity="warning",
                message=f"Weak class separation: silhouette={silhouette:.3f}. Consider more distinct pairs.",
                details={"silhouette": silhouette}
            ))
        
        return silhouette, issues
    except ImportError:
        return None, issues


def _compute_held_out_transfer(
    difference_vectors: torch.Tensor,
    config: VectorQualityConfig
) -> Optional[float]:
    """Compute transfer score: train on 80%, test alignment on held-out 20%."""
    n = len(difference_vectors)
    
    if n < config.min_pairs_for_analysis:
        return None
    
    # 80/20 split
    n_train = int(n * 0.8)
    if n_train < 2 or n - n_train < 1:
        return None
    
    train_vec = _create_vector_from_diffs(difference_vectors[:n_train])
    test_diffs = difference_vectors[n_train:]
    
    # Mean alignment of held-out pairs with training vector
    test_alignments = [_cosine_similarity(diff, train_vec) for diff in test_diffs]
    return statistics.mean(test_alignments)


def _compute_cv_classification(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    config: VectorQualityConfig
) -> Optional[float]:
    """Compute cross-validation classification accuracy using logistic regression."""
    n_pos = len(positive_activations)
    n_neg = len(negative_activations)
    
    if n_pos < 3 or n_neg < 3:
        return None
    
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        X = torch.cat([positive_activations, negative_activations], dim=0).float().numpy()
        y = np.array([1] * n_pos + [0] * n_neg)
        
        n_folds = min(config.cv_folds, min(n_pos, n_neg))
        if n_folds < 2:
            return None
        
        clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except ImportError:
        return None


def _compute_cohens_d(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
) -> Optional[float]:
    """Compute Cohen's d effect size along the mean difference direction."""
    n_pos = len(positive_activations)
    n_neg = len(negative_activations)
    
    if n_pos < 2 or n_neg < 2:
        return None
    
    # Project onto the mean difference direction
    mean_pos = positive_activations.mean(dim=0)
    mean_neg = negative_activations.mean(dim=0)
    direction = mean_pos - mean_neg
    direction_norm = torch.norm(direction)
    
    if direction_norm < 1e-8:
        return None
    
    direction = direction / direction_norm
    
    # Project all activations onto this direction
    pos_proj = (positive_activations @ direction).float().numpy()
    neg_proj = (negative_activations @ direction).float().numpy()
    
    # Cohen's d = (mean1 - mean2) / pooled_std
    mean_diff = pos_proj.mean() - neg_proj.mean()
    pooled_var = ((n_pos - 1) * pos_proj.var() + (n_neg - 1) * neg_proj.var()) / (n_pos + n_neg - 2)
    pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 1e-8
    
    return float(mean_diff / pooled_std)


def run_vector_quality_diagnostics(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
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
    silhouette, issues = _compute_clustering(positive_activations, negative_activations, cfg)
    report.silhouette_score = silhouette
    all_issues.extend(issues)
    
    # 7. Held-out transfer
    report.held_out_transfer = _compute_held_out_transfer(difference_vectors, cfg)
    
    # 8. CV classification accuracy
    report.cv_classification_accuracy = _compute_cv_classification(
        positive_activations, negative_activations, cfg
    )
    
    # 9. Cohen's d
    report.cohens_d = _compute_cohens_d(positive_activations, negative_activations)
    
    # Determine overall quality
    critical_count = sum(1 for i in all_issues if i.severity == "critical")
    warning_count = sum(1 for i in all_issues if i.severity == "warning")
    
    if critical_count > 0:
        report.overall_quality = "poor"
    elif warning_count >= 3:
        report.overall_quality = "fair"
    elif warning_count > 0:
        report.overall_quality = "good"
    else:
        report.overall_quality = "excellent"
    
    # Generate recommendations
    if report.outlier_pairs:
        report.recommendations.append(
            f"Remove {len(report.outlier_pairs)} outlier pair(s) with low alignment and retrain."
        )
    if report.convergence_score and report.convergence_score < cfg.convergence_warning:
        report.recommendations.append(
            "Add more contrastive pairs to improve vector stability."
        )
    if report.snr and report.snr < cfg.snr_warning:
        report.recommendations.append(
            "Use more distinct positive/negative examples to increase signal-to-noise ratio."
        )
    if report.pca_pc1_variance and report.pca_pc1_variance < cfg.pca_variance_warning:
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
