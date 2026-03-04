"""Compute helpers for vector quality diagnostics."""
from __future__ import annotations

import statistics
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np

from wisent.core.utils.config_tools.constants import (
    NORM_EPS, COMBO_OFFSET, COMBO_BASE, PARSER_DEFAULT_LAYER_START,
    PERCENT_MULTIPLIER, BINARY_CLASS_NEGATIVE, BINARY_CLASS_POSITIVE,
)
from wisent.core.reading.diagnostics.analysis.vector_quality import (
    VectorQualityConfig,
    _cosine_similarity,
    _create_vector_from_diffs,
)
from wisent.core.reading.diagnostics.base import DiagnosticsIssue


def _compute_snr(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    config: VectorQualityConfig,
) -> Tuple[float, float, float, List[DiagnosticsIssue]]:
    """Compute signal-to-noise ratio."""
    issues = []

    mean_pos = positive_activations.mean(dim=PARSER_DEFAULT_LAYER_START)
    mean_neg = negative_activations.mean(dim=PARSER_DEFAULT_LAYER_START)
    signal = torch.norm(mean_pos - mean_neg).item()

    noise_pos = positive_activations.std(dim=PARSER_DEFAULT_LAYER_START).mean().item()
    noise_neg = negative_activations.std(dim=PARSER_DEFAULT_LAYER_START).mean().item()
    noise = (noise_pos + noise_neg) / COMBO_BASE

    snr = signal / (noise + NORM_EPS)

    if snr < config.vq_snr_critical:
        issues.append(DiagnosticsIssue(
            metric="snr",
            severity="critical",
            message=f"Very low signal-to-noise ratio: {snr:.2f} (< {config.vq_snr_critical}). Pairs may not capture distinct concept.",
            details={"snr": snr, "signal": signal, "noise": noise}
        ))
    elif snr < config.vq_snr_warning:
        issues.append(DiagnosticsIssue(
            metric="snr",
            severity="warning",
            message=f"Low signal-to-noise ratio: {snr:.2f}. Consider using cleaner pairs.",
            details={"snr": snr, "signal": signal, "noise": noise}
        ))

    return snr, signal, noise, issues


def _compute_pca(
    difference_vectors: torch.Tensor,
    config: VectorQualityConfig,
    default_score: float,
    *,
    min_samples_pca: int,
    pca_quality_components: int,
    cumulative_variance_top_n: int,
) -> Tuple[float, float, float, List[DiagnosticsIssue]]:
    """Compute PCA variance explained by top components."""
    issues = []
    n = len(difference_vectors)

    if n < min_samples_pca:
        return None, None, None, issues

    try:
        from sklearn.decomposition import PCA

        n_components = min(pca_quality_components, n - COMBO_OFFSET)
        pca = PCA(n_components=n_components)
        # Convert to float32 for sklearn compatibility (BFloat16 not supported)
        pca.fit(difference_vectors.float().numpy())

        pc1_var = pca.explained_variance_ratio_[PARSER_DEFAULT_LAYER_START]
        pc2_var = pca.explained_variance_ratio_[COMBO_OFFSET] if n_components > COMBO_OFFSET else default_score
        cumulative_3 = sum(pca.explained_variance_ratio_[:min(cumulative_variance_top_n, n_components)])

        if pc1_var < config.vq_pca_variance_critical:
            issues.append(DiagnosticsIssue(
                metric="pca",
                severity="critical",
                message=f"No dominant direction: PC1 explains only {pc1_var*PERCENT_MULTIPLIER:.1f}% variance (< {config.vq_pca_variance_critical*PERCENT_MULTIPLIER}%).",
                details={"pc1_variance": pc1_var, "pc2_variance": pc2_var}
            ))
        elif pc1_var < config.vq_pca_variance_warning:
            issues.append(DiagnosticsIssue(
                metric="pca",
                severity="warning",
                message=f"Weak dominant direction: PC1 explains {pc1_var*PERCENT_MULTIPLIER:.1f}% variance.",
                details={"pc1_variance": pc1_var, "pc2_variance": pc2_var}
            ))

        return pc1_var, pc2_var, cumulative_3, issues
    except ImportError:
        return None, None, None, issues


def _compute_pair_alignment(
    difference_vectors: torch.Tensor,
    final_vector: torch.Tensor,
    pair_prompts: Optional[List[str]],
    config: VectorQualityConfig,
    default_score: float,
) -> Tuple[float, float, float, float, List[Tuple[int, float, str]], List[DiagnosticsIssue]]:
    """Analyze how well each pair aligns with the final vector."""
    issues = []
    outliers = []

    n = len(difference_vectors)
    if n < COMBO_BASE:
        return None, None, None, None, [], issues

    alignments = []
    for i, diff in enumerate(difference_vectors):
        alignment = _cosine_similarity(diff, final_vector)
        alignments.append(alignment)

        prompt = pair_prompts[i] if pair_prompts and i < len(pair_prompts) else f"pair_{i}"

        if alignment < config.vq_alignment_min_critical:
            outliers.append((i, alignment, prompt))

    mean_align = statistics.mean(alignments)
    std_align = statistics.stdev(alignments) if len(alignments) > COMBO_OFFSET else default_score
    min_align = min(alignments)
    max_align = max(alignments)

    if outliers:
        issues.append(DiagnosticsIssue(
            metric="pair_alignment",
            severity="warning",
            message=f"{len(outliers)} outlier pair(s) with alignment < {config.vq_alignment_min_critical}. Consider removing them.",
            details={"outliers": [(i, a, p) for i, a, p in outliers]}
        ))

    if std_align > config.vq_alignment_std_warning:
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
    config: VectorQualityConfig,
    min_clusters: int,
) -> Tuple[float, List[DiagnosticsIssue]]:
    """Compute clustering separation using silhouette score."""
    issues = []

    n_pos = len(positive_activations)
    n_neg = len(negative_activations)

    if n_pos < min_clusters or n_neg < min_clusters:
        return None, issues

    try:
        from sklearn.metrics import silhouette_score

        all_activations = torch.cat([positive_activations, negative_activations], dim=PARSER_DEFAULT_LAYER_START).float().numpy()
        labels = [BINARY_CLASS_NEGATIVE] * n_pos + [BINARY_CLASS_POSITIVE] * n_neg

        silhouette = silhouette_score(all_activations, labels)

        if silhouette < config.vq_silhouette_critical:
            issues.append(DiagnosticsIssue(
                metric="clustering",
                severity="critical",
                message=f"Very poor class separation: silhouette={silhouette:.3f} (< {config.vq_silhouette_critical}). Positive/negative activations overlap significantly.",
                details={"silhouette": silhouette}
            ))
        elif silhouette < config.vq_silhouette_warning:
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
    config: VectorQualityConfig,
) -> Optional[float]:
    """Compute transfer score: train on split ratio, test alignment on held-out remainder."""
    n = len(difference_vectors)

    if n < config.vq_min_pairs:
        return None

    n_train = int(n * 0.8)
    if n_train < COMBO_BASE or n - n_train < COMBO_OFFSET:
        return None

    train_vec = _create_vector_from_diffs(difference_vectors[:n_train])
    test_diffs = difference_vectors[n_train:]

    # Mean alignment of held-out pairs with training vector
    test_alignments = [_cosine_similarity(diff, train_vec) for diff in test_diffs]
    return statistics.mean(test_alignments)


def _compute_cv_classification(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    config: VectorQualityConfig,
    *, adaptive_cv_min_folds: int, cv_folds: int, min_cloud_points: int,
) -> Optional[float]:
    """Compute cross-validation classification accuracy using logistic regression."""
    n_pos = len(positive_activations)
    n_neg = len(negative_activations)

    if n_pos < min_cloud_points or n_neg < min_cloud_points:
        return None

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        X = torch.cat([positive_activations, negative_activations], dim=PARSER_DEFAULT_LAYER_START).float().numpy()
        y = np.array([BINARY_CLASS_POSITIVE] * n_pos + [BINARY_CLASS_NEGATIVE] * n_neg)

        n_folds = min(cv_folds, min(n_pos, n_neg))
        if n_folds < adaptive_cv_min_folds:
            return None

        clf = LogisticRegression(solver='lbfgs')
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except ImportError:
        return None


def _compute_cohens_d(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    min_clusters: int,
) -> Optional[float]:
    """Compute Cohen's d effect size along the mean difference direction."""
    n_pos = len(positive_activations)
    n_neg = len(negative_activations)

    if n_pos < min_clusters or n_neg < min_clusters:
        return None

    # Project onto the mean difference direction
    mean_pos = positive_activations.mean(dim=PARSER_DEFAULT_LAYER_START)
    mean_neg = negative_activations.mean(dim=PARSER_DEFAULT_LAYER_START)
    direction = mean_pos - mean_neg
    direction_norm = torch.norm(direction)

    if direction_norm < NORM_EPS:
        return None

    direction = direction / direction_norm

    # Project all activations onto this direction
    pos_proj = (positive_activations @ direction).float().numpy()
    neg_proj = (negative_activations @ direction).float().numpy()

    # Cohen's d = (mean1 - mean2) / pooled_std
    mean_diff = pos_proj.mean() - neg_proj.mean()
    pooled_var = (
        (n_pos - COMBO_OFFSET) * pos_proj.var() + (n_neg - COMBO_OFFSET) * neg_proj.var()
    ) / (n_pos + n_neg - COMBO_BASE)
    pooled_std = np.sqrt(pooled_var) if pooled_var > PARSER_DEFAULT_LAYER_START else NORM_EPS

    return float(mean_diff / pooled_std)


