"""Compute helpers for vector quality diagnostics."""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np

from wisent.core.constants import NORM_EPS, DATA_SPLIT_RATIO
from wisent.core.contrastive_pairs.diagnostics.analysis.vector_quality import (
    _cosine_similarity,
    _create_vector_from_diffs,
)

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
    
    snr = signal / (noise + NORM_EPS)
    
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
    n_train = int(n * DATA_SPLIT_RATIO)
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
        
        clf = LogisticRegression( solver='lbfgs')
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
    
    if direction_norm < NORM_EPS:
        return None
    
    direction = direction / direction_norm
    
    # Project all activations onto this direction
    pos_proj = (positive_activations @ direction).float().numpy()
    neg_proj = (negative_activations @ direction).float().numpy()
    
    # Cohen's d = (mean1 - mean2) / pooled_std
    mean_diff = pos_proj.mean() - neg_proj.mean()
    pooled_var = ((n_pos - 1) * pos_proj.var() + (n_neg - 1) * neg_proj.var()) / (n_pos + n_neg - 2)
    pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else NORM_EPS
    
    return float(mean_diff / pooled_std)


