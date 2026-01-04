"""
Geometry search runner.

Runs geometry tests across the search space using cached activations.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import torch

import numpy as np

from wisent.core.geometry_search_space import GeometrySearchSpace, GeometrySearchConfig
from wisent.core.activations.extraction_strategy import ExtractionStrategy
from wisent.core.activations.activation_cache import (
    ActivationCache,
    CachedActivations,
    RawActivationCache,
    RawCachedActivations,
    collect_and_cache_activations,
    collect_and_cache_raw_activations,
    get_strategy_text_family,
)
from wisent.core.utils.layer_combinations import get_layer_combinations


def compute_signal_strength(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_folds: int = 5,
) -> float:
    """
    Compute signal strength using MLP cross-validation accuracy.
    
    This measures whether there is ANY extractable signal (linear or nonlinear)
    that generalizes to unseen data. Random/nonsense data gives ~0.5.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        n_folds: Number of CV folds
        
    Returns:
        Cross-validation accuracy (0.5 = no signal, >0.7 = signal exists)
    """
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import cross_val_score
        
        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        
        if n_pos < 5 or n_neg < 5:
            return 0.5  # Not enough data
        
        X = torch.cat([pos_activations, neg_activations], dim=0).float().cpu().numpy()
        y = np.array([1] * n_pos + [0] * n_neg)
        
        n_folds = min(n_folds, min(n_pos, n_neg))
        if n_folds < 2:
            return 0.5
        
        clf = MLPClassifier(
            hidden_layer_sizes=(16,),
            max_iter=500,
            random_state=42,
        )
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except Exception:
        return 0.5


def compute_knn_accuracy(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    k: int = 10,
    n_folds: int = 5,
) -> float:
    """
    Compute k-NN cross-validation accuracy.
    
    Measures local separability without assuming linearity.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        k: Number of neighbors
        n_folds: Number of CV folds
        
    Returns:
        Cross-validation accuracy
    """
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        
        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        
        if n_pos < k + 1 or n_neg < k + 1:
            return 0.5
        
        X = torch.cat([pos_activations, neg_activations], dim=0).float().cpu().numpy()
        y = np.array([1] * n_pos + [0] * n_neg)
        
        n_folds = min(n_folds, min(n_pos, n_neg))
        if n_folds < 2:
            return 0.5
        
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except Exception:
        return 0.5


def compute_knn_pca_accuracy(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    k: int = 10,
    n_components: int = 50,
    n_folds: int = 5,
) -> float:
    """
    Compute k-NN accuracy on PCA-reduced features.
    
    This addresses the curse of dimensionality by projecting to lower dimensions
    before k-NN. If k-NN on PCA features exceeds linear probe, it indicates
    genuine nonlinear structure (not just high-d artifacts).
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        k: Number of neighbors
        n_components: Number of PCA components to keep
        n_folds: Number of CV folds
        
    Returns:
        Cross-validation accuracy on PCA-reduced features
    """
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.decomposition import PCA
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        
        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        
        if n_pos < k + 1 or n_neg < k + 1:
            return 0.5
        
        X = torch.cat([pos_activations, neg_activations], dim=0).float().cpu().numpy()
        y = np.array([1] * n_pos + [0] * n_neg)
        
        n_folds = min(n_folds, min(n_pos, n_neg))
        if n_folds < 2:
            return 0.5
        
        # Reduce to min(n_components, n_samples-1, n_features)
        actual_components = min(n_components, len(X) - 1, X.shape[1])
        
        # Pipeline: PCA then k-NN
        clf = Pipeline([
            ('pca', PCA(n_components=actual_components)),
            ('knn', KNeighborsClassifier(n_neighbors=k))
        ])
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except Exception:
        return 0.5


def compute_knn_umap_accuracy(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    k: int = 10,
    n_components: int = 10,
    n_folds: int = 5,
) -> float:
    """
    Compute k-NN accuracy on UMAP-reduced features.
    
    UMAP preserves nonlinear structure better than PCA. If k-NN on UMAP
    features significantly exceeds linear probe, it indicates genuine
    nonlinear structure in the representation.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        k: Number of neighbors for k-NN
        n_components: Number of UMAP components
        n_folds: Number of CV folds
        
    Returns:
        Cross-validation accuracy on UMAP-reduced features
    """
    try:
        import umap
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        
        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        
        if n_pos < k + 1 or n_neg < k + 1:
            return 0.5
        
        X = torch.cat([pos_activations, neg_activations], dim=0).float().cpu().numpy()
        y = np.array([1] * n_pos + [0] * n_neg)
        
        n_folds = min(n_folds, min(n_pos, n_neg))
        if n_folds < 2:
            return 0.5
        
        # UMAP reduction - use smaller n_neighbors for small datasets
        umap_n_neighbors = min(15, len(X) // 4)
        if umap_n_neighbors < 2:
            return 0.5
            
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=0.1,
            random_state=42,
        )
        X_umap = reducer.fit_transform(X)
        
        # k-NN on UMAP features
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, X_umap, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except ImportError:
        # UMAP not installed - fall back to 0.5 (will use other methods)
        return 0.5
    except Exception:
        return 0.5


def compute_knn_pacmap_accuracy(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    k: int = 10,
    n_components: int = 10,
    n_folds: int = 5,
) -> float:
    """
    Compute k-NN accuracy on PaCMAP-reduced features.
    
    PaCMAP (Pairwise Controlled Manifold Approximation) preserves both local
    AND global structure better than UMAP. It's faster and more robust for
    high-dimensional data. If k-NN on PaCMAP features significantly exceeds
    linear probe, it indicates genuine nonlinear structure.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        k: Number of neighbors for k-NN
        n_components: Number of PaCMAP components
        n_folds: Number of CV folds
        
    Returns:
        Cross-validation accuracy on PaCMAP-reduced features
    """
    try:
        import pacmap
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        
        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        
        if n_pos < k + 1 or n_neg < k + 1:
            return 0.5
        
        X = torch.cat([pos_activations, neg_activations], dim=0).float().cpu().numpy()
        y = np.array([1] * n_pos + [0] * n_neg)
        
        n_folds = min(n_folds, min(n_pos, n_neg))
        if n_folds < 2:
            return 0.5
        
        # PaCMAP reduction
        reducer = pacmap.PaCMAP(
            n_components=n_components,
            n_neighbors=min(10, len(X) // 4),
            MN_ratio=0.5,
            FP_ratio=2.0,
            random_state=42,
        )
        X_pacmap = reducer.fit_transform(X)
        
        # k-NN on PaCMAP features
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, X_pacmap, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except ImportError:
        # PaCMAP not installed - fall back to 0.5 (will use other methods)
        return 0.5
    except Exception:
        return 0.5


def compute_mlp_probe_accuracy(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    hidden_size: int = 64,
    n_folds: int = 5,
) -> float:
    """
    Compute MLP probe cross-validation accuracy.
    
    This provides a nonlinear baseline that is more robust than k-NN in high
    dimensions. If MLP substantially exceeds linear probe, it indicates
    genuine nonlinear structure.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        hidden_size: Hidden layer size
        n_folds: Number of CV folds
        
    Returns:
        Cross-validation accuracy
    """
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import cross_val_score
        
        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        
        if n_pos < 5 or n_neg < 5:
            return 0.5
        
        X = torch.cat([pos_activations, neg_activations], dim=0).float().cpu().numpy()
        y = np.array([1] * n_pos + [0] * n_neg)
        
        n_folds = min(n_folds, min(n_pos, n_neg))
        if n_folds < 2:
            return 0.5
        
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden_size,),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            alpha=0.01,  # L2 regularization
        )
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except Exception:
        return 0.5


def compute_mmd_rbf(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> float:
    """
    Compute Maximum Mean Discrepancy with RBF kernel.
    
    Measures distribution difference without assuming linearity.
    Higher values indicate more separable distributions.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        
    Returns:
        MMD value (0 = identical distributions)
    """
    try:
        from sklearn.metrics.pairwise import rbf_kernel
        from scipy.spatial.distance import cdist
        
        pos = pos_activations.float().cpu().numpy()
        neg = neg_activations.float().cpu().numpy()
        
        # Use median heuristic for gamma
        all_data = np.vstack([pos, neg])
        dists = cdist(all_data, all_data, 'euclidean')
        gamma = 1.0 / (2 * np.median(dists[dists > 0]) ** 2 + 1e-10)
        
        K_pp = rbf_kernel(pos, pos, gamma=gamma)
        K_nn = rbf_kernel(neg, neg, gamma=gamma)
        K_pn = rbf_kernel(pos, neg, gamma=gamma)
        
        m = len(pos)
        n = len(neg)
        
        mmd = (K_pp.sum() / (m * m) + 
               K_nn.sum() / (n * n) - 
               2 * K_pn.sum() / (m * n))
        
        return float(max(0, mmd))
    except Exception:
        return 0.0


def estimate_local_intrinsic_dim(X: np.ndarray, k: int = 10) -> float:
    """
    Estimate local intrinsic dimensionality using MLE method.
    Based on Levina & Bickel (2004).
    
    Args:
        X: [N, D] data matrix
        k: Number of neighbors for estimation
        
    Returns:
        Estimated intrinsic dimension
    """
    from scipy.spatial.distance import cdist
    
    if len(X) < k + 1:
        return float(X.shape[1])
    
    dists = cdist(X, X, 'euclidean')
    np.fill_diagonal(dists, np.inf)
    
    sorted_dists = np.sort(dists, axis=1)[:, :k]
    
    dims = []
    for i in range(len(X)):
        T_k = sorted_dists[i, k-1]
        if T_k < 1e-10:
            continue
        log_ratios = np.log(sorted_dists[i, :k-1] / T_k + 1e-10)
        if len(log_ratios) > 0 and log_ratios.sum() < 0:
            dim_est = -(k - 1) / log_ratios.sum()
            dims.append(min(dim_est, X.shape[1]))
    
    return float(np.median(dims)) if dims else float(X.shape[1])


def compute_local_intrinsic_dims(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    k: int = 10,
) -> tuple:
    """
    Compute local intrinsic dimension for pos and neg separately.
    
    Different local dimensions suggest different geometric structures.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        k: Number of neighbors
        
    Returns:
        (local_dim_pos, local_dim_neg, ratio)
    """
    try:
        pos = pos_activations.float().cpu().numpy()
        neg = neg_activations.float().cpu().numpy()
        
        dim_pos = estimate_local_intrinsic_dim(pos, k)
        dim_neg = estimate_local_intrinsic_dim(neg, k)
        ratio = dim_pos / (dim_neg + 1e-10)
        
        return dim_pos, dim_neg, ratio
    except Exception:
        return 0.0, 0.0, 1.0


def compute_fisher_per_dimension(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> dict:
    """
    Compute Fisher ratio for each dimension and summary stats.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        
    Returns:
        Dict with fisher_max, fisher_gini, fisher_top10_ratio, num_dims_above_1
    """
    try:
        pos = pos_activations.float().cpu().numpy()
        neg = neg_activations.float().cpu().numpy()
        
        n_dims = pos.shape[1]
        fishers = np.zeros(n_dims)
        
        for d in range(n_dims):
            pos_d = pos[:, d]
            neg_d = neg[:, d]
            
            mean_pos = pos_d.mean()
            mean_neg = neg_d.mean()
            var_pos = pos_d.var()
            var_neg = neg_d.var()
            
            between_var = (mean_pos - mean_neg) ** 2
            within_var = (var_pos + var_neg) / 2
            
            if within_var > 1e-10:
                fishers[d] = between_var / within_var
        
        # Summary stats
        fisher_max = float(fishers.max())
        
        # Gini coefficient
        values = np.abs(fishers)
        if values.sum() > 1e-10:
            values = np.sort(values)
            n = len(values)
            fisher_gini = (2 * np.sum((np.arange(1, n+1) * values)) / (n * values.sum())) - (n + 1) / n
        else:
            fisher_gini = 0.0
        
        # Top 10 ratio
        sorted_fishers = np.sort(fishers)[::-1]
        top10_sum = sorted_fishers[:10].sum()
        total_sum = fishers.sum() + 1e-10
        fisher_top10_ratio = float(top10_sum / total_sum)
        
        num_dims_above_1 = int((fishers > 1.0).sum())
        
        return {
            "fisher_max": fisher_max,
            "fisher_gini": float(fisher_gini),
            "fisher_top10_ratio": fisher_top10_ratio,
            "num_dims_fisher_above_1": num_dims_above_1,
        }
    except Exception:
        return {
            "fisher_max": 0.0,
            "fisher_gini": 0.0,
            "fisher_top10_ratio": 0.0,
            "num_dims_fisher_above_1": 0,
        }


def compute_density_ratio(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> float:
    """
    Compute ratio of average intra-class distances.
    
    Values far from 1 suggest different local geometries.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        
    Returns:
        Density ratio (pos avg dist / neg avg dist)
    """
    try:
        from scipy.spatial.distance import cdist
        
        pos = pos_activations.float().cpu().numpy()
        neg = neg_activations.float().cpu().numpy()
        
        if len(pos) < 2 or len(neg) < 2:
            return 1.0
        
        pos_dists = cdist(pos, pos, 'euclidean')
        neg_dists = cdist(neg, neg, 'euclidean')
        
        np.fill_diagonal(pos_dists, np.nan)
        np.fill_diagonal(neg_dists, np.nan)
        
        avg_pos = np.nanmean(pos_dists)
        avg_neg = np.nanmean(neg_dists)
        
        if avg_neg < 1e-10:
            return 1.0
        
        return float(avg_pos / avg_neg)
    except Exception:
        return 1.0


def compute_linear_probe_accuracy(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_folds: int = 5,
) -> float:
    """
    Compute linear probe cross-validation accuracy.
    
    If signal_strength is high but linear_probe is low, the signal is nonlinear.
    If both are high, signal is linear and CAA should work.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        n_folds: Number of CV folds
        
    Returns:
        Cross-validation accuracy (0.5 = no linear signal)
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        
        if n_pos < 5 or n_neg < 5:
            return 0.5
        
        X = torch.cat([pos_activations, neg_activations], dim=0).float().cpu().numpy()
        y = np.array([1] * n_pos + [0] * n_neg)
        
        n_folds = min(n_folds, min(n_pos, n_neg))
        if n_folds < 2:
            return 0.5
        
        clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except Exception:
        return 0.5


def generate_nonsense_activations(
    model,
    tokenizer,
    n_pairs: int = 50,
    layer: int = None,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate activations from random token sequences (nonsense baseline).
    
    This creates a null baseline - what activation differences look like
    when there is NO semantic concept. If a benchmark's metrics are similar
    to nonsense, it has no extractable concept.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        n_pairs: Number of nonsense pairs to generate
        layer: Which layer to extract (default: middle layer)
        device: Device to run on
        
    Returns:
        Tuple of (pos_activations, neg_activations) tensors
    """
    import random
    
    vocab_size = tokenizer.vocab_size
    if layer is None:
        layer = model.config.num_hidden_layers // 2
    
    def generate_random_tokens(min_tokens=5, max_tokens=25):
        n_tokens = random.randint(min_tokens, max_tokens)
        # Avoid special tokens (first 100 and last 100)
        token_ids = [random.randint(100, vocab_size - 100) for _ in range(n_tokens)]
        return tokenizer.decode(token_ids)
    
    def get_activation(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states[layer + 1]  # +1 because index 0 is embeddings
            return hs[0, -1, :]  # Last token
    
    pos_acts = []
    neg_acts = []
    
    for _ in range(n_pairs):
        try:
            pos_text = generate_random_tokens()
            neg_text = generate_random_tokens()
            pos_acts.append(get_activation(pos_text))
            neg_acts.append(get_activation(neg_text))
        except Exception:
            continue
    
    if len(pos_acts) < 10:
        raise ValueError(f"Could only generate {len(pos_acts)} nonsense pairs")
    
    return torch.stack(pos_acts), torch.stack(neg_acts)


def compute_nonsense_baseline(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    nonsense_pos: torch.Tensor,
    nonsense_neg: torch.Tensor,
    n_folds: int = 5,
) -> Dict[str, float]:
    """
    Compare real activations against nonsense baseline.
    
    Computes linear probe accuracy for both real and nonsense pairs,
    then determines if real signal is meaningfully above baseline.
    
    Args:
        pos_activations: Real positive activations
        neg_activations: Real negative activations  
        nonsense_pos: Nonsense positive activations
        nonsense_neg: Nonsense negative activations
        n_folds: Number of CV folds
        
    Returns:
        Dict with:
            - real_accuracy: Linear probe accuracy on real data
            - nonsense_accuracy: Linear probe accuracy on nonsense
            - signal_ratio: real / nonsense (>2 = real signal)
            - has_real_signal: Boolean verdict
            - signal_above_baseline: real - nonsense
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    def compute_accuracy(pos, neg):
        n = min(len(pos), len(neg))
        if n < 10:
            return 0.5
        
        X = torch.cat([pos[:n], neg[:n]], dim=0).float().cpu().numpy()
        y = np.array([1] * n + [0] * n)
        
        n_folds_actual = min(n_folds, n)
        if n_folds_actual < 2:
            return 0.5
        
        clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        try:
            scores = cross_val_score(clf, X, y, cv=n_folds_actual, scoring='accuracy')
            return float(scores.mean())
        except Exception:
            return 0.5
    
    real_acc = compute_accuracy(pos_activations, neg_activations)
    nonsense_acc = compute_accuracy(nonsense_pos, nonsense_neg)
    
    # Compute ratio (avoid division by zero)
    if nonsense_acc <= 0.5:
        signal_ratio = real_acc / 0.5  # Compare to chance
    else:
        signal_ratio = real_acc / nonsense_acc
    
    signal_above_baseline = real_acc - nonsense_acc
    
    # Real signal if:
    # 1. Real accuracy > 60% AND
    # 2. Real is at least 10% above nonsense OR ratio > 1.5
    has_real_signal = (
        real_acc > 0.60 and 
        (signal_above_baseline > 0.10 or signal_ratio > 1.5)
    )
    
    return {
        "real_accuracy": real_acc,
        "nonsense_accuracy": nonsense_acc,
        "signal_ratio": signal_ratio,
        "signal_above_baseline": signal_above_baseline,
        "has_real_signal": has_real_signal,
    }


def compute_icd(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute Intrinsic Concept Dimensionality (ICD) of difference vectors.
    
    ICD measures effective rank - how many independent directions are needed
    to represent the concept. Low ICD = concentrated signal, high ICD = diffuse/noise.
    
    Args:
        pos_activations: [N, hidden_dim] positive activations
        neg_activations: [N, hidden_dim] negative activations
        
    Returns:
        Dict with:
            - icd: Intrinsic Concept Dimensionality (effective rank)
            - top1_variance: Fraction of variance explained by top direction
            - top5_variance: Fraction explained by top 5 directions
            - n_samples: Number of samples used
    """
    n = min(len(pos_activations), len(neg_activations))
    if n < 5:
        return {"icd": 0.0, "top1_variance": 0.0, "top5_variance": 0.0, "n_samples": n}
    
    diff_vectors = (pos_activations[:n] - neg_activations[:n]).float().cpu().numpy()
    diff_vectors = diff_vectors.astype(np.float64)
    
    try:
        U, S, Vh = np.linalg.svd(diff_vectors, full_matrices=False)
        S = S[S > 1e-10]
        
        if len(S) == 0:
            return {"icd": 0.0, "top1_variance": 0.0, "top5_variance": 0.0, "n_samples": n}
        
        # ICD = effective rank
        icd = float((S.sum() ** 2) / (S ** 2).sum())
        
        # Variance explained
        total_var = (S ** 2).sum()
        top1_var = float((S[0] ** 2) / total_var) if total_var > 0 else 0.0
        top5_var = float((S[:5] ** 2).sum() / total_var) if total_var > 0 else 0.0
        
        return {
            "icd": icd,
            "top1_variance": top1_var,
            "top5_variance": top5_var,
            "n_samples": n,
        }
    except Exception:
        return {"icd": 0.0, "top1_variance": 0.0, "top5_variance": 0.0, "n_samples": n}


def compute_multi_direction_accuracy(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    k_values: List[int] = [1, 2, 3, 5, 10],
    n_folds: int = 5,
    n_bootstrap: int = 50,
) -> Dict[str, Any]:
    """
    Test how many separation directions are needed for good classification.
    
    Uses bootstrap + SVD to find multiple separation directions:
    1. Bootstrap N subsets of pairs
    2. Compute diff-mean for each subset (each is a separation direction)
    3. SVD on matrix of diff-means → principal separation directions
    4. Test linear probe accuracy using top-k directions
    
    This finds directions that SEPARATE classes, not just explain variance.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        k_values: List of k values to test (number of directions)
        n_folds: Number of CV folds
        n_bootstrap: Number of bootstrap samples for direction discovery
        
    Returns:
        Dict with:
            - accuracy_by_k: {k: accuracy} for each k
            - min_k_for_good: minimum k where accuracy >= 0.6
            - saturation_k: k where accuracy stops improving significantly
            - gain_from_multi: accuracy(best_k) - accuracy(k=1)
            - direction_variance_explained: how much variance top-k directions capture
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        n_pairs = min(n_pos, n_neg)
        
        if n_pairs < 5:
            return {
                "accuracy_by_k": {k: 0.5 for k in k_values},
                "min_k_for_good": -1,
                "saturation_k": 1,
                "gain_from_multi": 0.0,
            }
        
        pos_np = pos_activations.float().cpu().numpy()
        neg_np = neg_activations.float().cpu().numpy()
        hidden_dim = pos_np.shape[1]
        
        # Bootstrap: compute diff-mean for random subsets
        # Each diff-mean is a candidate separation direction
        rng = np.random.RandomState(42)
        subset_size = max(n_pairs // 2, 3)  # Use half the pairs per subset
        
        diff_means = []
        for _ in range(n_bootstrap):
            # Random subset of pairs
            indices = rng.choice(n_pairs, size=subset_size, replace=False)
            pos_subset = pos_np[indices]
            neg_subset = neg_np[indices]
            
            # Diff-mean for this subset
            diff_mean = pos_subset.mean(axis=0) - neg_subset.mean(axis=0)
            norm = np.linalg.norm(diff_mean)
            if norm > 1e-8:
                diff_means.append(diff_mean / norm)  # Normalize
        
        if len(diff_means) < 2:
            return {
                "accuracy_by_k": {k: 0.5 for k in k_values},
                "min_k_for_good": -1,
                "saturation_k": 1,
                "gain_from_multi": 0.0,
            }
        
        # Stack diff-means into matrix [n_bootstrap, hidden_dim]
        diff_matrix = np.stack(diff_means, axis=0)
        
        # SVD to find principal separation directions
        # U @ S @ Vh = diff_matrix
        # Vh contains the principal directions
        U, S, Vh = np.linalg.svd(diff_matrix, full_matrices=False)
        
        # Vh: [min(n_bootstrap, hidden_dim), hidden_dim]
        # Each row is a principal separation direction
        max_k = min(max(k_values), len(S), Vh.shape[0])
        if max_k < 1:
            return {
                "accuracy_by_k": {k: 0.5 for k in k_values},
                "min_k_for_good": -1,
                "saturation_k": 1,
                "gain_from_multi": 0.0,
            }
        
        # Prepare full data for classification
        X_full = np.vstack([pos_np, neg_np])
        y = np.array([1] * n_pos + [0] * n_neg)
        
        n_folds_actual = min(n_folds, min(n_pos, n_neg))
        if n_folds_actual < 2:
            n_folds_actual = 2
        
        accuracy_by_k = {}
        
        for k in k_values:
            if k > max_k:
                accuracy_by_k[k] = accuracy_by_k.get(max_k, 0.5)
                continue
            
            # Project data onto top-k separation directions
            top_k_directions = Vh[:k]  # [k, hidden_dim]
            X_projected = X_full @ top_k_directions.T  # [N, k]
            
            # Linear probe on projected features
            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            try:
                scores = cross_val_score(clf, X_projected, y, cv=n_folds_actual, scoring='accuracy')
                accuracy_by_k[k] = float(scores.mean())
            except Exception:
                accuracy_by_k[k] = 0.5
        
        # Find minimum k for good accuracy (>= 0.6)
        min_k_for_good = -1
        for k in sorted(k_values):
            if accuracy_by_k.get(k, 0) >= 0.6:
                min_k_for_good = k
                break
        
        # Find saturation point (where adding more directions doesn't help much)
        sorted_ks = sorted([k for k in k_values if k <= max_k])
        saturation_k = sorted_ks[0] if sorted_ks else 1
        
        for i in range(1, len(sorted_ks)):
            k_prev = sorted_ks[i-1]
            k_curr = sorted_ks[i]
            improvement = accuracy_by_k.get(k_curr, 0) - accuracy_by_k.get(k_prev, 0)
            if improvement < 0.02:  # Less than 2% improvement
                saturation_k = k_prev
                break
            saturation_k = k_curr
        
        # Gain from using multiple directions
        acc_k1 = accuracy_by_k.get(1, 0.5)
        best_acc = max(accuracy_by_k.values()) if accuracy_by_k else 0.5
        gain_from_multi = best_acc - acc_k1
        
        return {
            "accuracy_by_k": accuracy_by_k,
            "min_k_for_good": min_k_for_good,
            "saturation_k": saturation_k,
            "gain_from_multi": gain_from_multi,
        }
    except Exception:
        return {
            "accuracy_by_k": {k: 0.5 for k in k_values},
            "min_k_for_good": -1,
            "saturation_k": 1,
            "gain_from_multi": 0.0,
        }


def compute_direction_stability(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_bootstrap: int = 30,
    subset_fraction: float = 0.5,
) -> Dict[str, float]:
    """
    Measure stability of the separation direction across bootstrap samples.
    
    If the direction is stable (high cosine similarity across subsets),
    then there is likely ONE consistent direction encoding the concept.
    If unstable, different samples use different directions.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        n_bootstrap: Number of bootstrap samples
        subset_fraction: Fraction of data to use per bootstrap
        
    Returns:
        Dict with:
            - mean_cosine: Mean pairwise cosine similarity between bootstrap directions
            - std_cosine: Std of pairwise cosine similarities
            - min_cosine: Minimum pairwise cosine similarity
            - stability_score: 0-1 score (1 = perfectly stable)
    """
    try:
        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        n_pairs = min(n_pos, n_neg)
        
        if n_pairs < 10:
            return {
                "mean_cosine": 0.0,
                "std_cosine": 1.0,
                "min_cosine": -1.0,
                "stability_score": 0.0,
            }
        
        pos_np = pos_activations.float().cpu().numpy()
        neg_np = neg_activations.float().cpu().numpy()
        
        rng = np.random.RandomState(42)
        subset_size = max(int(n_pairs * subset_fraction), 5)
        
        # Compute diff-mean directions for bootstrap samples
        directions = []
        for _ in range(n_bootstrap):
            indices = rng.choice(n_pairs, size=subset_size, replace=False)
            pos_subset = pos_np[indices]
            neg_subset = neg_np[indices]
            
            diff_mean = pos_subset.mean(axis=0) - neg_subset.mean(axis=0)
            norm = np.linalg.norm(diff_mean)
            if norm > 1e-8:
                directions.append(diff_mean / norm)
        
        if len(directions) < 2:
            return {
                "mean_cosine": 0.0,
                "std_cosine": 1.0,
                "min_cosine": -1.0,
                "stability_score": 0.0,
            }
        
        # Compute pairwise cosine similarities
        directions = np.stack(directions)  # [n_bootstrap, hidden_dim]
        cos_sim_matrix = directions @ directions.T  # [n_bootstrap, n_bootstrap]
        
        # Get off-diagonal elements
        n = cos_sim_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off_diagonal = cos_sim_matrix[mask]
        
        mean_cosine = float(off_diagonal.mean())
        std_cosine = float(off_diagonal.std())
        min_cosine = float(off_diagonal.min())
        
        # Stability score: high mean, low std, high min
        # Map mean from [-1, 1] to [0, 1], penalize for std
        stability_score = max(0, (mean_cosine + 1) / 2 - std_cosine * 0.5)
        stability_score = min(1.0, stability_score)
        
        return {
            "mean_cosine": mean_cosine,
            "std_cosine": std_cosine,
            "min_cosine": min_cosine,
            "stability_score": stability_score,
        }
    except Exception:
        return {
            "mean_cosine": 0.0,
            "std_cosine": 1.0,
            "min_cosine": -1.0,
            "stability_score": 0.0,
        }


def compute_diff_intrinsic_dim(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    k: int = 10,
) -> Dict[str, float]:
    """
    Compute intrinsic dimensionality of DIFFERENCE vectors (pos - neg).
    
    This tells us how many dimensions are needed to represent the
    separation between classes, not the data itself.
    
    Low ID → separation is low-dimensional (few directions needed)
    High ID → separation is high-dimensional (many directions needed)
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        k: Number of neighbors for MLE estimator
        
    Returns:
        Dict with:
            - intrinsic_dim: Estimated intrinsic dimension of differences
            - relative_dim: intrinsic_dim / ambient_dim (fraction)
            - is_low_dim: True if intrinsic_dim < 10
    """
    try:
        n_pairs = min(len(pos_activations), len(neg_activations))
        
        if n_pairs < k + 2:
            return {
                "intrinsic_dim": float(pos_activations.shape[1]),
                "relative_dim": 1.0,
                "is_low_dim": False,
            }
        
        # Compute difference vectors
        diff_vectors = pos_activations[:n_pairs] - neg_activations[:n_pairs]
        diff_np = diff_vectors.float().cpu().numpy()
        
        ambient_dim = diff_np.shape[1]
        
        # MLE intrinsic dimension estimator
        intrinsic_dim = estimate_local_intrinsic_dim(diff_np, k=k)
        relative_dim = intrinsic_dim / ambient_dim
        is_low_dim = intrinsic_dim < 10
        
        return {
            "intrinsic_dim": float(intrinsic_dim),
            "relative_dim": float(relative_dim),
            "is_low_dim": is_low_dim,
        }
    except Exception:
        return {
            "intrinsic_dim": float(pos_activations.shape[1]),
            "relative_dim": 1.0,
            "is_low_dim": False,
        }


def compute_steerability_metrics(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute metrics that predict whether steering will work.
    
    Key insight from experiments:
    - TruthfulQA: diff_mean_alignment=0.22, steering works (+12% accuracy)
    - HellaSwag: diff_mean_alignment=0.05, steering fails (0% improvement)
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        
    Returns:
        Dict with steerability metrics:
            - diff_mean_alignment: mean cosine between individual diffs and diff-mean
            - pct_positive_alignment: % of pairs where alignment > 0
            - steering_vector_norm: norm of diff-mean (if small, directions cancel)
            - cluster_direction_angle: angle between k=2 cluster steering directions
            - per_cluster_alignment_k2: alignment within k=2 clusters
            - spherical_silhouette_k2: silhouette using cosine distance
            - effective_steering_dims: how many independent directions needed (from eigenvalues)
            - steerability_score: overall 0-1 score predicting steering success
    """
    try:
        n_pairs = min(len(pos_activations), len(neg_activations))
        
        if n_pairs < 5:
            return _empty_steerability_metrics()
        
        # Compute difference vectors
        diff_vectors = (pos_activations[:n_pairs] - neg_activations[:n_pairs]).float().cpu().numpy()
        
        # 1. Diff-mean alignment (THE key predictor)
        diff_mean = diff_vectors.mean(axis=0)
        diff_mean_norm = np.linalg.norm(diff_mean)
        
        if diff_mean_norm < 1e-8:
            return _empty_steerability_metrics()
        
        diff_mean_normalized = diff_mean / diff_mean_norm
        
        # Normalize individual diffs
        diff_norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
        valid_mask = diff_norms.squeeze() > 1e-8
        
        if valid_mask.sum() < 5:
            return _empty_steerability_metrics()
        
        diff_normalized = diff_vectors[valid_mask] / diff_norms[valid_mask]
        
        # Alignment with diff-mean
        alignments = diff_normalized @ diff_mean_normalized
        diff_mean_alignment = float(alignments.mean())
        pct_positive_alignment = float((alignments > 0).mean())
        
        # 2. Steering vector norm (relative to individual diff norms)
        avg_diff_norm = float(diff_norms[valid_mask].mean())
        steering_vector_norm_ratio = diff_mean_norm / (avg_diff_norm + 1e-8)
        
        # 3. Cluster analysis (k=2) - do clusters have different steering directions?
        from sklearn.cluster import KMeans
        
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(diff_normalized)
        
        # Compute steering direction per cluster
        c0_mask = cluster_labels == 0
        c1_mask = cluster_labels == 1
        
        if c0_mask.sum() >= 2 and c1_mask.sum() >= 2:
            dir_c0 = diff_vectors[valid_mask][c0_mask].mean(axis=0)
            dir_c1 = diff_vectors[valid_mask][c1_mask].mean(axis=0)
            
            dir_c0_norm = dir_c0 / (np.linalg.norm(dir_c0) + 1e-8)
            dir_c1_norm = dir_c1 / (np.linalg.norm(dir_c1) + 1e-8)
            
            # Angle between cluster directions
            cos_angle = np.clip(np.dot(dir_c0_norm, dir_c1_norm), -1, 1)
            cluster_direction_angle = float(np.degrees(np.arccos(np.abs(cos_angle))))
            
            # Per-cluster alignment
            align_c0 = (diff_normalized[c0_mask] @ dir_c0_norm).mean()
            align_c1 = (diff_normalized[c1_mask] @ dir_c1_norm).mean()
            per_cluster_alignment = float((align_c0 * c0_mask.sum() + align_c1 * c1_mask.sum()) / valid_mask.sum())
        else:
            cluster_direction_angle = 0.0
            per_cluster_alignment = diff_mean_alignment
        
        # 4. Spherical silhouette (cosine distance)
        def spherical_silhouette(X_norm, labels):
            n = len(X_norm)
            k = len(set(labels))
            if k < 2:
                return 0.0
            
            silhouettes = []
            for i in range(min(n, 200)):  # Sample for speed
                same_cluster = labels == labels[i]
                if same_cluster.sum() > 1:
                    a = 1 - (X_norm[i] @ X_norm[same_cluster].T).mean()
                else:
                    a = 0
                
                b = float('inf')
                for c in range(k):
                    if c != labels[i]:
                        other = labels == c
                        if other.sum() > 0:
                            b_c = 1 - (X_norm[i] @ X_norm[other].T).mean()
                            b = min(b, b_c)
                
                if b != float('inf'):
                    silhouettes.append((b - a) / max(a, b) if max(a, b) > 0 else 0)
            
            return float(np.mean(silhouettes)) if silhouettes else 0.0
        
        spherical_sil = spherical_silhouette(diff_normalized, cluster_labels)
        
        # 5. Effective steering dimensions (eigenvalue analysis)
        try:
            cov = np.cov(diff_normalized.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
            
            # How many dimensions explain 90% of variance?
            total_var = eigenvalues.sum()
            if total_var > 0:
                cumsum = np.cumsum(eigenvalues) / total_var
                effective_dims = int(np.searchsorted(cumsum, 0.9) + 1)
            else:
                effective_dims = 1
        except:
            effective_dims = 1
        
        # 6. CAA-Probe alignment (THE key predictor of steering effectiveness!)
        # This measures if the CAA direction (mean of diffs) aligns with the probe's decision boundary
        # High alignment = steering in CAA direction will move toward "positive" class
        # Low alignment = CAA direction is orthogonal to what probe learned, steering won't work
        try:
            from sklearn.linear_model import LogisticRegression
            pos_np = pos_activations[:n_pairs].float().cpu().numpy()
            neg_np = neg_activations[:n_pairs].float().cpu().numpy()
            X = np.vstack([pos_np, neg_np])
            y = np.array([1] * n_pairs + [0] * n_pairs)
            probe = LogisticRegression(max_iter=500, random_state=42)
            probe.fit(X, y)
            probe_dir = probe.coef_[0]
            probe_dir_norm = probe_dir / (np.linalg.norm(probe_dir) + 1e-8)
            caa_probe_alignment = float(np.dot(diff_mean_normalized, probe_dir_norm))
        except:
            caa_probe_alignment = diff_mean_alignment  # Fallback
        
        # 7. Steerability score (weighted combination)
        # KEY CHANGE: Use caa_probe_alignment as main predictor, not diff_mean_alignment
        # caa_probe_alignment directly predicts if steering will work
        steerability_score = (
            0.5 * max(0, caa_probe_alignment) +  # Main predictor (CAA-Probe alignment)
            0.2 * pct_positive_alignment +
            0.15 * min(1.0, steering_vector_norm_ratio) +
            0.15 * (1 - cluster_direction_angle / 180)  # Lower angle = better
        )
        steerability_score = float(np.clip(steerability_score, 0, 1))
        
        return {
            "diff_mean_alignment": diff_mean_alignment,
            "caa_probe_alignment": caa_probe_alignment,  # NEW: the key metric!
            "pct_positive_alignment": pct_positive_alignment,
            "steering_vector_norm_ratio": float(steering_vector_norm_ratio),
            "cluster_direction_angle": cluster_direction_angle,
            "per_cluster_alignment_k2": per_cluster_alignment,
            "spherical_silhouette_k2": spherical_sil,
            "effective_steering_dims": effective_dims,
            "steerability_score": steerability_score,
        }
    except Exception:
        return _empty_steerability_metrics()


def _empty_steerability_metrics() -> Dict[str, float]:
    """Return empty steerability metrics."""
    return {
        "diff_mean_alignment": 0.0,
        "caa_probe_alignment": 0.0,
        "pct_positive_alignment": 0.5,
        "steering_vector_norm_ratio": 0.0,
        "cluster_direction_angle": 90.0,
        "per_cluster_alignment_k2": 0.0,
        "spherical_silhouette_k2": 0.0,
        "effective_steering_dims": 1,
        "steerability_score": 0.0,
    }


def compute_pairwise_diff_consistency(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, float]:
    """
    Measure consistency of individual difference vectors.
    
    For each pair, compute diff = pos_i - neg_i.
    Then measure how similar these diffs are to each other.
    
    High consistency → all pairs use same direction → ONE steering vector enough
    Low consistency → different pairs use different directions → need MULTIPLE vectors
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        
    Returns:
        Dict with:
            - mean_pairwise_cosine: Mean cosine sim between diff vectors
            - std_pairwise_cosine: Std of cosine similarities
            - fraction_positive: Fraction of pairs with positive cosine (same half-space)
            - fraction_high_sim: Fraction of pairs with cosine > 0.5
            - consistency_score: 0-1 summary score
    """
    try:
        n_pairs = min(len(pos_activations), len(neg_activations))
        
        if n_pairs < 3:
            return {
                "mean_pairwise_cosine": 0.0,
                "std_pairwise_cosine": 1.0,
                "fraction_positive": 0.5,
                "fraction_high_sim": 0.0,
                "consistency_score": 0.0,
            }
        
        # Compute and normalize difference vectors
        diff_vectors = pos_activations[:n_pairs] - neg_activations[:n_pairs]
        diff_np = diff_vectors.float().cpu().numpy()
        
        # Normalize
        norms = np.linalg.norm(diff_np, axis=1, keepdims=True)
        valid_mask = (norms.squeeze() > 1e-8)
        
        if valid_mask.sum() < 3:
            return {
                "mean_pairwise_cosine": 0.0,
                "std_pairwise_cosine": 1.0,
                "fraction_positive": 0.5,
                "fraction_high_sim": 0.0,
                "consistency_score": 0.0,
            }
        
        diff_normalized = diff_np[valid_mask] / norms[valid_mask]
        
        # Pairwise cosine similarity
        cos_sim_matrix = diff_normalized @ diff_normalized.T
        
        # Get off-diagonal
        n = cos_sim_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off_diagonal = cos_sim_matrix[mask]
        
        mean_cos = float(off_diagonal.mean())
        std_cos = float(off_diagonal.std())
        fraction_positive = float((off_diagonal > 0).mean())
        fraction_high_sim = float((off_diagonal > 0.5).mean())
        
        # Consistency score: combine metrics
        # High mean, low std, high fraction_positive → high consistency
        consistency_score = (
            0.4 * max(0, (mean_cos + 1) / 2) +  # mean cosine mapped to [0,1]
            0.3 * fraction_positive +
            0.3 * fraction_high_sim
        )
        consistency_score = min(1.0, max(0.0, consistency_score))
        
        return {
            "mean_pairwise_cosine": mean_cos,
            "std_pairwise_cosine": std_cos,
            "fraction_positive": fraction_positive,
            "fraction_high_sim": fraction_high_sim,
            "consistency_score": consistency_score,
        }
    except Exception:
        return {
            "mean_pairwise_cosine": 0.0,
            "std_pairwise_cosine": 1.0,
            "fraction_positive": 0.5,
            "fraction_high_sim": 0.0,
            "consistency_score": 0.0,
        }


def compute_linearity_score(
    linear_probe_accuracy: float,
    best_nonlinear_accuracy: float,
    direction_stability: float,
    diff_intrinsic_dim: float,
    pairwise_consistency: float,
    ambient_dim: int = 4096,
) -> Dict[str, Any]:
    """
    Compute overall linearity score combining multiple signals.
    
    A representation is "linear" if:
    1. Linear probe ≈ nonlinear probe (no hidden nonlinear structure)
    2. Direction is stable across samples (ONE consistent direction)
    3. Difference vectors have low intrinsic dimension (few directions needed)
    4. Individual diff vectors are consistent (all point same way)
    
    Args:
        linear_probe_accuracy: Linear probe CV accuracy
        best_nonlinear_accuracy: Best nonlinear probe accuracy
        direction_stability: Stability score from compute_direction_stability
        diff_intrinsic_dim: Intrinsic dimension of differences
        pairwise_consistency: Consistency score from compute_pairwise_diff_consistency
        ambient_dim: Ambient dimension for relative ID calculation
        
    Returns:
        Dict with:
            - linearity_score: 0-1 score (1 = definitely linear)
            - is_linear: Boolean verdict
            - evidence: Dict explaining each component
            - recommendation: CAA / MULTI_VECTOR / NONLINEAR / NO_SIGNAL
    """
    # Component 1: Linear vs Nonlinear gap
    # If linear ≈ nonlinear, no hidden nonlinear structure
    if best_nonlinear_accuracy < 0.55:
        gap_score = 0.0  # No signal
    else:
        gap = best_nonlinear_accuracy - linear_probe_accuracy
        if gap <= 0:
            gap_score = 1.0  # Linear is as good or better
        elif gap < 0.05:
            gap_score = 0.9
        elif gap < 0.10:
            gap_score = 0.7
        elif gap < 0.15:
            gap_score = 0.5
        else:
            gap_score = 0.2  # Significant nonlinear advantage
    
    # Component 2: Direction stability
    stability_score = direction_stability
    
    # Component 3: Intrinsic dimensionality
    # Low ID → few directions needed → more linear
    relative_id = diff_intrinsic_dim / ambient_dim
    if relative_id < 0.001:  # < 0.1% of ambient dim
        id_score = 1.0
    elif relative_id < 0.005:
        id_score = 0.8
    elif relative_id < 0.01:
        id_score = 0.6
    elif relative_id < 0.05:
        id_score = 0.4
    else:
        id_score = 0.2
    
    # Component 4: Pairwise consistency
    consistency_score = pairwise_consistency
    
    # Combine scores
    linearity_score = (
        0.30 * gap_score +
        0.25 * stability_score +
        0.20 * id_score +
        0.25 * consistency_score
    )
    
    # Determine verdict and recommendation
    if best_nonlinear_accuracy < 0.55:
        is_linear = False
        recommendation = "NO_SIGNAL"
    elif linearity_score >= 0.7:
        is_linear = True
        recommendation = "CAA"
    elif linearity_score >= 0.5:
        is_linear = True  # Somewhat linear
        recommendation = "CAA"  # Try CAA first
    elif linearity_score >= 0.3:
        is_linear = False
        recommendation = "MULTI_VECTOR"  # May need multiple steering vectors
    else:
        is_linear = False
        recommendation = "NONLINEAR"  # Need nonlinear methods
    
    return {
        "linearity_score": float(linearity_score),
        "is_linear": is_linear,
        "recommendation": recommendation,
        "evidence": {
            "gap_score": float(gap_score),
            "stability_score": float(stability_score),
            "id_score": float(id_score),
            "consistency_score": float(consistency_score),
            "linear_probe": float(linear_probe_accuracy),
            "best_nonlinear": float(best_nonlinear_accuracy),
            "diff_intrinsic_dim": float(diff_intrinsic_dim),
        }
    }


@dataclass
class GeometryTestResult:
    """Result of a single geometry test."""
    benchmark: str
    strategy: str
    layers: List[int]
    
    # Step 1: Is there any signal? (MLP CV accuracy)
    signal_strength: float  # MLP CV accuracy, ~0.5 = no signal, >0.6 = signal exists
    has_signal: bool  # signal_strength > 0.6
    
    # Step 2: Is signal linear? (Linear probe CV accuracy)
    linear_probe_accuracy: float  # Linear CV accuracy, high = linear, low = nonlinear
    is_linear: bool  # linear_probe_accuracy > 0.6 AND close to signal_strength
    
    # NEW: Nonlinear signal metrics
    knn_accuracy_k5: float  # k-NN CV accuracy with k=5
    knn_accuracy_k10: float  # k-NN CV accuracy with k=10
    knn_accuracy_k20: float  # k-NN CV accuracy with k=20
    knn_pca_accuracy: float  # k-NN on PCA-50 features (addresses curse of dimensionality)
    knn_umap_accuracy: float  # k-NN on UMAP-10 features (preserves nonlinear structure)
    knn_pacmap_accuracy: float  # k-NN on PaCMAP-10 features (preserves local+global structure)
    mlp_probe_accuracy: float  # MLP probe accuracy (nonlinear baseline)
    best_nonlinear_accuracy: float  # max(knn_k10, knn_pca, knn_umap, knn_pacmap, mlp) - best nonlinear signal
    mmd_rbf: float  # Maximum Mean Discrepancy with RBF kernel
    local_dim_pos: float  # Local intrinsic dimension of positive class
    local_dim_neg: float  # Local intrinsic dimension of negative class
    local_dim_ratio: float  # Ratio of local dimensions
    fisher_max: float  # Max Fisher ratio across all dimensions
    fisher_gini: float  # Gini coefficient of Fisher ratios (concentration)
    fisher_top10_ratio: float  # Fraction of total Fisher in top 10 dims
    num_dims_fisher_above_1: int  # Number of dimensions with Fisher > 1
    density_ratio: float  # Ratio of avg intra-class distances
    
    # Step 3: Geometry details (only meaningful if has_signal=True)
    # Best structure detected
    best_structure: str  # 'linear', 'cone', 'cluster', 'manifold', 'sparse', 'bimodal', 'orthogonal'
    best_score: float
    
    # All structure scores
    linear_score: float
    cone_score: float
    orthogonal_score: float
    manifold_score: float
    sparse_score: float
    cluster_score: float
    bimodal_score: float
    
    # Detailed metrics per structure
    # Linear
    cohens_d: float  # separation quality
    variance_explained: float  # by primary direction
    within_class_consistency: float
    
    # Cone
    raw_mean_cosine_similarity: float  # between diff vectors
    positive_correlation_fraction: float  # fraction in same half-space
    
    # Orthogonal
    near_zero_fraction: float  # fraction of near-zero correlations
    
    # Manifold
    pca_top2_variance: float  # variance by top 2 PCs
    local_nonlinearity: float  # curvature measure
    
    # Sparse
    gini_coefficient: float  # inequality of activations
    active_fraction: float  # fraction of active neurons
    top_10_contribution: float  # contribution of top 10 neurons
    
    # Cluster
    best_silhouette: float  # clustering quality
    best_k: int  # optimal number of clusters
    
    # Multi-direction analysis: how many directions needed?
    multi_dir_accuracy_k1: float  # accuracy with 1 direction (same as diff-mean)
    multi_dir_accuracy_k2: float  # accuracy with 2 directions
    multi_dir_accuracy_k3: float  # accuracy with 3 directions
    multi_dir_accuracy_k5: float  # accuracy with 5 directions
    multi_dir_accuracy_k10: float  # accuracy with 10 directions
    multi_dir_min_k_for_good: int  # minimum k where accuracy >= 0.6 (-1 if never)
    multi_dir_saturation_k: int  # k where accuracy stops improving
    multi_dir_gain: float  # gain from using multiple directions vs 1
    
    # Steerability metrics: predict whether CAA steering will work
    # Key insight: TQA has diff_mean_alignment=0.22 (steering works +12%)
    #              HS has diff_mean_alignment=0.05 (steering fails 0%)
    diff_mean_alignment: float  # mean cosine between individual diffs and diff-mean
    pct_positive_alignment: float  # % of pairs where alignment > 0
    steering_vector_norm_ratio: float  # norm of diff-mean / avg individual diff norm
    cluster_direction_angle: float  # angle between k=2 cluster steering directions
    per_cluster_alignment_k2: float  # alignment within k=2 clusters
    spherical_silhouette_k2: float  # silhouette using cosine distance
    effective_steering_dims: int  # how many dimensions explain 90% of diff variance
    steerability_score: float  # overall 0-1 score predicting steering success
    
    # Recommendation
    recommended_method: str
    
    # ICD (Intrinsic Concept Dimensionality) metrics
    icd: float = 0.0  # Effective rank of difference vectors
    icd_top1_variance: float = 0.0  # Variance explained by top direction
    icd_top5_variance: float = 0.0  # Variance explained by top 5 directions
    
    # Nonsense baseline comparison (if computed)
    nonsense_baseline_accuracy: float = 0.5  # Accuracy on random token pairs
    signal_vs_baseline_ratio: float = 1.0  # real_acc / nonsense_acc
    signal_above_baseline: float = 0.0  # real_acc - nonsense_acc
    has_real_signal: bool = False  # True if signal significantly above baseline
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "strategy": self.strategy,
            "layers": self.layers,
            # Step 1: Signal detection
            "signal_strength": self.signal_strength,
            "has_signal": self.has_signal,
            # Step 2: Linearity check
            "linear_probe_accuracy": self.linear_probe_accuracy,
            "is_linear": self.is_linear,
            # NEW: Nonlinear signal metrics
            "nonlinear_metrics": {
                "knn_accuracy_k5": self.knn_accuracy_k5,
                "knn_accuracy_k10": self.knn_accuracy_k10,
                "knn_accuracy_k20": self.knn_accuracy_k20,
                "knn_pca_accuracy": self.knn_pca_accuracy,
                "knn_umap_accuracy": self.knn_umap_accuracy,
                "knn_pacmap_accuracy": self.knn_pacmap_accuracy,
                "mlp_probe_accuracy": self.mlp_probe_accuracy,
                "best_nonlinear_accuracy": self.best_nonlinear_accuracy,
                "mmd_rbf": self.mmd_rbf,
                "local_dim_pos": self.local_dim_pos,
                "local_dim_neg": self.local_dim_neg,
                "local_dim_ratio": self.local_dim_ratio,
                "fisher_max": self.fisher_max,
                "fisher_gini": self.fisher_gini,
                "fisher_top10_ratio": self.fisher_top10_ratio,
                "num_dims_fisher_above_1": self.num_dims_fisher_above_1,
                "density_ratio": self.density_ratio,
            },
            # Step 3: Geometry (only meaningful if has_signal)
            "best_structure": self.best_structure,
            "best_score": self.best_score,
            "structure_scores": {
                "linear": self.linear_score,
                "cone": self.cone_score,
                "orthogonal": self.orthogonal_score,
                "manifold": self.manifold_score,
                "sparse": self.sparse_score,
                "cluster": self.cluster_score,
                "bimodal": self.bimodal_score,
            },
            "linear_details": {
                "cohens_d": self.cohens_d,
                "variance_explained": self.variance_explained,
                "within_class_consistency": self.within_class_consistency,
            },
            "cone_details": {
                "raw_mean_cosine_similarity": self.raw_mean_cosine_similarity,
                "positive_correlation_fraction": self.positive_correlation_fraction,
            },
            "orthogonal_details": {
                "near_zero_fraction": self.near_zero_fraction,
            },
            "manifold_details": {
                "pca_top2_variance": self.pca_top2_variance,
                "local_nonlinearity": self.local_nonlinearity,
            },
            "sparse_details": {
                "gini_coefficient": self.gini_coefficient,
                "active_fraction": self.active_fraction,
                "top_10_contribution": self.top_10_contribution,
            },
            "cluster_details": {
                "best_silhouette": self.best_silhouette,
                "best_k": self.best_k,
            },
            "multi_direction_analysis": {
                "accuracy_k1": self.multi_dir_accuracy_k1,
                "accuracy_k2": self.multi_dir_accuracy_k2,
                "accuracy_k3": self.multi_dir_accuracy_k3,
                "accuracy_k5": self.multi_dir_accuracy_k5,
                "accuracy_k10": self.multi_dir_accuracy_k10,
                "min_k_for_good": self.multi_dir_min_k_for_good,
                "saturation_k": self.multi_dir_saturation_k,
                "gain_from_multi": self.multi_dir_gain,
            },
            "steerability_metrics": {
                "diff_mean_alignment": self.diff_mean_alignment,
                "pct_positive_alignment": self.pct_positive_alignment,
                "steering_vector_norm_ratio": self.steering_vector_norm_ratio,
                "cluster_direction_angle": self.cluster_direction_angle,
                "per_cluster_alignment_k2": self.per_cluster_alignment_k2,
                "spherical_silhouette_k2": self.spherical_silhouette_k2,
                "effective_steering_dims": self.effective_steering_dims,
                "steerability_score": self.steerability_score,
            },
            "icd_metrics": {
                "icd": self.icd,
                "top1_variance": self.icd_top1_variance,
                "top5_variance": self.icd_top5_variance,
            },
            "nonsense_baseline": {
                "nonsense_accuracy": self.nonsense_baseline_accuracy,
                "signal_ratio": self.signal_vs_baseline_ratio,
                "signal_above_baseline": self.signal_above_baseline,
                "has_real_signal": self.has_real_signal,
            },
            "recommended_method": self.recommended_method,
        }


@dataclass
class GeometrySearchResults:
    """Results from a full geometry search."""
    model_name: str
    config: GeometrySearchConfig
    results: List[GeometryTestResult] = field(default_factory=list)
    
    # Timing
    total_time_seconds: float = 0.0
    extraction_time_seconds: float = 0.0
    test_time_seconds: float = 0.0
    
    # Counts
    benchmarks_tested: int = 0
    strategies_tested: int = 0
    layer_combos_tested: int = 0
    
    def add_result(self, result: GeometryTestResult) -> None:
        self.results.append(result)
    
    def get_best_by_linear_score(self, n: int = 10) -> List[GeometryTestResult]:
        """Get top N configurations by linear score."""
        return sorted(self.results, key=lambda r: r.linear_score, reverse=True)[:n]
    
    def get_best_by_structure(self, structure: str, n: int = 10) -> List[GeometryTestResult]:
        """Get top N configurations by a specific structure score."""
        score_attr = f"{structure}_score"
        return sorted(
            self.results, 
            key=lambda r: getattr(r, score_attr, 0.0), 
            reverse=True
        )[:n]
    
    def get_structure_distribution(self) -> Dict[str, int]:
        """Count how many configurations have each structure as best."""
        counts: Dict[str, int] = {}
        for r in self.results:
            s = r.best_structure
            counts[s] = counts.get(s, 0) + 1
        return counts
    
    def get_summary_by_benchmark(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics grouped by benchmark."""
        by_bench: Dict[str, List[float]] = {}
        for r in self.results:
            if r.benchmark not in by_bench:
                by_bench[r.benchmark] = []
            by_bench[r.benchmark].append(r.linear_score)
        
        return {
            bench: {
                "mean": sum(scores) / len(scores),
                "max": max(scores),
                "min": min(scores),
                "count": len(scores),
            }
            for bench, scores in by_bench.items()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "config": self.config.to_dict(),
            "total_time_seconds": self.total_time_seconds,
            "extraction_time_seconds": self.extraction_time_seconds,
            "test_time_seconds": self.test_time_seconds,
            "benchmarks_tested": self.benchmarks_tested,
            "strategies_tested": self.strategies_tested,
            "layer_combos_tested": self.layer_combos_tested,
            "results": [r.to_dict() for r in self.results],
        }
    
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "GeometrySearchResults":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        config = GeometrySearchConfig(
            pairs_per_benchmark=data.get("config", {}).get("pairs_per_benchmark", 50),
            max_layer_combo_size=data.get("config", {}).get("max_layer_combo_size", 3),
            random_seed=data.get("config", {}).get("random_seed", 42),
        )
        
        results = cls(
            model_name=data.get("model_name", "unknown"),
            config=config,
            total_time_seconds=data.get("total_time_seconds", 0.0),
            extraction_time_seconds=data.get("extraction_time_seconds", 0.0),
            test_time_seconds=data.get("test_time_seconds", 0.0),
            benchmarks_tested=data.get("benchmarks_tested", 0),
            strategies_tested=data.get("strategies_tested", 0),
            layer_combos_tested=data.get("layer_combos_tested", 0),
        )
        
        # Load individual test results
        for r_data in data.get("results", []):
            try:
                result = GeometryTestResult(
                    benchmark=r_data.get("benchmark", "unknown"),
                    strategy=r_data.get("strategy", "unknown"),
                    layers=r_data.get("layers", []),
                    n_samples=r_data.get("n_samples", 0),
                    signal_strength=r_data.get("signal_strength", 0.5),
                    linear_score=r_data.get("linear_score", 0.0),
                    linear_probe_accuracy=r_data.get("linear_probe_accuracy", 0.5),
                    best_structure=r_data.get("best_structure", "unknown"),
                    is_linear=r_data.get("is_linear", False),
                    cohens_d=r_data.get("cohens_d", 0.0),
                    knn_accuracy_k5=r_data.get("nonlinear_signals", {}).get("knn_accuracy_k5", 0.5),
                    knn_accuracy_k10=r_data.get("nonlinear_signals", {}).get("knn_accuracy_k10", 0.5),
                    knn_accuracy_k20=r_data.get("nonlinear_signals", {}).get("knn_accuracy_k20", 0.5),
                    knn_pca_accuracy=r_data.get("nonlinear_signals", {}).get("knn_pca_accuracy", 0.5),
                    knn_umap_accuracy=r_data.get("nonlinear_signals", {}).get("knn_umap_accuracy", 0.5),
                    knn_pacmap_accuracy=r_data.get("nonlinear_signals", {}).get("knn_pacmap_accuracy", 0.5),
                    mlp_probe_accuracy=r_data.get("nonlinear_signals", {}).get("mlp_probe_accuracy", 0.5),
                    mmd_rbf=r_data.get("nonlinear_signals", {}).get("mmd_rbf", 0.0),
                    local_dim_pos=r_data.get("nonlinear_signals", {}).get("local_dim_pos", 0.0),
                    local_dim_neg=r_data.get("nonlinear_signals", {}).get("local_dim_neg", 0.0),
                    local_dim_ratio=r_data.get("nonlinear_signals", {}).get("local_dim_ratio", 1.0),
                    fisher_max=r_data.get("nonlinear_signals", {}).get("fisher_max", 0.0),
                    fisher_mean=r_data.get("nonlinear_signals", {}).get("fisher_mean", 0.0),
                    fisher_top10_mean=r_data.get("nonlinear_signals", {}).get("fisher_top10_mean", 0.0),
                    density_ratio=r_data.get("nonlinear_signals", {}).get("density_ratio", 1.0),
                    manifold_score=r_data.get("manifold_score", 0.0),
                    cluster_score=r_data.get("cluster_score", 0.0),
                    sparse_score=r_data.get("sparse_score", 0.0),
                    hybrid_score=r_data.get("hybrid_score", 0.0),
                    all_scores={},
                    pca_top2_variance=r_data.get("manifold_details", {}).get("pca_top2_variance", 0.0),
                    local_nonlinearity=r_data.get("manifold_details", {}).get("local_nonlinearity", 0.0),
                    gini_coefficient=r_data.get("sparse_details", {}).get("gini_coefficient", 0.0),
                    active_fraction=r_data.get("sparse_details", {}).get("active_fraction", 0.0),
                    top_10_contribution=r_data.get("sparse_details", {}).get("top_10_contribution", 0.0),
                    best_silhouette=r_data.get("cluster_details", {}).get("best_silhouette", 0.0),
                    best_k=r_data.get("cluster_details", {}).get("best_k", 2),
                    multi_dir_accuracy_k1=r_data.get("multi_direction_analysis", {}).get("accuracy_k1", 0.5),
                    multi_dir_accuracy_k2=r_data.get("multi_direction_analysis", {}).get("accuracy_k2", 0.5),
                    multi_dir_accuracy_k3=r_data.get("multi_direction_analysis", {}).get("accuracy_k3", 0.5),
                    multi_dir_accuracy_k5=r_data.get("multi_direction_analysis", {}).get("accuracy_k5", 0.5),
                    multi_dir_accuracy_k10=r_data.get("multi_direction_analysis", {}).get("accuracy_k10", 0.5),
                    multi_dir_min_k_for_good=r_data.get("multi_direction_analysis", {}).get("min_k_for_good", -1),
                    multi_dir_saturation_k=r_data.get("multi_direction_analysis", {}).get("saturation_k", 1),
                    multi_dir_gain=r_data.get("multi_direction_analysis", {}).get("gain_from_multi", 0.0),
                    diff_mean_alignment=r_data.get("steerability_metrics", {}).get("diff_mean_alignment", 0.0),
                    pct_positive_alignment=r_data.get("steerability_metrics", {}).get("pct_positive_alignment", 0.5),
                    steering_vector_norm_ratio=r_data.get("steerability_metrics", {}).get("steering_vector_norm_ratio", 0.0),
                    cluster_direction_angle=r_data.get("steerability_metrics", {}).get("cluster_direction_angle", 90.0),
                    per_cluster_alignment_k2=r_data.get("steerability_metrics", {}).get("per_cluster_alignment_k2", 0.0),
                    spherical_silhouette_k2=r_data.get("steerability_metrics", {}).get("spherical_silhouette_k2", 0.0),
                    effective_steering_dims=r_data.get("steerability_metrics", {}).get("effective_steering_dims", 1),
                    steerability_score=r_data.get("steerability_metrics", {}).get("steerability_score", 0.0),
                    icd=r_data.get("icd_metrics", {}).get("icd", 0.0),
                    icd_top1_variance=r_data.get("icd_metrics", {}).get("top1_variance", 0.0),
                    icd_top5_variance=r_data.get("icd_metrics", {}).get("top5_variance", 0.0),
                    nonsense_baseline_accuracy=r_data.get("nonsense_baseline", {}).get("nonsense_accuracy", 0.5),
                    signal_vs_baseline_ratio=r_data.get("nonsense_baseline", {}).get("signal_ratio", 1.0),
                    signal_above_baseline=r_data.get("nonsense_baseline", {}).get("signal_above_baseline", 0.0),
                    has_real_signal=r_data.get("nonsense_baseline", {}).get("has_real_signal", True),
                    recommended_method=r_data.get("recommended_method", "unknown"),
                )
                results.results.append(result)
            except Exception:
                pass
        
        return results


def compute_recommendation(
    # Classification metrics (on raw activations)
    linear_probe_accuracy: float,
    best_nonlinear_accuracy: float,
    knn_umap_accuracy: float,
    knn_pacmap_accuracy: float,
    # Diff-based metrics (on difference vectors)
    icd: float,
    icd_top1_variance: float,
    diff_mean_alignment: float,
    steerability_score: float,
    effective_steering_dims: int,
    # Multi-direction metrics
    multi_dir_gain: float,
    spherical_silhouette_k2: float,
    cluster_direction_angle: float,
    # Quality metrics
    cohens_d: float,
    signal_above_baseline: float,
) -> Dict[str, Any]:
    """
    Compute steering method recommendation using ALL available metrics.
    
    Flow:
    1. Check if signal exists (classification metrics)
    2. Determine signal type (linear vs nonlinear vs multimodal)
    3. Check steerability (diff-based metrics)
    4. Recommend method
    
    Returns:
        Dict with:
            - signal_exists: bool
            - signal_type: LINEAR / NONLINEAR / MULTIMODAL / NO_SIGNAL
            - recommended_method: CAA / PRISM / PULSE / TITAN / NO_METHOD
            - confidence: 0-1 confidence in recommendation
            - evidence: Dict explaining the decision
    """
    
    # Thresholds
    SIGNAL_THRESHOLD = 0.6
    LINEAR_GAP_THRESHOLD = 0.1
    ICD_LOW_THRESHOLD = 5.0
    ALIGNMENT_THRESHOLD = 0.15
    STEERABILITY_THRESHOLD = 0.4
    MULTI_DIR_GAIN_THRESHOLD = 0.1
    CLUSTER_ANGLE_THRESHOLD = 45.0
    
    evidence = {
        "linear_probe": linear_probe_accuracy,
        "best_nonlinear": best_nonlinear_accuracy,
        "knn_umap": knn_umap_accuracy,
        "knn_pacmap": knn_pacmap_accuracy,
        "icd": icd,
        "icd_top1_variance": icd_top1_variance,
        "diff_mean_alignment": diff_mean_alignment,
        "steerability_score": steerability_score,
        "effective_steering_dims": effective_steering_dims,
        "multi_dir_gain": multi_dir_gain,
        "spherical_silhouette_k2": spherical_silhouette_k2,
        "cohens_d": cohens_d,
        "signal_above_baseline": signal_above_baseline,
    }
    
    # Step 1: Check if signal exists
    if best_nonlinear_accuracy < SIGNAL_THRESHOLD:
        return {
            "signal_exists": False,
            "signal_type": "NO_SIGNAL",
            "recommended_method": "NO_METHOD",
            "confidence": 1.0 - best_nonlinear_accuracy,  # More confident if accuracy is lower
            "evidence": evidence,
            "reason": f"No separable signal (best_nonlinear={best_nonlinear_accuracy:.2f} < {SIGNAL_THRESHOLD})",
        }
    
    # Step 2: Determine signal type
    linear_gap = best_nonlinear_accuracy - linear_probe_accuracy
    
    # Check if signal is LINEAR
    is_linear_separable = linear_gap <= LINEAR_GAP_THRESHOLD
    is_low_icd = icd < ICD_LOW_THRESHOLD
    is_concentrated = icd_top1_variance > 0.5
    
    # Check if signal is MULTIMODAL (multiple clusters/directions)
    has_multi_dir_gain = multi_dir_gain > MULTI_DIR_GAIN_THRESHOLD
    has_clusters = spherical_silhouette_k2 > 0.2
    needs_multiple_dirs = effective_steering_dims > 1
    
    # Check if signal is context-dependent
    is_context_dependent = cluster_direction_angle > CLUSTER_ANGLE_THRESHOLD
    
    if is_linear_separable and is_low_icd and is_concentrated:
        signal_type = "LINEAR"
    elif has_multi_dir_gain or (has_clusters and needs_multiple_dirs):
        signal_type = "MULTIMODAL"
    elif linear_gap > LINEAR_GAP_THRESHOLD:
        signal_type = "NONLINEAR"
    else:
        signal_type = "LINEAR"  # Default to linear if unclear
    
    # Step 3: Check steerability (diff-based metrics)
    is_steerable = diff_mean_alignment > ALIGNMENT_THRESHOLD
    has_good_steerability = steerability_score > STEERABILITY_THRESHOLD
    
    # Step 4: Recommend method based on signal type and steerability
    if signal_type == "LINEAR":
        if is_steerable and has_good_steerability:
            recommended_method = "CAA"
            confidence = min(1.0, steerability_score + 0.3)
            reason = f"Linear signal with good steerability (alignment={diff_mean_alignment:.2f}, score={steerability_score:.2f})"
        elif is_steerable:
            recommended_method = "CAA"
            confidence = 0.6
            reason = f"Linear signal, moderate steerability (alignment={diff_mean_alignment:.2f})"
        else:
            # Linear but poor alignment - may need PRISM
            recommended_method = "PRISM"
            confidence = 0.5
            reason = f"Linear but poor alignment ({diff_mean_alignment:.2f}), try multi-direction"
    
    elif signal_type == "MULTIMODAL":
        if is_context_dependent:
            recommended_method = "PULSE"
            confidence = 0.7
            reason = f"Multimodal + context-dependent (cluster_angle={cluster_direction_angle:.1f}°)"
        else:
            recommended_method = "PRISM"
            confidence = 0.8
            reason = f"Multimodal signal (multi_dir_gain={multi_dir_gain:.2f}, k={effective_steering_dims})"
    
    elif signal_type == "NONLINEAR":
        # Nonlinear but single direction
        if needs_multiple_dirs:
            recommended_method = "PRISM"
            confidence = 0.6
            reason = f"Nonlinear with multiple directions needed (k={effective_steering_dims})"
        else:
            recommended_method = "TITAN"
            confidence = 0.5
            reason = f"Complex nonlinear structure, try TITAN"
    
    else:
        recommended_method = "CAA"
        confidence = 0.4
        reason = "Unclear signal type, defaulting to CAA"
    
    return {
        "signal_exists": True,
        "signal_type": signal_type,
        "recommended_method": recommended_method,
        "confidence": confidence,
        "evidence": evidence,
        "reason": reason,
    }


def should_increase_pairs(
    best_nonlinear_accuracy: float,
    n_pairs: int,
    max_pairs: int,
    signal_threshold: float = 0.6,
) -> bool:
    """
    Determine if we should try with more contrastive pairs.
    
    When signal is weak but not definitively absent, more pairs might help.
    
    Args:
        best_nonlinear_accuracy: Best accuracy from any method
        n_pairs: Current number of pairs
        max_pairs: Maximum pairs available in dataset
        signal_threshold: Threshold for signal existence
        
    Returns:
        True if should increase pairs, False otherwise
    """
    # Don't increase if we've already used all pairs
    if n_pairs >= max_pairs:
        return False
    
    # Don't increase if signal is clearly present
    if best_nonlinear_accuracy >= signal_threshold:
        return False
    
    # Don't increase if signal is clearly absent (random)
    if best_nonlinear_accuracy < 0.52:
        return False
    
    # Increase if signal is borderline (might improve with more data)
    # Range: 0.52 - 0.6 is the "uncertain" zone
    return True


def compute_adaptive_recommendation(
    compute_metrics_fn,
    initial_n_pairs: int = 50,
    max_pairs: int = 500,
    pair_multiplier: float = 2.0,
    signal_threshold: float = 0.6,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute recommendation with adaptive pair count.
    
    If initial pairs don't show signal, automatically increase and retry.
    Stops when:
    - Signal is found (best_nonlinear >= threshold)
    - Signal is definitely absent (best_nonlinear < 0.52)
    - Max pairs reached
    
    Args:
        compute_metrics_fn: Function that takes n_pairs and returns metrics dict
                           Must return dict with 'best_nonlinear_accuracy' key
        initial_n_pairs: Starting number of pairs
        max_pairs: Maximum pairs to try
        pair_multiplier: Multiply pairs by this each iteration
        signal_threshold: Threshold for signal detection
        verbose: Print progress
        
    Returns:
        Dict with:
            - final_n_pairs: Number of pairs used
            - iterations: Number of attempts
            - metrics_history: List of (n_pairs, best_acc) tuples
            - final_recommendation: Result from compute_recommendation
            - reason: Why stopped
    """
    n_pairs = initial_n_pairs
    iterations = 0
    metrics_history = []
    
    while n_pairs <= max_pairs:
        iterations += 1
        
        if verbose:
            print(f"  Attempt {iterations}: n_pairs={n_pairs}")
        
        # Compute metrics with current pair count
        try:
            metrics = compute_metrics_fn(n_pairs)
        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            break
        
        best_acc = metrics.get('best_nonlinear_accuracy', 0.5)
        metrics_history.append((n_pairs, best_acc))
        
        if verbose:
            print(f"    best_nonlinear={best_acc:.3f}")
        
        # Check if we should stop or continue
        if best_acc >= signal_threshold:
            # Signal found!
            reason = f"Signal found with {n_pairs} pairs (acc={best_acc:.3f})"
            break
        elif best_acc < 0.52:
            # Definitely no signal
            reason = f"No signal detected (acc={best_acc:.3f} ≈ random)"
            break
        elif n_pairs >= max_pairs:
            # Exhausted pairs
            reason = f"Max pairs reached ({max_pairs}), weak signal (acc={best_acc:.3f})"
            break
        else:
            # Increase pairs and retry
            n_pairs = min(int(n_pairs * pair_multiplier), max_pairs)
    
    # Compute final recommendation
    if metrics_history:
        final_n_pairs, final_acc = metrics_history[-1]
        
        # Get full metrics for recommendation
        final_metrics = compute_metrics_fn(final_n_pairs)
        
        recommendation = compute_recommendation(
            linear_probe_accuracy=final_metrics.get('linear_probe_accuracy', 0.5),
            best_nonlinear_accuracy=final_metrics.get('best_nonlinear_accuracy', 0.5),
            knn_umap_accuracy=final_metrics.get('knn_umap_accuracy', 0.5),
            knn_pacmap_accuracy=final_metrics.get('knn_pacmap_accuracy', 0.5),
            icd=final_metrics.get('icd', 0.0),
            icd_top1_variance=final_metrics.get('icd_top1_variance', 0.0),
            diff_mean_alignment=final_metrics.get('diff_mean_alignment', 0.0),
            steerability_score=final_metrics.get('steerability_score', 0.0),
            effective_steering_dims=final_metrics.get('effective_steering_dims', 1),
            multi_dir_gain=final_metrics.get('multi_dir_gain', 0.0),
            spherical_silhouette_k2=final_metrics.get('spherical_silhouette_k2', 0.0),
            cluster_direction_angle=final_metrics.get('cluster_direction_angle', 90.0),
            cohens_d=final_metrics.get('cohens_d', 0.0),
            signal_above_baseline=final_metrics.get('signal_above_baseline', 0.0),
        )
    else:
        recommendation = {
            "signal_exists": False,
            "signal_type": "ERROR",
            "recommended_method": "NO_METHOD",
            "confidence": 0.0,
            "evidence": {},
            "reason": "No metrics computed",
        }
        reason = "No metrics computed"
        final_n_pairs = initial_n_pairs
    
    return {
        "final_n_pairs": final_n_pairs,
        "iterations": iterations,
        "metrics_history": metrics_history,
        "final_recommendation": recommendation,
        "reason": reason,
    }


def compute_bootstrap_signal_estimate(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_bootstrap: int = 10,
    sample_fraction: float = 0.8,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Estimate signal strength with bootstrap sampling to measure uncertainty.
    
    Instead of computing metrics on a single sample, we:
    1. Subsample 80% of pairs multiple times
    2. Compute metrics on each subsample
    3. Return mean and std of results
    
    This helps distinguish:
    - Stable signal: low variance across bootstraps
    - Unstable/borderline signal: high variance
    - No signal: consistently low accuracy
    
    Args:
        pos_activations: Positive class activations [n_pairs, hidden_size]
        neg_activations: Negative class activations [n_pairs, hidden_size]
        n_bootstrap: Number of bootstrap samples
        sample_fraction: Fraction of pairs to use in each bootstrap
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict with:
            - linear_probe_mean/std: Linear probe statistics
            - best_nonlinear_mean/std: Best nonlinear probe statistics
            - signal_stable: True if std < 0.05
            - recommendation_confidence: 1 - (std / mean) 
    """
    import numpy as np
    
    n_pairs = pos_activations.shape[0]
    n_sample = max(10, int(n_pairs * sample_fraction))
    
    rng = np.random.RandomState(random_seed)
    
    linear_scores = []
    nonlinear_scores = []
    knn_umap_scores = []
    knn_pacmap_scores = []
    
    for i in range(n_bootstrap):
        # Random subsample
        indices = rng.choice(n_pairs, size=n_sample, replace=False)
        pos_sub = pos_activations[indices]
        neg_sub = neg_activations[indices]
        
        # Compute probes
        linear = compute_linear_probe_accuracy(pos_sub, neg_sub)
        knn = compute_knn_accuracy(pos_sub, neg_sub, k=10)
        knn_pca = compute_knn_pca_accuracy(pos_sub, neg_sub, k=10, n_components=min(50, n_sample - 1))
        
        # UMAP/PaCMAP can be slow, compute less frequently
        if i < 3:  # Only first 3 bootstraps
            knn_umap = compute_knn_umap_accuracy(pos_sub, neg_sub, k=10, n_components=10)
            knn_pacmap = compute_knn_pacmap_accuracy(pos_sub, neg_sub, k=10, n_components=10)
            knn_umap_scores.append(knn_umap)
            knn_pacmap_scores.append(knn_pacmap)
        
        mlp = compute_mlp_probe_accuracy(pos_sub, neg_sub, hidden_size=64)
        
        linear_scores.append(linear)
        nonlinear_scores.append(max(knn, knn_pca, mlp))
    
    # Compute statistics
    linear_mean = np.mean(linear_scores)
    linear_std = np.std(linear_scores)
    nonlinear_mean = np.mean(nonlinear_scores)
    nonlinear_std = np.std(nonlinear_scores)
    
    knn_umap_mean = np.mean(knn_umap_scores) if knn_umap_scores else 0.5
    knn_pacmap_mean = np.mean(knn_pacmap_scores) if knn_pacmap_scores else 0.5
    
    # Signal is stable if variance is low
    signal_stable = nonlinear_std < 0.05
    
    # Confidence: higher when mean is high and std is low
    if nonlinear_mean > 0.5:
        recommendation_confidence = max(0, min(1, 1 - (nonlinear_std / (nonlinear_mean - 0.5 + 0.01))))
    else:
        recommendation_confidence = 0.0
    
    return {
        "linear_probe_mean": float(linear_mean),
        "linear_probe_std": float(linear_std),
        "best_nonlinear_mean": float(nonlinear_mean),
        "best_nonlinear_std": float(nonlinear_std),
        "knn_umap_mean": float(knn_umap_mean),
        "knn_pacmap_mean": float(knn_pacmap_mean),
        "signal_stable": signal_stable,
        "recommendation_confidence": float(recommendation_confidence),
        "n_bootstrap": n_bootstrap,
        "sample_fraction": sample_fraction,
        "bootstrap_scores": {
            "linear": linear_scores,
            "nonlinear": nonlinear_scores,
        }
    }


def compute_robust_recommendation(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_bootstrap: int = 5,
    sample_fraction: float = 0.8,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compute recommendation with bootstrap-based uncertainty estimation.
    
    This is more reliable than single-sample estimation because it:
    1. Measures variance across different subsamples
    2. Only recommends methods when signal is stable
    3. Suggests increasing pairs if signal is unstable
    
    Args:
        pos_activations: Positive activations [n_pairs, hidden_dim]
        neg_activations: Negative activations [n_pairs, hidden_dim]
        n_bootstrap: Number of bootstrap iterations
        sample_fraction: Fraction to sample each iteration
        verbose: Print progress
        
    Returns:
        Dict with recommendation and uncertainty metrics
    """
    n_pairs = pos_activations.shape[0]
    
    if verbose:
        print(f"Computing robust recommendation with {n_bootstrap} bootstraps on {n_pairs} pairs...")
    
    # Get bootstrap estimates
    bootstrap_results = compute_bootstrap_signal_estimate(
        pos_activations, neg_activations,
        n_bootstrap=n_bootstrap,
        sample_fraction=sample_fraction,
    )
    
    # Also compute full metrics for geometry analysis
    steerability = compute_steerability_metrics(pos_activations, neg_activations)
    icd_results = compute_icd(pos_activations, neg_activations)
    multi_dir = compute_multi_direction_accuracy(pos_activations, neg_activations)
    
    # Use bootstrap means for recommendation
    recommendation = compute_recommendation(
        linear_probe_accuracy=bootstrap_results["linear_probe_mean"],
        best_nonlinear_accuracy=bootstrap_results["best_nonlinear_mean"],
        knn_umap_accuracy=bootstrap_results["knn_umap_mean"],
        knn_pacmap_accuracy=bootstrap_results["knn_pacmap_mean"],
        icd=icd_results["icd"],
        icd_top1_variance=icd_results["top1_variance"],
        diff_mean_alignment=steerability["diff_mean_alignment"],
        steerability_score=steerability["steerability_score"],
        effective_steering_dims=steerability["effective_steering_dims"],
        multi_dir_gain=multi_dir["gain_from_multi"],
        spherical_silhouette_k2=steerability["spherical_silhouette_k2"],
        cluster_direction_angle=steerability["cluster_direction_angle"],
        cohens_d=0.0,  # Would need to compute
        signal_above_baseline=0.0,  # Would need nonsense baseline
    )
    
    # Adjust confidence based on bootstrap stability
    adjusted_confidence = recommendation["confidence"] * bootstrap_results["recommendation_confidence"]
    
    # Determine if we should suggest more pairs
    suggest_more_pairs = (
        not bootstrap_results["signal_stable"] and 
        bootstrap_results["best_nonlinear_mean"] > 0.52 and
        bootstrap_results["best_nonlinear_mean"] < 0.7
    )
    
    return {
        "recommendation": recommendation,
        "bootstrap": bootstrap_results,
        "adjusted_confidence": adjusted_confidence,
        "suggest_more_pairs": suggest_more_pairs,
        "reason": (
            f"Signal={'STABLE' if bootstrap_results['signal_stable'] else 'UNSTABLE'} "
            f"(mean={bootstrap_results['best_nonlinear_mean']:.3f}, "
            f"std={bootstrap_results['best_nonlinear_std']:.3f})"
        ),
    }


def compute_direction_from_pairs(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> torch.Tensor:
    """Compute steering direction as normalized diff-mean."""
    diff = pos_activations.mean(dim=0) - neg_activations.mean(dim=0)
    return diff / (diff.norm() + 1e-8)


def compute_saturation_check(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    previous_accuracy: float,
    previous_direction: torch.Tensor | None,
    accuracy_threshold: float = 0.02,
    direction_threshold: float = 0.95,
    variance_threshold: float = 0.03,
    n_bootstrap: int = 5,
) -> Dict[str, Any]:
    """
    Check if adding more pairs would improve results (saturation detection).
    
    This answers: "We have strong signal with N pairs - should we get more?"
    
    Saturation is reached when:
    1. Accuracy improvement < threshold (e.g., < 2%)
    2. Direction is stable (cosine similarity > 0.95)
    3. Bootstrap variance is low (< 0.03)
    
    Args:
        pos_activations: Current positive activations
        neg_activations: Current negative activations
        previous_accuracy: Accuracy from previous (smaller) sample
        previous_direction: Direction vector from previous sample
        accuracy_threshold: Min accuracy improvement to continue
        direction_threshold: Min cosine similarity for direction stability
        variance_threshold: Max bootstrap std for stability
        n_bootstrap: Number of bootstrap samples for variance
        
    Returns:
        Dict with saturation status and metrics
    """
    import numpy as np
    
    n_pairs = pos_activations.shape[0]
    
    # Current metrics
    current_accuracy = compute_linear_probe_accuracy(pos_activations, neg_activations)
    current_direction = compute_direction_from_pairs(pos_activations, neg_activations)
    
    # Bootstrap for variance estimation
    bootstrap = compute_bootstrap_signal_estimate(
        pos_activations, neg_activations,
        n_bootstrap=n_bootstrap,
        sample_fraction=0.8,
    )
    
    # Check saturation criteria
    accuracy_saturated = abs(current_accuracy - previous_accuracy) < accuracy_threshold
    
    if previous_direction is not None:
        direction_similarity = torch.nn.functional.cosine_similarity(
            current_direction.unsqueeze(0),
            previous_direction.unsqueeze(0)
        ).item()
        direction_stable = direction_similarity > direction_threshold
    else:
        direction_similarity = 0.0
        direction_stable = False
    
    variance_low = bootstrap["best_nonlinear_std"] < variance_threshold
    
    # Saturated if all criteria met
    is_saturated = accuracy_saturated and direction_stable and variance_low
    
    # Recommendation
    if is_saturated:
        recommendation = "STOP"
        reason = f"Saturated: acc_delta={abs(current_accuracy - previous_accuracy):.3f}, dir_sim={direction_similarity:.3f}, std={bootstrap['best_nonlinear_std']:.3f}"
    elif not accuracy_saturated:
        recommendation = "CONTINUE"
        reason = f"Accuracy still improving: {previous_accuracy:.3f} -> {current_accuracy:.3f}"
    elif not direction_stable:
        recommendation = "CONTINUE"
        reason = f"Direction unstable: similarity={direction_similarity:.3f}"
    else:
        recommendation = "CONTINUE"
        reason = f"Variance still high: std={bootstrap['best_nonlinear_std']:.3f}"
    
    return {
        "is_saturated": is_saturated,
        "recommendation": recommendation,
        "reason": reason,
        "n_pairs": n_pairs,
        "current_accuracy": current_accuracy,
        "previous_accuracy": previous_accuracy,
        "accuracy_delta": abs(current_accuracy - previous_accuracy),
        "direction_similarity": direction_similarity,
        "bootstrap_std": bootstrap["best_nonlinear_std"],
        "current_direction": current_direction,
        "criteria": {
            "accuracy_saturated": accuracy_saturated,
            "direction_stable": direction_stable,
            "variance_low": variance_low,
        }
    }


def find_optimal_pair_count(
    get_activations_fn,
    pair_counts: list[int] = [25, 50, 100, 200, 400],
    accuracy_threshold: float = 0.02,
    direction_threshold: float = 0.95,
    variance_threshold: float = 0.03,
    min_signal_threshold: float = 0.6,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Find optimal number of pairs by detecting saturation.
    
    Process:
    1. Start with smallest pair count
    2. If no signal -> increase pairs
    3. If signal found -> check saturation
    4. If saturated -> stop, else increase
    
    Args:
        get_activations_fn: Function(n_pairs) -> (pos_activations, neg_activations)
        pair_counts: List of pair counts to try (ascending order)
        accuracy_threshold: Threshold for accuracy saturation
        direction_threshold: Threshold for direction stability
        variance_threshold: Threshold for variance
        min_signal_threshold: Minimum accuracy to consider signal exists
        verbose: Print progress
        
    Returns:
        Dict with optimal pair count and history
    """
    history = []
    previous_accuracy = 0.5
    previous_direction = None
    optimal_n_pairs = pair_counts[0]
    
    for n_pairs in pair_counts:
        if verbose:
            print(f"Testing n_pairs={n_pairs}...")
        
        try:
            pos_act, neg_act = get_activations_fn(n_pairs)
        except Exception as e:
            if verbose:
                print(f"  Error getting activations: {e}")
            break
        
        # Check current accuracy
        current_accuracy = compute_linear_probe_accuracy(pos_act, neg_act)
        current_direction = compute_direction_from_pairs(pos_act, neg_act)
        
        if verbose:
            print(f"  Accuracy: {current_accuracy:.3f}")
        
        # If no signal yet, continue to more pairs
        if current_accuracy < min_signal_threshold:
            if verbose:
                print(f"  No signal yet (acc < {min_signal_threshold}), continuing...")
            previous_accuracy = current_accuracy
            previous_direction = current_direction
            history.append({
                "n_pairs": n_pairs,
                "accuracy": current_accuracy,
                "status": "NO_SIGNAL",
            })
            continue
        
        # Signal found - check saturation
        saturation = compute_saturation_check(
            pos_act, neg_act,
            previous_accuracy=previous_accuracy,
            previous_direction=previous_direction,
            accuracy_threshold=accuracy_threshold,
            direction_threshold=direction_threshold,
            variance_threshold=variance_threshold,
        )
        
        history.append({
            "n_pairs": n_pairs,
            "accuracy": current_accuracy,
            "status": "SATURATED" if saturation["is_saturated"] else "IMPROVING",
            "direction_similarity": saturation["direction_similarity"],
            "bootstrap_std": saturation["bootstrap_std"],
        })
        
        if saturation["is_saturated"]:
            optimal_n_pairs = n_pairs
            if verbose:
                print(f"  SATURATED: {saturation['reason']}")
            break
        else:
            if verbose:
                print(f"  Not saturated: {saturation['reason']}")
            optimal_n_pairs = n_pairs
            previous_accuracy = current_accuracy
            previous_direction = current_direction
    
    # Determine final status
    if history:
        final_status = history[-1]["status"]
        if final_status == "NO_SIGNAL":
            recommendation = "NO_SIGNAL"
        elif final_status == "SATURATED":
            recommendation = "OPTIMAL"
        else:
            recommendation = "MAX_REACHED"
    else:
        recommendation = "ERROR"
    
    return {
        "optimal_n_pairs": optimal_n_pairs,
        "recommendation": recommendation,
        "history": history,
        "final_accuracy": history[-1]["accuracy"] if history else 0.5,
    }


@dataclass
class RepScanResult:
    """Complete result from run_full_repscan()."""
    # Basic info
    benchmark: str
    model_name: str
    layer: int
    
    # Pair optimization
    optimal_n_pairs: int
    pair_search_history: List[Dict[str, Any]]
    saturation_status: str  # "OPTIMAL", "MAX_REACHED", "NO_SIGNAL"
    
    # Signal classification
    signal_exists: bool
    signal_type: str  # "LINEAR", "NONLINEAR", "MULTIMODAL", "NO_SIGNAL"
    
    # Method recommendation
    recommended_method: str  # "CAA", "PRISM", "PULSE", "TITAN", "NO_METHOD"
    confidence: float
    reason: str
    
    # Core metrics
    linear_probe_accuracy: float
    best_nonlinear_accuracy: float
    knn_umap_accuracy: float
    knn_pacmap_accuracy: float
    
    # Geometry metrics (diff vectors)
    icd: float
    icd_top1_variance: float
    diff_mean_alignment: float
    caa_probe_alignment: float  # NEW: cosine(CAA_direction, probe_direction) - key predictor!
    steerability_score: float
    effective_steering_dims: int
    
    # Multi-direction metrics
    multi_dir_gain: float
    spherical_silhouette_k2: float
    cluster_direction_angle: float
    
    # Quality metrics
    cohens_d: float
    bootstrap_std: float
    direction_stability: float  # cosine similarity between bootstraps
    
    # Steering vector (if signal exists)
    steering_direction: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding tensor)."""
        d = {
            "benchmark": self.benchmark,
            "model_name": self.model_name,
            "layer": self.layer,
            "optimal_n_pairs": self.optimal_n_pairs,
            "pair_search_history": self.pair_search_history,
            "saturation_status": self.saturation_status,
            "signal_exists": self.signal_exists,
            "signal_type": self.signal_type,
            "recommended_method": self.recommended_method,
            "confidence": self.confidence,
            "reason": self.reason,
            "linear_probe_accuracy": self.linear_probe_accuracy,
            "best_nonlinear_accuracy": self.best_nonlinear_accuracy,
            "knn_umap_accuracy": self.knn_umap_accuracy,
            "knn_pacmap_accuracy": self.knn_pacmap_accuracy,
            "icd": self.icd,
            "icd_top1_variance": self.icd_top1_variance,
            "diff_mean_alignment": self.diff_mean_alignment,
            "caa_probe_alignment": self.caa_probe_alignment,
            "steerability_score": self.steerability_score,
            "effective_steering_dims": self.effective_steering_dims,
            "multi_dir_gain": self.multi_dir_gain,
            "spherical_silhouette_k2": self.spherical_silhouette_k2,
            "cluster_direction_angle": self.cluster_direction_angle,
            "cohens_d": self.cohens_d,
            "bootstrap_std": self.bootstrap_std,
            "direction_stability": self.direction_stability,
        }
        return d


def run_full_repscan(
    get_activations_fn,
    benchmark: str = "unknown",
    model_name: str = "unknown", 
    layer: int = -1,
    pair_counts: List[int] = [50, 100, 200, 400],
    n_bootstrap: int = 5,
    accuracy_threshold: float = 0.02,
    direction_threshold: float = 0.95,
    variance_threshold: float = 0.03,
    min_signal_threshold: float = 0.6,
    random_signal_threshold: float = 0.52,
    verbose: bool = True,
) -> RepScanResult:
    """
    Run complete RepScan analysis: find optimal pairs, classify signal, recommend method.
    
    This is the main entry point that combines all RepScan functionality:
    1. Find optimal number of pairs (with saturation detection)
    2. Compute all metrics (14 metrics across classification/geometry/multi-dir)
    3. Classify signal type (LINEAR/NONLINEAR/MULTIMODAL/NO_SIGNAL)
    4. Recommend steering method (CAA/PRISM/PULSE/TITAN/NO_METHOD)
    
    Args:
        get_activations_fn: Function(n_pairs) -> (pos_activations, neg_activations)
                           Returns tensors of shape [n_pairs, hidden_dim]
        benchmark: Name of benchmark being analyzed
        model_name: Name of model being analyzed
        layer: Layer number being analyzed
        pair_counts: List of pair counts to try (ascending order)
        n_bootstrap: Number of bootstrap samples for variance estimation
        accuracy_threshold: Threshold for accuracy saturation (default: 0.02)
        direction_threshold: Threshold for direction stability (default: 0.95)
        variance_threshold: Threshold for variance (default: 0.03)
        min_signal_threshold: Minimum accuracy to consider signal exists (default: 0.6)
        random_signal_threshold: Below this, signal is definitively random (default: 0.52)
        verbose: Print progress
        
    Returns:
        RepScanResult with all metrics, classification, and recommendation
        
    Example:
        ```python
        def get_acts(n):
            # Your activation extraction logic
            return pos_activations, neg_activations
            
        result = run_full_repscan(
            get_acts,
            benchmark="truthfulqa",
            model_name="Qwen3-8B",
            layer=15,
        )
        
        print(f"Signal: {result.signal_type}")
        print(f"Method: {result.recommended_method}")
        print(f"Optimal pairs: {result.optimal_n_pairs}")
        ```
    """
    import numpy as np
    
    if verbose:
        print(f"=" * 60)
        print(f"RepScan: {benchmark} @ {model_name} layer {layer}")
        print(f"=" * 60)
    
    # =========================================================================
    # PHASE 1: Find optimal number of pairs with saturation detection
    # =========================================================================
    if verbose:
        print(f"\n[Phase 1] Finding optimal pair count...")
    
    history = []
    previous_accuracy = 0.5
    previous_direction = None
    optimal_n_pairs = pair_counts[0]
    final_pos_act = None
    final_neg_act = None
    saturation_status = "ERROR"
    
    for n_pairs in pair_counts:
        if verbose:
            print(f"  Testing n_pairs={n_pairs}...")
        
        try:
            pos_act, neg_act = get_activations_fn(n_pairs)
            final_pos_act = pos_act
            final_neg_act = neg_act
        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            break
        
        # Bootstrap for variance
        bootstrap = compute_bootstrap_signal_estimate(
            pos_act, neg_act,
            n_bootstrap=n_bootstrap,
            sample_fraction=0.8,
        )
        
        current_accuracy = bootstrap["best_nonlinear_mean"]
        current_direction = compute_direction_from_pairs(pos_act, neg_act)
        
        if verbose:
            print(f"    Accuracy: {current_accuracy:.3f} +/- {bootstrap['best_nonlinear_std']:.3f}")
        
        # Check if signal is definitively random
        if current_accuracy < random_signal_threshold:
            if verbose:
                print(f"    Signal is random (< {random_signal_threshold})")
            history.append({
                "n_pairs": n_pairs,
                "accuracy": current_accuracy,
                "std": bootstrap["best_nonlinear_std"],
                "status": "RANDOM",
            })
            saturation_status = "NO_SIGNAL"
            optimal_n_pairs = n_pairs
            break
        
        # Check if signal exists but weak - might need more pairs
        if current_accuracy < min_signal_threshold:
            if verbose:
                print(f"    Weak signal, trying more pairs...")
            history.append({
                "n_pairs": n_pairs,
                "accuracy": current_accuracy,
                "std": bootstrap["best_nonlinear_std"],
                "status": "WEAK",
            })
            previous_accuracy = current_accuracy
            previous_direction = current_direction
            continue
        
        # Signal found - check saturation
        if previous_direction is not None:
            direction_sim = torch.nn.functional.cosine_similarity(
                current_direction.unsqueeze(0),
                previous_direction.unsqueeze(0)
            ).item()
        else:
            direction_sim = 0.0
        
        accuracy_saturated = abs(current_accuracy - previous_accuracy) < accuracy_threshold
        direction_stable = direction_sim > direction_threshold
        variance_low = bootstrap["best_nonlinear_std"] < variance_threshold
        
        is_saturated = accuracy_saturated and direction_stable and variance_low
        
        history.append({
            "n_pairs": n_pairs,
            "accuracy": current_accuracy,
            "std": bootstrap["best_nonlinear_std"],
            "direction_sim": direction_sim,
            "status": "SATURATED" if is_saturated else "IMPROVING",
        })
        
        if is_saturated:
            optimal_n_pairs = n_pairs
            saturation_status = "OPTIMAL"
            if verbose:
                print(f"    SATURATED (acc_delta={abs(current_accuracy - previous_accuracy):.3f}, dir_sim={direction_sim:.3f})")
            break
        else:
            optimal_n_pairs = n_pairs
            saturation_status = "MAX_REACHED"
            if verbose:
                print(f"    Not saturated, continuing...")
            previous_accuracy = current_accuracy
            previous_direction = current_direction
    
    # =========================================================================
    # PHASE 2: Compute all metrics on optimal pair count
    # =========================================================================
    if verbose:
        print(f"\n[Phase 2] Computing full metrics on {optimal_n_pairs} pairs...")
    
    if final_pos_act is None:
        # No activations could be extracted
        return RepScanResult(
            benchmark=benchmark,
            model_name=model_name,
            layer=layer,
            optimal_n_pairs=0,
            pair_search_history=history,
            saturation_status="ERROR",
            signal_exists=False,
            signal_type="NO_SIGNAL",
            recommended_method="NO_METHOD",
            confidence=0.0,
            reason="Failed to extract activations",
            linear_probe_accuracy=0.5,
            best_nonlinear_accuracy=0.5,
            knn_umap_accuracy=0.5,
            knn_pacmap_accuracy=0.5,
            icd=0.0,
            icd_top1_variance=0.0,
            diff_mean_alignment=0.0,
            caa_probe_alignment=0.0,
            steerability_score=0.0,
            effective_steering_dims=0,
            multi_dir_gain=0.0,
            spherical_silhouette_k2=0.0,
            cluster_direction_angle=90.0,
            cohens_d=0.0,
            bootstrap_std=1.0,
            direction_stability=0.0,
        )
    
    # Compute all probes
    linear_probe = compute_linear_probe_accuracy(final_pos_act, final_neg_act)
    knn_k10 = compute_knn_accuracy(final_pos_act, final_neg_act, k=10)
    knn_pca = compute_knn_pca_accuracy(final_pos_act, final_neg_act, k=10, n_components=50)
    knn_umap = compute_knn_umap_accuracy(final_pos_act, final_neg_act, k=10, n_components=10)
    knn_pacmap = compute_knn_pacmap_accuracy(final_pos_act, final_neg_act, k=10, n_components=10)
    mlp_probe = compute_mlp_probe_accuracy(final_pos_act, final_neg_act, hidden_size=64)
    
    best_nonlinear = max(knn_k10, knn_pca, knn_umap, knn_pacmap, mlp_probe)
    
    # Geometry metrics
    steerability = compute_steerability_metrics(final_pos_act, final_neg_act)
    icd_results = compute_icd(final_pos_act, final_neg_act)
    multi_dir = compute_multi_direction_accuracy(final_pos_act, final_neg_act)
    
    # Bootstrap for final variance
    final_bootstrap = compute_bootstrap_signal_estimate(
        final_pos_act, final_neg_act,
        n_bootstrap=n_bootstrap,
    )
    
    # Direction stability (compare bootstrap directions)
    direction_stability = 1.0 - final_bootstrap["best_nonlinear_std"] / max(0.01, final_bootstrap["best_nonlinear_mean"] - 0.5)
    direction_stability = max(0.0, min(1.0, direction_stability))
    
    if verbose:
        print(f"    Linear probe: {linear_probe:.3f}")
        print(f"    Best nonlinear: {best_nonlinear:.3f}")
        print(f"    ICD: {icd_results['icd']:.2f}")
        print(f"    Diff-mean alignment: {steerability['diff_mean_alignment']:.3f}")
    
    # =========================================================================
    # PHASE 3: Classify signal and recommend method
    # =========================================================================
    if verbose:
        print(f"\n[Phase 3] Computing recommendation...")
    
    recommendation = compute_recommendation(
        linear_probe_accuracy=linear_probe,
        best_nonlinear_accuracy=best_nonlinear,
        knn_umap_accuracy=knn_umap,
        knn_pacmap_accuracy=knn_pacmap,
        icd=icd_results["icd"],
        icd_top1_variance=icd_results["top1_variance"],
        diff_mean_alignment=steerability["diff_mean_alignment"],
        steerability_score=steerability["steerability_score"],
        effective_steering_dims=steerability["effective_steering_dims"],
        multi_dir_gain=multi_dir["gain_from_multi"],
        spherical_silhouette_k2=steerability["spherical_silhouette_k2"],
        cluster_direction_angle=steerability["cluster_direction_angle"],
        cohens_d=0.0,  # Would need additional computation
        signal_above_baseline=0.0,  # Would need nonsense baseline
    )
    
    # Compute steering direction if signal exists
    steering_direction = None
    if recommendation["signal_exists"]:
        steering_direction = compute_direction_from_pairs(final_pos_act, final_neg_act)
    
    if verbose:
        print(f"    Signal type: {recommendation['signal_type']}")
        print(f"    Method: {recommendation['recommended_method']}")
        print(f"    Confidence: {recommendation['confidence']:.2f}")
        print(f"    Reason: {recommendation['reason']}")
    
    # =========================================================================
    # PHASE 4: Build result
    # =========================================================================
    result = RepScanResult(
        benchmark=benchmark,
        model_name=model_name,
        layer=layer,
        optimal_n_pairs=optimal_n_pairs,
        pair_search_history=history,
        saturation_status=saturation_status,
        signal_exists=recommendation["signal_exists"],
        signal_type=recommendation["signal_type"],
        recommended_method=recommendation["recommended_method"],
        confidence=recommendation["confidence"],
        reason=recommendation["reason"],
        linear_probe_accuracy=linear_probe,
        best_nonlinear_accuracy=best_nonlinear,
        knn_umap_accuracy=knn_umap,
        knn_pacmap_accuracy=knn_pacmap,
        icd=icd_results["icd"],
        icd_top1_variance=icd_results["top1_variance"],
        diff_mean_alignment=steerability["diff_mean_alignment"],
        caa_probe_alignment=steerability.get("caa_probe_alignment", steerability["diff_mean_alignment"]),
        steerability_score=steerability["steerability_score"],
        effective_steering_dims=steerability["effective_steering_dims"],
        multi_dir_gain=multi_dir["gain_from_multi"],
        spherical_silhouette_k2=steerability["spherical_silhouette_k2"],
        cluster_direction_angle=steerability["cluster_direction_angle"],
        cohens_d=0.0,
        bootstrap_std=final_bootstrap["best_nonlinear_std"],
        direction_stability=direction_stability,
        steering_direction=steering_direction,
    )
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"RepScan Complete: {recommendation['recommended_method']} ({recommendation['signal_type']})")
        print(f"{'=' * 60}")
    
    return result


@dataclass
class RepScanLayerResult:
    """Result from layer search - extends RepScanResult with layer analysis."""
    # Best layer result
    best_result: RepScanResult
    
    # Layer search info
    optimal_layer: int
    optimal_layer_range: List[int]  # Top layers within 5% of best
    layer_search_history: List[Dict[str, Any]]
    
    # Model info
    num_layers: int
    layers_tested: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = self.best_result.to_dict()
        d.update({
            "optimal_layer": self.optimal_layer,
            "optimal_layer_range": self.optimal_layer_range,
            "layer_search_history": self.layer_search_history,
            "num_layers": self.num_layers,
            "layers_tested": self.layers_tested,
        })
        return d


def run_full_repscan_with_layer_search(
    get_activations_fn_for_layer,
    num_layers: int,
    benchmark: str = "unknown",
    model_name: str = "unknown",
    pair_counts: List[int] = [50, 100, 200],
    n_bootstrap: int = 5,
    verbose: bool = True,
) -> RepScanLayerResult:
    """
    Run RepScan with automatic layer search.
    
    Strategy:
    1. Test ALL layers (skip only layer 0 embedding and last layer)
    2. Select layer with best alignment (CAA-probe cosine similarity)
    3. Run full RepScan on optimal layer
    
    Key insight: Steering effectiveness correlates with alignment, not accuracy.
    Accuracy is often ~1.0 across all layers, but alignment varies significantly.
    Early layers often have better alignment (e.g., layer 5: align=0.79 vs layer 30: align=0.38).
    
    Args:
        get_activations_fn_for_layer: Function(layer, n_pairs) -> (pos_act, neg_act)
        num_layers: Total number of layers in model
        benchmark: Benchmark name
        model_name: Model name
        pair_counts: Pair counts to try
        n_bootstrap: Bootstrap samples
        verbose: Print progress
        
    Returns:
        RepScanLayerResult with optimal layer and full analysis
        
    Example:
        ```python
        def get_acts(layer, n_pairs):
            # Extract activations at given layer
            return pos_activations, neg_activations
            
        result = run_full_repscan_with_layer_search(
            get_acts,
            num_layers=32,
            benchmark="truthfulqa",
            model_name="Qwen3-8B",
        )
        
        print(f"Optimal layer: {result.optimal_layer}")
        print(f"Method: {result.best_result.recommended_method}")
        ```
    """
    if verbose:
        print(f"=" * 70)
        print(f"RepScan Layer Search: {benchmark} @ {model_name}")
        print(f"Model has {num_layers} layers")
        print(f"=" * 70)
    
    # =========================================================================
    # PHASE 1: Test ALL layers
    # =========================================================================
    # Skip layer 0 (embedding) and last layer (often output-specific)
    start_layer = 1
    end_layer = num_layers - 1
    all_layers = list(range(start_layer, end_layer + 1))
    
    if verbose:
        print(f"\n[Phase 1] Testing ALL {len(all_layers)} layers: {all_layers[0]}...{all_layers[-1]}")
    
    layer_results = {}
    
    for layer in all_layers:
        if verbose:
            print(f"  Layer {layer}...", end=" ", flush=True)
        
        # Create activation function for this layer
        def get_acts_for_layer(n_pairs, _layer=layer):
            return get_activations_fn_for_layer(_layer, n_pairs)
        
        # Run quick RepScan (fewer pairs for speed)
        quick_pair_counts = [pair_counts[0]]  # Just first count for layer scan
        
        result = run_full_repscan(
            get_acts_for_layer,
            benchmark=benchmark,
            model_name=model_name,
            layer=layer,
            pair_counts=quick_pair_counts,
            n_bootstrap=3,  # Fewer bootstraps for speed
            verbose=False,
        )
        
        layer_results[layer] = {
            "accuracy": result.best_nonlinear_accuracy,
            "linear_probe": result.linear_probe_accuracy,
            "signal_type": result.signal_type,
            "alignment": result.caa_probe_alignment,  # Use CAA-Probe alignment (key predictor!)
            "diff_mean_alignment": result.diff_mean_alignment,
            "steerability": result.steerability_score,
            "result": result,
        }
        
        if verbose:
            print(f"acc={result.best_nonlinear_accuracy:.2f}, caa_probe={result.caa_probe_alignment:.2f}, {result.signal_type}")
    
    # Find best layer BY ALIGNMENT (not accuracy!)
    # Key insight: accuracy is often ~1.0 across all layers, but alignment varies
    # Steering effectiveness correlates with alignment, not accuracy
    optimal_layer = max(layer_results.keys(), key=lambda l: layer_results[l]["alignment"])
    best_acc = layer_results[optimal_layer]["accuracy"]
    best_align = layer_results[optimal_layer]["alignment"]
    
    # Find layers within 10% of best alignment
    threshold = best_align * 0.90
    optimal_layer_range = sorted([
        l for l, data in layer_results.items()
        if data["alignment"] >= threshold
    ])
    
    if verbose:
        print(f"\n[Phase 1 Result] Optimal layer: {optimal_layer} (acc={best_acc:.3f}, align={best_align:.3f})")
        print(f"Layers within 10% alignment: {optimal_layer_range}")
    
    # =========================================================================
    # PHASE 2: Full RepScan on optimal layer
    # =========================================================================
    if verbose:
        print(f"\n[Phase 2] Running full RepScan on layer {optimal_layer}...")
    
    def get_acts_optimal(n_pairs):
        return get_activations_fn_for_layer(optimal_layer, n_pairs)
    
    final_result = run_full_repscan(
        get_acts_optimal,
        benchmark=benchmark,
        model_name=model_name,
        layer=optimal_layer,
        pair_counts=pair_counts,
        n_bootstrap=n_bootstrap,
        verbose=verbose,
    )
    
    # Build layer search history
    layer_search_history = [
        {
            "layer": layer,
            "accuracy": data["accuracy"],
            "linear_probe": data["linear_probe"],
            "signal_type": data["signal_type"],
            "alignment": data.get("alignment", 0.0),
            "steerability": data.get("steerability", 0.0),
        }
        for layer, data in sorted(layer_results.items())
    ]
    
    # =========================================================================
    # PHASE 3: Build result
    # =========================================================================
    result = RepScanLayerResult(
        best_result=final_result,
        optimal_layer=optimal_layer,
        optimal_layer_range=optimal_layer_range,
        layer_search_history=layer_search_history,
        num_layers=num_layers,
        layers_tested=sorted(layer_results.keys()),
    )
    
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"RepScan Layer Search Complete")
        print(f"  Optimal layer: {optimal_layer} (selected by ALIGNMENT)")
        print(f"  Layer range: {optimal_layer_range}")
        print(f"  Signal type: {final_result.signal_type}")
        print(f"  Method: {final_result.recommended_method}")
        print(f"{'=' * 70}")
    
    return result


# =============================================================================
# STEERING EVALUATION - Does steering actually work?
# =============================================================================

@dataclass
class SteeringEvaluationResult:
    """Result from evaluating steering effectiveness."""
    # Basic info
    steering_strength: float
    
    # Activation shift metrics
    neg_to_pos_shift: float  # How much neg activations moved toward pos
    pos_stability: float  # How much pos activations stayed in place
    separation_after: float  # Linear probe accuracy after steering
    
    # Classification metrics
    neg_classified_as_pos_before: float  # % of neg classified as pos before
    neg_classified_as_pos_after: float  # % of neg classified as pos after
    flip_rate: float  # % of neg that flipped to pos classification
    
    # Geometric metrics
    cosine_shift_neg: float  # Avg cosine similarity of neg shift to steering direction
    magnitude_ratio: float  # |steering_vector| / |avg_activation|
    
    # Quality metrics
    steering_effective: bool  # Did steering work?
    optimal_strength: float  # Best strength found
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "steering_strength": self.steering_strength,
            "neg_to_pos_shift": self.neg_to_pos_shift,
            "pos_stability": self.pos_stability,
            "separation_after": self.separation_after,
            "neg_classified_as_pos_before": self.neg_classified_as_pos_before,
            "neg_classified_as_pos_after": self.neg_classified_as_pos_after,
            "flip_rate": self.flip_rate,
            "cosine_shift_neg": self.cosine_shift_neg,
            "magnitude_ratio": self.magnitude_ratio,
            "steering_effective": self.steering_effective,
            "optimal_strength": self.optimal_strength,
        }


def evaluate_steering_effectiveness(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    steering_direction: torch.Tensor,
    steering_strengths: List[float] = [0.5, 1.0, 2.0, 3.0, 5.0],
    verbose: bool = False,
) -> SteeringEvaluationResult:
    """
    Evaluate if steering actually moves activations in the right direction.
    
    Key questions:
    1. Do negative activations move toward positive region after steering?
    2. Do positive activations stay stable?
    3. Does a classifier trained on original data classify steered-neg as pos?
    
    Args:
        pos_activations: Original positive activations [n_pos, hidden_dim]
        neg_activations: Original negative activations [n_neg, hidden_dim]
        steering_direction: Normalized steering vector [hidden_dim]
        steering_strengths: List of steering strengths to test
        verbose: Print progress
        
    Returns:
        SteeringEvaluationResult with effectiveness metrics
    """
    # Ensure direction is normalized
    steering_direction = steering_direction / (steering_direction.norm() + 1e-8)
    
    # Compute centers
    pos_center = pos_activations.mean(dim=0)
    neg_center = neg_activations.mean(dim=0)
    
    # Train classifier on original data
    from sklearn.linear_model import LogisticRegression
    X_train = torch.cat([pos_activations, neg_activations], dim=0).numpy()
    y_train = np.array([1] * len(pos_activations) + [0] * len(neg_activations))
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # Baseline: how many neg classified as pos before steering?
    neg_pred_before = clf.predict(neg_activations.numpy())
    neg_as_pos_before = (neg_pred_before == 1).mean()
    
    # Test different steering strengths
    best_strength = steering_strengths[0]  # Default to first strength
    best_flip_rate = -1.0
    results_by_strength = []
    
    for strength in steering_strengths:
        # Apply steering to negative activations
        steering_vector = steering_direction * strength
        neg_steered = neg_activations + steering_vector
        
        # How much did neg move toward pos?
        neg_center_steered = neg_steered.mean(dim=0)
        
        # Distance before and after
        dist_before = (neg_center - pos_center).norm().item()
        dist_after = (neg_center_steered - pos_center).norm().item()
        neg_to_pos_shift = (dist_before - dist_after) / (dist_before + 1e-8)
        
        # How much did the shift align with steering direction?
        actual_shift = neg_center_steered - neg_center
        cosine_shift = torch.nn.functional.cosine_similarity(
            actual_shift.unsqueeze(0),
            steering_direction.unsqueeze(0)
        ).item()
        
        # Classify steered neg
        neg_pred_after = clf.predict(neg_steered.numpy())
        neg_as_pos_after = (neg_pred_after == 1).mean()
        flip_rate = neg_as_pos_after - neg_as_pos_before
        
        # Check separation after (retrain classifier)
        X_after = torch.cat([pos_activations, neg_steered], dim=0).numpy()
        clf_after = LogisticRegression(max_iter=1000, random_state=42)
        clf_after.fit(X_after, y_train)
        separation_after = clf_after.score(X_after, y_train)
        
        results_by_strength.append({
            "strength": strength,
            "neg_to_pos_shift": neg_to_pos_shift,
            "flip_rate": flip_rate,
            "separation_after": separation_after,
            "cosine_shift": cosine_shift,
        })
        
        if flip_rate > best_flip_rate:
            best_flip_rate = flip_rate
            best_strength = strength
        
        if verbose:
            print(f"  Strength {strength}: flip_rate={flip_rate:.3f}, shift={neg_to_pos_shift:.3f}")
    
    # Use best strength for final metrics
    best_result = next(r for r in results_by_strength if r["strength"] == best_strength)
    
    # Compute magnitude ratio
    avg_activation_norm = torch.cat([pos_activations, neg_activations]).norm(dim=1).mean().item()
    magnitude_ratio = best_strength / (avg_activation_norm + 1e-8)
    
    # Determine if steering is effective
    # Effective if: flip_rate > 0.2 (20% of neg now classified as pos)
    steering_effective = best_flip_rate > 0.2
    
    return SteeringEvaluationResult(
        steering_strength=best_strength,
        neg_to_pos_shift=best_result["neg_to_pos_shift"],
        pos_stability=1.0,  # Pos not steered, so stable
        separation_after=best_result["separation_after"],
        neg_classified_as_pos_before=neg_as_pos_before,
        neg_classified_as_pos_after=neg_as_pos_before + best_flip_rate,
        flip_rate=best_flip_rate,
        cosine_shift_neg=best_result["cosine_shift"],
        magnitude_ratio=magnitude_ratio,
        steering_effective=steering_effective,
        optimal_strength=best_strength,
    )


def evaluate_activation_regions(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    neg_steered: torch.Tensor,
    n_neighbors: int = 10,
) -> Dict[str, Any]:
    """
    Check if steered negative activations land in positive region.
    
    Uses k-NN to determine "region membership":
    - For each steered-neg activation, find k nearest neighbors
    - Count how many neighbors are pos vs neg
    - If majority are pos, the activation is in "positive region"
    
    Args:
        pos_activations: Original positive [n_pos, dim]
        neg_activations: Original negative [n_neg, dim]
        neg_steered: Steered negative [n_neg, dim]
        n_neighbors: Number of neighbors to check
        
    Returns:
        Dict with region membership metrics
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Build index on original pos + neg
    X_ref = torch.cat([pos_activations, neg_activations], dim=0).numpy()
    labels = np.array([1] * len(pos_activations) + [0] * len(neg_activations))
    
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    nn.fit(X_ref)
    
    # For each steered neg, find neighbors
    distances, indices = nn.kneighbors(neg_steered.numpy())
    
    # Count how many neighbors are positive
    neighbor_labels = labels[indices]  # [n_neg, k]
    pos_neighbor_ratio = neighbor_labels.mean(axis=1)  # [n_neg]
    
    # In positive region if majority neighbors are pos
    in_pos_region = (pos_neighbor_ratio > 0.5).mean()
    
    # Also check distance to pos vs neg centers
    pos_center = pos_activations.mean(dim=0).numpy()
    neg_center = neg_activations.mean(dim=0).numpy()
    
    dist_to_pos = np.linalg.norm(neg_steered.numpy() - pos_center, axis=1)
    dist_to_neg = np.linalg.norm(neg_steered.numpy() - neg_center, axis=1)
    
    closer_to_pos = (dist_to_pos < dist_to_neg).mean()
    
    return {
        "in_positive_region": float(in_pos_region),
        "avg_pos_neighbor_ratio": float(pos_neighbor_ratio.mean()),
        "closer_to_pos_center": float(closer_to_pos),
        "avg_dist_to_pos": float(dist_to_pos.mean()),
        "avg_dist_to_neg": float(dist_to_neg.mean()),
    }


@dataclass 
class ComponentAnalysisResult:
    """Result from analyzing different transformer components."""
    # Which component has strongest signal
    best_component: str  # "residual", "mlp", "attn", "head_X"
    best_component_accuracy: float
    
    # Per-component results
    residual_accuracy: float
    mlp_accuracy: float
    attn_accuracy: float
    head_accuracies: Dict[int, float]  # head_idx -> accuracy
    
    # Best attention heads
    top_heads: List[Tuple[int, float]]  # [(head_idx, accuracy), ...]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_component": self.best_component,
            "best_component_accuracy": self.best_component_accuracy,
            "residual_accuracy": self.residual_accuracy,
            "mlp_accuracy": self.mlp_accuracy,
            "attn_accuracy": self.attn_accuracy,
            "head_accuracies": self.head_accuracies,
            "top_heads": self.top_heads,
        }


def analyze_transformer_components(
    get_component_activations_fn,
    n_pairs: int = 50,
    n_heads: int = 32,
    verbose: bool = False,
) -> ComponentAnalysisResult:
    """
    Analyze which transformer component has the strongest signal.
    
    Components:
    - Residual stream (default)
    - MLP output
    - Attention output (combined)
    - Individual attention heads
    
    Args:
        get_component_activations_fn: Function(component, head_idx) -> (pos, neg)
            component: "residual", "mlp", "attn", "head"
            head_idx: Only used when component="head"
        n_pairs: Number of pairs to use
        n_heads: Number of attention heads to test
        verbose: Print progress
        
    Returns:
        ComponentAnalysisResult with per-component analysis
    """
    results = {}
    
    # Test main components
    for component in ["residual", "mlp", "attn"]:
        if verbose:
            print(f"Testing {component}...")
        
        try:
            pos, neg = get_component_activations_fn(component, None)
            acc = compute_linear_probe_accuracy(pos[:n_pairs], neg[:n_pairs])
            results[component] = acc
            if verbose:
                print(f"  {component}: {acc:.3f}")
        except Exception as e:
            results[component] = 0.5
            if verbose:
                print(f"  {component}: error - {e}")
    
    # Test individual heads
    head_accuracies = {}
    for head_idx in range(n_heads):
        try:
            pos, neg = get_component_activations_fn("head", head_idx)
            acc = compute_linear_probe_accuracy(pos[:n_pairs], neg[:n_pairs])
            head_accuracies[head_idx] = acc
        except:
            head_accuracies[head_idx] = 0.5
    
    if verbose and head_accuracies:
        top_heads = sorted(head_accuracies.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top heads: {top_heads}")
    
    # Find best component
    all_results = {**results}
    for head_idx, acc in head_accuracies.items():
        all_results[f"head_{head_idx}"] = acc
    
    best_component = max(all_results.keys(), key=lambda k: all_results[k])
    best_accuracy = all_results[best_component]
    
    # Top heads
    top_heads = sorted(head_accuracies.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return ComponentAnalysisResult(
        best_component=best_component,
        best_component_accuracy=best_accuracy,
        residual_accuracy=results.get("residual", 0.5),
        mlp_accuracy=results.get("mlp", 0.5),
        attn_accuracy=results.get("attn", 0.5),
        head_accuracies=head_accuracies,
        top_heads=top_heads,
    )


def run_full_repscan_with_steering_eval(
    get_activations_fn,
    benchmark: str = "unknown",
    model_name: str = "unknown",
    layer: int = -1,
    pair_counts: List[int] = [50, 100, 200],
    steering_strengths: List[float] = [0.5, 1.0, 2.0, 3.0, 5.0],
    verbose: bool = True,
) -> Tuple[RepScanResult, SteeringEvaluationResult, Dict[str, Any]]:
    """
    Run RepScan + evaluate if steering actually works.
    
    This is the complete pipeline:
    1. Run RepScan to detect signal and get steering direction
    2. Evaluate if steering moves neg->pos effectively
    3. Verify steered activations land in correct region
    
    Args:
        get_activations_fn: Function(n_pairs) -> (pos, neg)
        benchmark: Benchmark name
        model_name: Model name
        layer: Layer number
        pair_counts: Pair counts to try
        steering_strengths: Steering strengths to evaluate
        verbose: Print progress
        
    Returns:
        Tuple of (RepScanResult, SteeringEvaluationResult, region_metrics)
    """
    # Phase 1: Run RepScan
    if verbose:
        print("=" * 60)
        print("Phase 1: Running RepScan...")
        print("=" * 60)
    
    repscan_result = run_full_repscan(
        get_activations_fn,
        benchmark=benchmark,
        model_name=model_name,
        layer=layer,
        pair_counts=pair_counts,
        verbose=verbose,
    )
    
    if not repscan_result.signal_exists or repscan_result.steering_direction is None:
        if verbose:
            print("\nNo signal detected - skipping steering evaluation")
        
        # Return empty results
        empty_steering = SteeringEvaluationResult(
            steering_strength=0.0,
            neg_to_pos_shift=0.0,
            pos_stability=1.0,
            separation_after=0.5,
            neg_classified_as_pos_before=0.0,
            neg_classified_as_pos_after=0.0,
            flip_rate=0.0,
            cosine_shift_neg=0.0,
            magnitude_ratio=0.0,
            steering_effective=False,
            optimal_strength=0.0,
        )
        return repscan_result, empty_steering, {}
    
    # Phase 2: Evaluate steering effectiveness
    if verbose:
        print("\n" + "=" * 60)
        print("Phase 2: Evaluating Steering Effectiveness...")
        print("=" * 60)
    
    # Get activations for evaluation
    pos_act, neg_act = get_activations_fn(repscan_result.optimal_n_pairs)
    
    steering_eval = evaluate_steering_effectiveness(
        pos_act, neg_act,
        repscan_result.steering_direction,
        steering_strengths=steering_strengths,
        verbose=verbose,
    )
    
    if verbose:
        print(f"\nSteering Evaluation:")
        print(f"  Optimal strength: {steering_eval.optimal_strength}")
        print(f"  Flip rate: {steering_eval.flip_rate:.3f}")
        print(f"  Steering effective: {steering_eval.steering_effective}")
    
    # Phase 3: Verify activation regions
    if verbose:
        print("\n" + "=" * 60)
        print("Phase 3: Verifying Activation Regions...")
        print("=" * 60)
    
    # Apply optimal steering
    neg_steered = neg_act + repscan_result.steering_direction * steering_eval.optimal_strength
    
    region_metrics = evaluate_activation_regions(
        pos_act, neg_act, neg_steered,
        n_neighbors=10,
    )
    
    if verbose:
        print(f"\nRegion Analysis:")
        print(f"  Steered neg in positive region: {region_metrics['in_positive_region']:.1%}")
        print(f"  Closer to pos center: {region_metrics['closer_to_pos_center']:.1%}")
        print(f"  Avg pos neighbor ratio: {region_metrics['avg_pos_neighbor_ratio']:.3f}")
    
    # Final summary
    if verbose:
        print("\n" + "=" * 60)
        print("RepScan + Steering Evaluation Complete")
        print("=" * 60)
        print(f"Signal: {repscan_result.signal_type}")
        print(f"Method: {repscan_result.recommended_method}")
        print(f"Steering works: {steering_eval.steering_effective}")
        print(f"Activations in correct region: {region_metrics['in_positive_region']:.1%}")
    
    return repscan_result, steering_eval, region_metrics


# =============================================================================
# TRANSFORMER COMPONENT EXTRACTION
# =============================================================================

class TransformerComponent(Enum):
    """Which part of transformer to extract activations from."""
    RESIDUAL = "residual"      # Block output (residual stream after MLP)
    RESIDUAL_PRE = "residual_pre"  # Before attention
    RESIDUAL_MID = "residual_mid"  # After attention, before MLP
    MLP_OUTPUT = "mlp_output"  # MLP output only
    ATTN_OUTPUT = "attn_output"  # Attention output only
    ATTN_HEAD = "attn_head"    # Individual attention head


def get_component_hook_points(model_type: str, layer: int, component: TransformerComponent) -> List[str]:
    """
    Get hook point names for different model architectures.
    
    Different models have different naming conventions:
    - Llama: model.layers.{layer}.mlp, model.layers.{layer}.self_attn
    - GPT2: transformer.h.{layer}.mlp, transformer.h.{layer}.attn
    - Qwen: model.layers.{layer}.mlp, model.layers.{layer}.self_attn
    
    Args:
        model_type: Type of model ("llama", "gpt2", "qwen", etc.)
        layer: Layer index
        component: Which component to hook
        
    Returns:
        List of hook point names
    """
    # Common patterns for Llama-style models (includes Qwen, Mistral, etc.)
    llama_patterns = {
        TransformerComponent.RESIDUAL: [f"model.layers.{layer}"],
        TransformerComponent.RESIDUAL_PRE: [f"model.layers.{layer}.input_layernorm"],
        TransformerComponent.RESIDUAL_MID: [f"model.layers.{layer}.post_attention_layernorm"],
        TransformerComponent.MLP_OUTPUT: [f"model.layers.{layer}.mlp"],
        TransformerComponent.ATTN_OUTPUT: [f"model.layers.{layer}.self_attn"],
    }
    
    # GPT2-style patterns
    gpt2_patterns = {
        TransformerComponent.RESIDUAL: [f"transformer.h.{layer}"],
        TransformerComponent.MLP_OUTPUT: [f"transformer.h.{layer}.mlp"],
        TransformerComponent.ATTN_OUTPUT: [f"transformer.h.{layer}.attn"],
    }
    
    model_type_lower = model_type.lower()
    
    if any(x in model_type_lower for x in ["llama", "qwen", "mistral", "gemma"]):
        return llama_patterns.get(component, [])
    elif "gpt2" in model_type_lower:
        return gpt2_patterns.get(component, [])
    else:
        # Default to llama-style
        return llama_patterns.get(component, [])


class ComponentActivationExtractor:
    """
    Extract activations from specific transformer components using hooks.
    
    Example:
        ```python
        extractor = ComponentActivationExtractor(model, tokenizer)
        
        # Get MLP output at layer 15
        mlp_acts = extractor.extract(
            texts=["Hello world", "Goodbye world"],
            layer=15,
            component=TransformerComponent.MLP_OUTPUT,
        )
        ```
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._hook_outputs = {}
        self._hooks = []
    
    def _get_module_by_name(self, name: str):
        """Get a module by its full name."""
        parts = name.split(".")
        module = self.model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    def _register_hook(self, module_name: str):
        """Register a forward hook on a module."""
        try:
            module = self._get_module_by_name(module_name)
            
            def hook_fn(module, input, output):
                # Handle different output formats
                if isinstance(output, tuple):
                    self._hook_outputs[module_name] = output[0].detach()
                else:
                    self._hook_outputs[module_name] = output.detach()
            
            handle = module.register_forward_hook(hook_fn)
            self._hooks.append(handle)
            return True
        except Exception as e:
            print(f"Warning: Could not register hook for {module_name}: {e}")
            return False
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._hook_outputs = {}
    
    def extract(
        self,
        texts: List[str],
        layer: int,
        component: TransformerComponent,
        token_position: int = -1,  # -1 = last token
    ) -> torch.Tensor:
        """
        Extract activations from a specific component.
        
        Args:
            texts: List of input texts
            layer: Layer index
            component: Which component (RESIDUAL, MLP_OUTPUT, ATTN_OUTPUT, etc.)
            token_position: Which token position to extract (-1 = last)
            
        Returns:
            Tensor of shape [len(texts), hidden_dim]
        """
        # Detect model type
        model_type = type(self.model).__name__
        
        # Get hook points
        hook_points = get_component_hook_points(model_type, layer, component)
        
        if not hook_points:
            raise ValueError(f"No hook points found for {component} in {model_type}")
        
        # Clear previous hooks
        self._clear_hooks()
        
        # Register hooks
        for hook_point in hook_points:
            self._register_hook(hook_point)
        
        activations = []
        
        try:
            with torch.no_grad():
                for text in texts:
                    # Tokenize
                    inputs = self.tokenizer(
                        text, 
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    ).to(self.device)
                    
                    # Forward pass (hooks will capture activations)
                    _ = self.model(**inputs)
                    
                    # Get hooked output
                    for hook_point in hook_points:
                        if hook_point in self._hook_outputs:
                            output = self._hook_outputs[hook_point]
                            # output shape: [batch, seq_len, hidden_dim]
                            act = output[0, token_position, :]  # [hidden_dim]
                            activations.append(act.cpu())
                            break
                    
                    # Clear for next iteration
                    self._hook_outputs = {}
        
        finally:
            self._clear_hooks()
        
        if not activations:
            raise RuntimeError(f"Failed to extract activations for {component}")
        
        return torch.stack(activations)
    
    def extract_all_components(
        self,
        texts: List[str],
        layer: int,
        token_position: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations from all components at a layer.
        
        Returns:
            Dict mapping component name to activations tensor
        """
        results = {}
        
        for component in [
            TransformerComponent.RESIDUAL,
            TransformerComponent.MLP_OUTPUT,
            TransformerComponent.ATTN_OUTPUT,
        ]:
            try:
                acts = self.extract(texts, layer, component, token_position)
                results[component.value] = acts
            except Exception as e:
                print(f"Warning: Failed to extract {component.value}: {e}")
        
        return results


def compare_components_for_benchmark(
    model,
    tokenizer,
    pos_texts: List[str],
    neg_texts: List[str],
    layer: int,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compare signal strength across different transformer components.
    
    This helps answer: "Should we steer residual, MLP, or attention?"
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        pos_texts: Positive example texts
        neg_texts: Negative example texts
        layer: Layer to analyze
        device: Device to use
        verbose: Print progress
        
    Returns:
        Dict mapping component name to linear probe accuracy
    """
    extractor = ComponentActivationExtractor(model, tokenizer, device)
    
    results = {}
    
    for component in [
        TransformerComponent.RESIDUAL,
        TransformerComponent.MLP_OUTPUT,
        TransformerComponent.ATTN_OUTPUT,
    ]:
        if verbose:
            print(f"Testing {component.value}...")
        
        try:
            pos_acts = extractor.extract(pos_texts, layer, component)
            neg_acts = extractor.extract(neg_texts, layer, component)
            
            acc = compute_linear_probe_accuracy(pos_acts, neg_acts)
            results[component.value] = acc
            
            if verbose:
                print(f"  {component.value}: {acc:.3f}")
        
        except Exception as e:
            results[component.value] = 0.5
            if verbose:
                print(f"  {component.value}: error - {e}")
    
    # Find best
    if results:
        best = max(results.items(), key=lambda x: x[1])
        if verbose:
            print(f"\nBest component: {best[0]} ({best[1]:.3f})")
    
    return results


# =============================================================================
# MULTI-CONCEPT DETECTION
# =============================================================================

@dataclass
class MultiConceptAnalysis:
    """Result from analyzing if multiple concepts exist in a contrastive pair set."""
    # Overall assessment
    num_concepts_detected: int
    is_multi_concept: bool
    confidence: float
    
    # Evidence from different methods
    icd: float  # Intrinsic Concept Dimensionality
    icd_suggests_multi: bool
    
    cluster_count: int  # Number of clusters in diff vectors
    cluster_silhouette: float  # Cluster quality
    clusters_suggest_multi: bool
    
    pca_variance_ratio: List[float]  # Top-k explained variance ratios
    pca_effective_rank: float  # How many PCs needed for 90% variance
    pca_suggests_multi: bool
    
    multi_dir_accuracy: Dict[int, float]  # k -> accuracy with k directions
    multi_dir_gain: float  # Gain from using multiple directions
    directions_suggest_multi: bool
    
    # Per-concept info (if clusters found)
    concept_directions: Optional[List[torch.Tensor]] = None
    concept_sizes: Optional[List[int]] = None
    concept_accuracies: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_concepts_detected": self.num_concepts_detected,
            "is_multi_concept": self.is_multi_concept,
            "confidence": self.confidence,
            "icd": self.icd,
            "icd_suggests_multi": self.icd_suggests_multi,
            "cluster_count": self.cluster_count,
            "cluster_silhouette": self.cluster_silhouette,
            "clusters_suggest_multi": self.clusters_suggest_multi,
            "pca_variance_ratio": self.pca_variance_ratio,
            "pca_effective_rank": self.pca_effective_rank,
            "pca_suggests_multi": self.pca_suggests_multi,
            "multi_dir_accuracy": self.multi_dir_accuracy,
            "multi_dir_gain": self.multi_dir_gain,
            "directions_suggest_multi": self.directions_suggest_multi,
            "concept_sizes": self.concept_sizes,
            "concept_accuracies": self.concept_accuracies,
        }


def detect_multiple_concepts(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    max_concepts: int = 5,
    icd_threshold: float = 3.0,
    cluster_silhouette_threshold: float = 0.2,
    multi_dir_gain_threshold: float = 0.05,
    verbose: bool = False,
) -> MultiConceptAnalysis:
    """
    Detect if a contrastive pair set contains multiple independent concepts.
    
    A single concept means all pos-neg differences point in roughly the same direction.
    Multiple concepts means differences cluster into distinct directions.
    
    Methods used:
    1. ICD (Intrinsic Concept Dimensionality) - effective rank of difference vectors
    2. Clustering on difference vectors - do they form distinct groups?
    3. PCA variance distribution - is variance spread across many components?
    4. Multi-direction accuracy - does using k directions improve over 1?
    
    Args:
        pos_activations: Positive activations [n_pairs, hidden_dim]
        neg_activations: Negative activations [n_pairs, hidden_dim]
        max_concepts: Maximum number of concepts to look for
        icd_threshold: ICD above this suggests multiple concepts
        cluster_silhouette_threshold: Silhouette above this suggests good clusters
        multi_dir_gain_threshold: Accuracy gain above this suggests multiple directions help
        verbose: Print progress
        
    Returns:
        MultiConceptAnalysis with detailed breakdown
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    
    n_pairs = pos_activations.shape[0]
    
    # Compute difference vectors
    diff_vectors = pos_activations - neg_activations  # [n_pairs, hidden_dim]
    diff_np = diff_vectors.numpy()
    
    if verbose:
        print(f"Analyzing {n_pairs} pairs for multiple concepts...")
    
    # =========================================================================
    # Method 1: ICD (Intrinsic Concept Dimensionality)
    # =========================================================================
    icd_results = compute_icd(pos_activations, neg_activations)
    icd = icd_results["icd"]
    icd_suggests_multi = icd > icd_threshold
    
    if verbose:
        print(f"  ICD: {icd:.2f} (threshold: {icd_threshold}) -> {'MULTI' if icd_suggests_multi else 'SINGLE'}")
    
    # =========================================================================
    # Method 2: Clustering difference vectors
    # =========================================================================
    best_k = 1
    best_silhouette = -1
    cluster_results = {}
    
    for k in range(2, min(max_concepts + 1, n_pairs // 5 + 1)):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(diff_np)
            
            if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette
                sil = silhouette_score(diff_np, labels)
                cluster_results[k] = sil
                
                if sil > best_silhouette:
                    best_silhouette = sil
                    best_k = k
        except:
            pass
    
    clusters_suggest_multi = best_k > 1 and best_silhouette > cluster_silhouette_threshold
    
    if verbose:
        print(f"  Clusters: k={best_k}, silhouette={best_silhouette:.3f} -> {'MULTI' if clusters_suggest_multi else 'SINGLE'}")
    
    # =========================================================================
    # Method 3: PCA variance distribution
    # =========================================================================
    n_components = min(10, n_pairs - 1, diff_np.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(diff_np)
    
    variance_ratios = pca.explained_variance_ratio_.tolist()
    
    # Effective rank: how many components for 90% variance?
    cumsum = np.cumsum(variance_ratios)
    effective_rank = np.searchsorted(cumsum, 0.9) + 1
    
    # If top-1 explains < 50%, likely multiple concepts
    pca_suggests_multi = variance_ratios[0] < 0.5 or effective_rank > 2
    
    if verbose:
        print(f"  PCA: top-1={variance_ratios[0]:.2%}, effective_rank={effective_rank} -> {'MULTI' if pca_suggests_multi else 'SINGLE'}")
    
    # =========================================================================
    # Method 4: Multi-direction accuracy
    # =========================================================================
    multi_dir_results = compute_multi_direction_accuracy(pos_activations, neg_activations)
    acc_by_k = multi_dir_results.get("accuracy_by_k", {})
    multi_dir_accuracy = {
        1: acc_by_k.get(1, 0.5),
        3: acc_by_k.get(3, 0.5),
        5: acc_by_k.get(5, 0.5),
    }
    multi_dir_gain = multi_dir_results.get("gain_from_multi", 0.0)
    
    directions_suggest_multi = multi_dir_gain > multi_dir_gain_threshold
    
    if verbose:
        print(f"  Multi-dir: k1={multi_dir_accuracy[1]:.3f}, k3={multi_dir_accuracy[3]:.3f}, gain={multi_dir_gain:.3f} -> {'MULTI' if directions_suggest_multi else 'SINGLE'}")
    
    # =========================================================================
    # Combine evidence
    # =========================================================================
    evidence_count = sum([
        icd_suggests_multi,
        clusters_suggest_multi,
        pca_suggests_multi,
        directions_suggest_multi,
    ])
    
    is_multi_concept = evidence_count >= 2  # At least 2 methods agree
    confidence = evidence_count / 4.0
    
    # Estimate number of concepts
    if is_multi_concept:
        # Use cluster count or effective rank, whichever is more conservative
        num_concepts = min(best_k, effective_rank)
        num_concepts = max(2, min(num_concepts, max_concepts))
    else:
        num_concepts = 1
    
    # =========================================================================
    # Extract per-concept directions (if multiple concepts)
    # =========================================================================
    concept_directions = None
    concept_sizes = None
    concept_accuracies = None
    
    if is_multi_concept and best_k > 1:
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(diff_np)
        
        concept_directions = []
        concept_sizes = []
        concept_accuracies = []
        
        for c in range(best_k):
            mask = labels == c
            cluster_diffs = diff_vectors[mask]
            
            # Direction = mean of cluster
            direction = cluster_diffs.mean(dim=0)
            direction = direction / (direction.norm() + 1e-8)
            concept_directions.append(direction)
            concept_sizes.append(int(mask.sum()))
            
            # Accuracy using this direction only
            # Project all pairs onto this direction
            pos_proj = (pos_activations @ direction).unsqueeze(1)
            neg_proj = (neg_activations @ direction).unsqueeze(1)
            
            X = torch.cat([pos_proj, neg_proj], dim=0).numpy()
            y = np.array([1] * n_pairs + [0] * n_pairs)
            
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X, y)
            acc = clf.score(X, y)
            concept_accuracies.append(acc)
        
        if verbose:
            print(f"\nPer-concept breakdown:")
            for i, (size, acc) in enumerate(zip(concept_sizes, concept_accuracies)):
                print(f"  Concept {i+1}: {size} pairs, accuracy={acc:.3f}")
    
    if verbose:
        print(f"\nConclusion: {num_concepts} concept(s) detected (confidence: {confidence:.0%})")
    
    return MultiConceptAnalysis(
        num_concepts_detected=num_concepts,
        is_multi_concept=is_multi_concept,
        confidence=confidence,
        icd=icd,
        icd_suggests_multi=icd_suggests_multi,
        cluster_count=best_k,
        cluster_silhouette=best_silhouette,
        clusters_suggest_multi=clusters_suggest_multi,
        pca_variance_ratio=variance_ratios[:5],
        pca_effective_rank=effective_rank,
        pca_suggests_multi=pca_suggests_multi,
        multi_dir_accuracy=multi_dir_accuracy,
        multi_dir_gain=multi_dir_gain,
        directions_suggest_multi=directions_suggest_multi,
        concept_directions=concept_directions,
        concept_sizes=concept_sizes,
        concept_accuracies=concept_accuracies,
    )


def split_by_concepts(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_concepts: int = 2,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Split a contrastive pair set into separate concepts based on clustering.
    
    Args:
        pos_activations: Positive activations [n_pairs, hidden_dim]
        neg_activations: Negative activations [n_pairs, hidden_dim]
        n_concepts: Number of concepts to split into
        
    Returns:
        List of (pos_subset, neg_subset, direction) tuples for each concept
    """
    from sklearn.cluster import KMeans
    
    diff_vectors = pos_activations - neg_activations
    diff_np = diff_vectors.numpy()
    
    kmeans = KMeans(n_clusters=n_concepts, random_state=42, n_init=10)
    labels = kmeans.fit_predict(diff_np)
    
    results = []
    for c in range(n_concepts):
        mask = torch.tensor(labels == c)
        pos_subset = pos_activations[mask]
        neg_subset = neg_activations[mask]
        
        # Direction for this concept
        direction = diff_vectors[mask].mean(dim=0)
        direction = direction / (direction.norm() + 1e-8)
        
        results.append((pos_subset, neg_subset, direction))
    
    return results


def analyze_concept_independence(
    concept_directions: List[torch.Tensor],
) -> Dict[str, Any]:
    """
    Analyze how independent the detected concepts are.
    
    Independent concepts should have low cosine similarity.
    
    Args:
        concept_directions: List of direction vectors for each concept
        
    Returns:
        Dict with independence metrics
    """
    n_concepts = len(concept_directions)
    
    if n_concepts < 2:
        return {
            "n_concepts": n_concepts,
            "avg_cosine_similarity": 0.0,
            "max_cosine_similarity": 0.0,
            "min_cosine_similarity": 0.0,
            "are_independent": True,
            "similarity_matrix": [],
        }
    
    # Compute pairwise cosine similarities
    similarities = []
    sim_matrix = np.zeros((n_concepts, n_concepts))
    
    for i in range(n_concepts):
        for j in range(i + 1, n_concepts):
            sim = torch.nn.functional.cosine_similarity(
                concept_directions[i].unsqueeze(0),
                concept_directions[j].unsqueeze(0)
            ).item()
            similarities.append(abs(sim))  # Use absolute value
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    
    avg_sim = np.mean(similarities)
    max_sim = np.max(similarities)
    min_sim = np.min(similarities)
    
    # Independent if avg similarity < 0.3
    are_independent = avg_sim < 0.3
    
    return {
        "n_concepts": n_concepts,
        "avg_cosine_similarity": float(avg_sim),
        "max_cosine_similarity": float(max_sim),
        "min_cosine_similarity": float(min_sim),
        "are_independent": are_independent,
        "similarity_matrix": sim_matrix.tolist(),
    }


# =============================================================================
# CONCEPT VALIDITY - Is this actually a concept?
# =============================================================================

@dataclass
class ConceptValidityResult:
    """Result from analyzing if a set of pairs represents a valid concept."""
    # Overall
    is_valid_concept: bool
    validity_score: float  # 0-1, higher = more valid
    concept_level: str  # "instance", "category", "domain", "noise"
    
    # Internal coherence - do all pairs point same direction?
    coherence_score: float  # 0-1, cosine similarity of individual diffs to mean
    coherence_std: float
    is_coherent: bool
    
    # Stability - does direction hold with subsamples?
    stability_score: float  # 0-1, avg cosine sim between subsample directions
    stability_std: float
    is_stable: bool
    
    # Signal quality
    signal_strength: float  # Linear probe accuracy
    signal_to_noise: float  # Cohen's d or similar
    has_signal: bool
    
    # Granularity indicators
    icd: float  # Lower = more abstract concept
    specificity: float  # How specific vs general is this concept
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid_concept": self.is_valid_concept,
            "validity_score": self.validity_score,
            "concept_level": self.concept_level,
            "coherence_score": self.coherence_score,
            "coherence_std": self.coherence_std,
            "is_coherent": self.is_coherent,
            "stability_score": self.stability_score,
            "stability_std": self.stability_std,
            "is_stable": self.is_stable,
            "signal_strength": self.signal_strength,
            "signal_to_noise": self.signal_to_noise,
            "has_signal": self.has_signal,
            "icd": self.icd,
            "specificity": self.specificity,
        }


def compute_concept_coherence(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Tuple[float, float]:
    """
    Measure how coherent a concept is - do all pairs point the same direction?
    
    A valid concept should have all pos-neg differences aligned.
    Noise or mixed concepts will have differences pointing in random directions.
    
    Returns:
        (coherence_score, coherence_std)
        coherence_score: average cosine similarity of each diff to mean diff (0-1)
        coherence_std: standard deviation of cosine similarities
    """
    diff_vectors = pos_activations - neg_activations  # [n, hidden_dim]
    
    # Mean direction
    mean_diff = diff_vectors.mean(dim=0)
    mean_diff_norm = mean_diff / (mean_diff.norm() + 1e-8)
    
    # Normalize individual diffs
    diff_norms = diff_vectors / (diff_vectors.norm(dim=1, keepdim=True) + 1e-8)
    
    # Cosine similarity of each diff to mean
    cosines = (diff_norms @ mean_diff_norm).abs()  # Use abs because direction can flip
    
    coherence = float(cosines.mean())
    coherence_std = float(cosines.std())
    
    return coherence, coherence_std


def compute_concept_stability(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_subsamples: int = 10,
    subsample_fraction: float = 0.5,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """
    Measure stability of concept direction across subsamples.
    
    A valid concept should have stable direction even with different subsets.
    Noise will have direction that changes drastically.
    
    Returns:
        (stability_score, stability_std)
        stability_score: average pairwise cosine similarity between subsample directions
        stability_std: standard deviation
    """
    n_pairs = pos_activations.shape[0]
    n_sample = max(5, int(n_pairs * subsample_fraction))
    
    rng = np.random.RandomState(random_seed)
    
    directions = []
    for _ in range(n_subsamples):
        indices = rng.choice(n_pairs, size=n_sample, replace=False)
        
        pos_sub = pos_activations[indices]
        neg_sub = neg_activations[indices]
        
        diff = pos_sub.mean(dim=0) - neg_sub.mean(dim=0)
        diff = diff / (diff.norm() + 1e-8)
        directions.append(diff)
    
    # Pairwise cosine similarities
    similarities = []
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            sim = torch.nn.functional.cosine_similarity(
                directions[i].unsqueeze(0),
                directions[j].unsqueeze(0)
            ).abs().item()
            similarities.append(sim)
    
    stability = float(np.mean(similarities))
    stability_std = float(np.std(similarities))
    
    return stability, stability_std


def compute_signal_to_noise(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> float:
    """
    Compute signal-to-noise ratio (Cohen's d style).
    
    High SNR = clear separation, valid concept
    Low SNR = noisy, possibly not a concept
    """
    # Project onto mean difference direction
    diff = pos_activations.mean(dim=0) - neg_activations.mean(dim=0)
    diff_norm = diff / (diff.norm() + 1e-8)
    
    pos_proj = (pos_activations @ diff_norm)
    neg_proj = (neg_activations @ diff_norm)
    
    # Cohen's d = (mean1 - mean2) / pooled_std
    mean_diff = pos_proj.mean() - neg_proj.mean()
    pooled_std = torch.sqrt((pos_proj.var() + neg_proj.var()) / 2)
    
    cohens_d = float(mean_diff / (pooled_std + 1e-8))
    
    return abs(cohens_d)


def compute_null_distribution(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_samples: int = 50,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute null distribution using random Gaussian noise with same statistics.
    
    This creates a baseline of what metrics look like when there's
    NO real concept - just random vectors with same mean/std as data.
    
    The key insight: if our concept is real, our metrics should be
    MUCH better than random noise with same basic statistics.
    
    Returns distribution of:
    - coherence under null
    - stability under null  
    - linear_probe accuracy under null
    - SNR under null
    """
    rng = np.random.RandomState(random_seed)
    n_pairs = pos_activations.shape[0]
    hidden_dim = pos_activations.shape[1]
    
    # Get statistics from real data
    all_acts = torch.cat([pos_activations, neg_activations], dim=0)
    data_mean = all_acts.mean().item()
    data_std = all_acts.std().item()
    
    null_coherences = []
    null_stabilities = []
    null_accuracies = []
    null_snrs = []
    
    for _ in range(n_samples):
        # Generate random "pos" and "neg" with same statistics as real data
        # No structure, just noise
        fake_pos = torch.tensor(rng.randn(n_pairs, hidden_dim) * data_std + data_mean, dtype=torch.float32)
        fake_neg = torch.tensor(rng.randn(n_pairs, hidden_dim) * data_std + data_mean, dtype=torch.float32)
        
        # Compute metrics on random pairs
        coherence, _ = compute_concept_coherence(fake_pos, fake_neg)
        null_coherences.append(coherence)
        
        # Only compute expensive metrics occasionally
        if len(null_stabilities) < 20:
            stability, _ = compute_concept_stability(fake_pos, fake_neg)
            null_stabilities.append(stability)
            
            acc = compute_linear_probe_accuracy(fake_pos, fake_neg)
            null_accuracies.append(acc)
            
            snr = compute_signal_to_noise(fake_pos, fake_neg)
            null_snrs.append(snr)
    
    return {
        "coherence": {
            "mean": float(np.mean(null_coherences)),
            "std": float(np.std(null_coherences)),
            "p95": float(np.percentile(null_coherences, 95)),
        },
        "stability": {
            "mean": float(np.mean(null_stabilities)) if null_stabilities else 0.0,
            "std": float(np.std(null_stabilities)) if null_stabilities else 0.0,
            "p95": float(np.percentile(null_stabilities, 95)) if null_stabilities else 0.5,
        },
        "accuracy": {
            "mean": float(np.mean(null_accuracies)) if null_accuracies else 0.5,
            "std": float(np.std(null_accuracies)) if null_accuracies else 0.05,
            "p95": float(np.percentile(null_accuracies, 95)) if null_accuracies else 0.55,
        },
        "snr": {
            "mean": float(np.mean(null_snrs)) if null_snrs else 0.0,
            "std": float(np.std(null_snrs)) if null_snrs else 0.1,
            "p95": float(np.percentile(null_snrs, 95)) if null_snrs else 0.3,
        },
    }


def compare_to_null(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_permutations: int = 100,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compare real metrics to null distribution to determine if concept is real.
    
    Returns:
        - p_values for each metric (probability of seeing this value under null)
        - z_scores (how many std above null mean)
        - is_significant (True if clearly above null)
    """
    # Get real metrics
    real_coherence, _ = compute_concept_coherence(pos_activations, neg_activations)
    real_stability, _ = compute_concept_stability(pos_activations, neg_activations)
    real_accuracy = compute_linear_probe_accuracy(pos_activations, neg_activations)
    real_snr = compute_signal_to_noise(pos_activations, neg_activations)
    
    if verbose:
        print("Computing null distribution...")
    
    # Get null distribution
    null = compute_null_distribution(pos_activations, neg_activations, n_permutations)
    
    # Compute z-scores (how many std above null mean)
    def z_score(real, null_mean, null_std):
        if null_std < 1e-8:
            return 0.0 if abs(real - null_mean) < 1e-8 else 10.0
        return (real - null_mean) / null_std
    
    z_coherence = z_score(real_coherence, null["coherence"]["mean"], null["coherence"]["std"])
    z_stability = z_score(real_stability, null["stability"]["mean"], null["stability"]["std"])
    z_accuracy = z_score(real_accuracy, null["accuracy"]["mean"], null["accuracy"]["std"])
    z_snr = z_score(real_snr, null["snr"]["mean"], null["snr"]["std"])
    
    # Significant if >2 std above null (p < 0.025 one-tailed)
    significant_threshold = 2.0
    
    results = {
        "coherence": {
            "real": real_coherence,
            "null_mean": null["coherence"]["mean"],
            "null_p95": null["coherence"]["p95"],
            "z_score": z_coherence,
            "is_significant": z_coherence > significant_threshold,
            "above_null_p95": real_coherence > null["coherence"]["p95"],
        },
        "stability": {
            "real": real_stability,
            "null_mean": null["stability"]["mean"],
            "null_p95": null["stability"]["p95"],
            "z_score": z_stability,
            "is_significant": z_stability > significant_threshold,
            "above_null_p95": real_stability > null["stability"]["p95"],
        },
        "accuracy": {
            "real": real_accuracy,
            "null_mean": null["accuracy"]["mean"],
            "null_p95": null["accuracy"]["p95"],
            "z_score": z_accuracy,
            "is_significant": z_accuracy > significant_threshold,
            "above_null_p95": real_accuracy > null["accuracy"]["p95"],
        },
        "snr": {
            "real": real_snr,
            "null_mean": null["snr"]["mean"],
            "null_p95": null["snr"]["p95"],
            "z_score": z_snr,
            "is_significant": z_snr > significant_threshold,
            "above_null_p95": real_snr > null["snr"]["p95"],
        },
        # Overall: significant if majority of metrics are significant
        "overall_significant": sum([
            z_coherence > significant_threshold,
            z_stability > significant_threshold,
            z_accuracy > significant_threshold,
            z_snr > significant_threshold,
        ]) >= 3,
        "mean_z_score": float(np.mean([z_coherence, z_stability, z_accuracy, z_snr])),
    }
    
    if verbose:
        print(f"\nComparison to null distribution:")
        print(f"  Coherence: {real_coherence:.3f} vs null {null['coherence']['mean']:.3f} (z={z_coherence:.1f}) {'***' if results['coherence']['is_significant'] else ''}")
        print(f"  Stability: {real_stability:.3f} vs null {null['stability']['mean']:.3f} (z={z_stability:.1f}) {'***' if results['stability']['is_significant'] else ''}")
        print(f"  Accuracy:  {real_accuracy:.3f} vs null {null['accuracy']['mean']:.3f} (z={z_accuracy:.1f}) {'***' if results['accuracy']['is_significant'] else ''}")
        print(f"  SNR:       {real_snr:.3f} vs null {null['snr']['mean']:.3f} (z={z_snr:.1f}) {'***' if results['snr']['is_significant'] else ''}")
        print(f"\n  Overall significant: {results['overall_significant']} (mean z={results['mean_z_score']:.1f})")
    
    return results


def validate_concept(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    coherence_threshold: float = 0.3,
    stability_threshold: float = 0.8,
    signal_threshold: float = 0.6,
    snr_threshold: float = 0.5,
    verbose: bool = False,
) -> ConceptValidityResult:
    """
    Validate if a set of contrastive pairs represents a true concept.
    
    A valid concept has:
    1. High coherence - all pairs point roughly same direction
    2. High stability - direction consistent across subsamples
    3. Strong signal - can separate pos from neg
    4. Good SNR - clear separation, not noise
    
    Args:
        pos_activations: Positive activations [n_pairs, hidden_dim]
        neg_activations: Negative activations [n_pairs, hidden_dim]
        coherence_threshold: Min coherence to be valid
        stability_threshold: Min stability to be valid
        signal_threshold: Min accuracy to have signal
        snr_threshold: Min signal-to-noise ratio
        verbose: Print progress
        
    Returns:
        ConceptValidityResult with detailed analysis
    """
    n_pairs = pos_activations.shape[0]
    
    if verbose:
        print(f"Validating concept with {n_pairs} pairs...")
    
    # 1. Coherence
    coherence, coherence_std = compute_concept_coherence(pos_activations, neg_activations)
    is_coherent = coherence > coherence_threshold
    
    if verbose:
        print(f"  Coherence: {coherence:.3f} +/- {coherence_std:.3f} -> {'PASS' if is_coherent else 'FAIL'}")
    
    # 2. Stability
    stability, stability_std = compute_concept_stability(pos_activations, neg_activations)
    is_stable = stability > stability_threshold
    
    if verbose:
        print(f"  Stability: {stability:.3f} +/- {stability_std:.3f} -> {'PASS' if is_stable else 'FAIL'}")
    
    # 3. Signal strength
    signal_strength = compute_linear_probe_accuracy(pos_activations, neg_activations)
    has_signal = signal_strength > signal_threshold
    
    if verbose:
        print(f"  Signal: {signal_strength:.3f} -> {'PASS' if has_signal else 'FAIL'}")
    
    # 4. Signal-to-noise
    snr = compute_signal_to_noise(pos_activations, neg_activations)
    good_snr = snr > snr_threshold
    
    if verbose:
        print(f"  SNR: {snr:.3f} -> {'PASS' if good_snr else 'FAIL'}")
    
    # 5. ICD for granularity
    icd_results = compute_icd(pos_activations, neg_activations)
    icd = icd_results["icd"]
    
    # Specificity: lower ICD = more abstract, higher ICD = more specific
    # Normalize to 0-1 range (assuming ICD typically 1-50)
    specificity = min(1.0, icd / 20.0)
    
    if verbose:
        print(f"  ICD: {icd:.2f}, Specificity: {specificity:.2f}")
    
    # Determine concept level based on characteristics
    if not has_signal:
        concept_level = "noise"
    elif icd < 3 and coherence > 0.5:
        concept_level = "domain"  # Very abstract, single direction
    elif icd < 10 and coherence > 0.3:
        concept_level = "category"  # Moderately abstract
    else:
        concept_level = "instance"  # Specific instances
    
    # Overall validity
    is_valid = is_coherent and is_stable and has_signal
    
    # Validity score (weighted average)
    validity_score = (
        0.3 * coherence +
        0.3 * stability +
        0.2 * min(1.0, signal_strength) +
        0.2 * min(1.0, snr / 2.0)
    )
    
    if verbose:
        print(f"\nConclusion: {'VALID' if is_valid else 'INVALID'} {concept_level} (score: {validity_score:.2f})")
    
    return ConceptValidityResult(
        is_valid_concept=is_valid,
        validity_score=validity_score,
        concept_level=concept_level,
        coherence_score=coherence,
        coherence_std=coherence_std,
        is_coherent=is_coherent,
        stability_score=stability,
        stability_std=stability_std,
        is_stable=is_stable,
        signal_strength=signal_strength,
        signal_to_noise=snr,
        has_signal=has_signal,
        icd=icd,
        specificity=specificity,
    )


def compare_concept_granularity(
    concepts: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compare multiple concepts to understand their hierarchical relationship.
    
    Args:
        concepts: Dict mapping concept_name -> (pos_activations, neg_activations)
        verbose: Print progress
        
    Returns:
        Dict with:
            - per_concept: validity results for each concept
            - hierarchy: detected hierarchy (if any)
            - direction_similarity: pairwise similarity matrix
    """
    results = {}
    directions = {}
    
    # Validate each concept
    for name, (pos, neg) in concepts.items():
        if verbose:
            print(f"\n=== {name} ===")
        
        validity = validate_concept(pos, neg, verbose=verbose)
        results[name] = validity.to_dict()
        
        # Store direction
        diff = pos.mean(dim=0) - neg.mean(dim=0)
        directions[name] = diff / (diff.norm() + 1e-8)
    
    # Compute pairwise direction similarities
    names = list(concepts.keys())
    n = len(names)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                sim = torch.nn.functional.cosine_similarity(
                    directions[names[i]].unsqueeze(0),
                    directions[names[j]].unsqueeze(0)
                ).item()
                sim_matrix[i, j] = sim
    
    # Detect hierarchy based on similarity + ICD
    # Higher-level concepts often have lower ICD and include lower-level ones
    hierarchy = []
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i != j:
                sim = sim_matrix[i, j]
                icd_i = results[name_i]["icd"]
                icd_j = results[name_j]["icd"]
                
                # If high similarity AND different ICD, might be hierarchy
                if sim > 0.5 and abs(icd_i - icd_j) > 5:
                    parent = name_i if icd_i < icd_j else name_j
                    child = name_j if icd_i < icd_j else name_i
                    hierarchy.append({
                        "parent": parent,
                        "child": child,
                        "similarity": sim,
                        "icd_diff": abs(icd_i - icd_j),
                    })
    
    if verbose and hierarchy:
        print(f"\nDetected hierarchy:")
        for h in hierarchy:
            print(f"  {h['parent']} -> {h['child']} (sim={h['similarity']:.2f})")
    
    return {
        "per_concept": results,
        "direction_similarity": {
            "matrix": sim_matrix.tolist(),
            "names": names,
        },
        "hierarchy": hierarchy,
    }


# =============================================================================
# MULTI-CONCEPT DECOMPOSITION - What concepts exist within a pair set?
# =============================================================================

@dataclass
class ConceptDecomposition:
    """Result from decomposing a contrastive pair set into constituent concepts."""
    # How many concepts found
    n_concepts: int
    decomposition_method: str  # "clustering", "nmf", "ica"
    
    # Per-concept info
    concept_directions: List[torch.Tensor]
    concept_sizes: List[int]  # How many pairs belong to each
    concept_coherences: List[float]
    concept_validities: List[bool]
    
    # Relationships between concepts
    concept_relationships: List[Dict[str, Any]]  # List of {concept_i, concept_j, relationship, strength}
    
    # Pair assignments
    pair_to_concept: List[int]  # Which concept each pair belongs to (-1 if unclear)
    pair_concept_scores: np.ndarray  # [n_pairs, n_concepts] soft assignment scores
    
    # Overall quality
    total_explained_variance: float
    reconstruction_error: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_concepts": self.n_concepts,
            "decomposition_method": self.decomposition_method,
            "concept_sizes": self.concept_sizes,
            "concept_coherences": self.concept_coherences,
            "concept_validities": self.concept_validities,
            "concept_relationships": self.concept_relationships,
            "pair_to_concept": self.pair_to_concept,
            "total_explained_variance": self.total_explained_variance,
            "reconstruction_error": self.reconstruction_error,
        }


def decompose_into_concepts(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    max_concepts: int = 5,
    min_concept_size: int = 5,
    method: str = "auto",
    verbose: bool = False,
) -> ConceptDecomposition:
    """
    Decompose a contrastive pair set into constituent concepts.
    
    This finds the underlying concepts and how they relate:
    - Which pairs belong to which concept
    - Are concepts independent, correlated, or conflicting
    - Which concept each pair primarily represents
    
    Methods:
    - "clustering": K-means on diff vectors
    - "nmf": Non-negative matrix factorization (finds additive components)
    - "ica": Independent Component Analysis (finds statistically independent sources)
    - "auto": Try all, pick best
    
    Args:
        pos_activations: Positive activations [n_pairs, hidden_dim]
        neg_activations: Negative activations [n_pairs, hidden_dim]
        max_concepts: Maximum concepts to look for
        min_concept_size: Minimum pairs per concept
        method: Decomposition method
        verbose: Print progress
        
    Returns:
        ConceptDecomposition with full analysis
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import NMF, FastICA, PCA
    from sklearn.metrics import silhouette_score
    
    n_pairs = pos_activations.shape[0]
    diff_vectors = (pos_activations - neg_activations).numpy()
    
    if verbose:
        print(f"Decomposing {n_pairs} pairs into concepts (max={max_concepts})...")
    
    # =========================================================================
    # Step 1: Determine optimal number of concepts
    # =========================================================================
    # Use multiple methods to estimate
    
    # Method A: Clustering silhouette - only accept if silhouette is good
    best_k_cluster = 1
    best_sil = 0.1  # Minimum threshold for splitting
    for k in range(2, min(max_concepts + 1, n_pairs // min_concept_size)):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(diff_vectors)
            if len(set(labels)) > 1:
                sil = silhouette_score(diff_vectors, labels)
                if sil > best_sil:
                    best_sil = sil
                    best_k_cluster = k
        except:
            pass
    
    # Method B: Check if all diff vectors are highly correlated (single concept)
    # Compute mean direction and check alignment
    mean_diff = diff_vectors.mean(axis=0)
    mean_diff_norm = mean_diff / (np.linalg.norm(mean_diff) + 1e-8)
    alignments = diff_vectors @ mean_diff_norm / (np.linalg.norm(diff_vectors, axis=1) + 1e-8)
    mean_alignment = np.mean(np.abs(alignments))
    
    # If highly aligned (>0.5), likely single concept
    if mean_alignment > 0.5 and best_sil < 0.3:
        best_k_cluster = 1
    
    # Method C: PCA effective rank (but only trust if silhouette confirms)
    pca = PCA(n_components=min(10, n_pairs - 1))
    pca.fit(diff_vectors)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    effective_rank = np.searchsorted(cumvar, 0.9) + 1
    
    # Combine estimates - prefer clustering result, but cap by PCA rank
    n_concepts = min(best_k_cluster, effective_rank, max_concepts)
    n_concepts = max(1, n_concepts)
    
    if verbose:
        print(f"  Estimated concepts: {n_concepts} (cluster={best_k_cluster}, pca_rank={effective_rank})")
    
    # =========================================================================
    # Step 2: Decompose using selected method
    # =========================================================================
    if method == "auto":
        # Try clustering first (most interpretable)
        method = "clustering"
    
    if method == "clustering" or n_concepts == 1:
        # K-means clustering
        if n_concepts > 1:
            kmeans = KMeans(n_clusters=n_concepts, random_state=42, n_init=10)
            labels = kmeans.fit_predict(diff_vectors)
            
            # Soft assignments based on distance
            distances = kmeans.transform(diff_vectors)  # [n_pairs, k]
            pair_concept_scores = 1.0 / (distances + 1e-8)
            pair_concept_scores = pair_concept_scores / pair_concept_scores.sum(axis=1, keepdims=True)
        else:
            labels = np.zeros(n_pairs, dtype=int)
            pair_concept_scores = np.ones((n_pairs, 1))
        
        # Get concept directions
        concept_directions = []
        concept_sizes = []
        for c in range(n_concepts):
            mask = labels == c
            if mask.sum() > 0:
                direction = torch.tensor(diff_vectors[mask].mean(axis=0))
                direction = direction / (direction.norm() + 1e-8)
                concept_directions.append(direction)
                concept_sizes.append(int(mask.sum()))
            else:
                concept_directions.append(torch.zeros(diff_vectors.shape[1]))
                concept_sizes.append(0)
        
        pair_to_concept = labels.tolist()
        decomposition_method = "clustering"
        
    elif method == "nmf":
        # Non-negative Matrix Factorization
        # Shift to non-negative
        diff_shifted = diff_vectors - diff_vectors.min()
        
        nmf = NMF(n_components=n_concepts, random_state=42, max_iter=500)
        W = nmf.fit_transform(diff_shifted)  # [n_pairs, n_concepts]
        H = nmf.components_  # [n_concepts, hidden_dim]
        
        concept_directions = [torch.tensor(H[c]) for c in range(n_concepts)]
        concept_directions = [d / (d.norm() + 1e-8) for d in concept_directions]
        
        pair_concept_scores = W / (W.sum(axis=1, keepdims=True) + 1e-8)
        pair_to_concept = W.argmax(axis=1).tolist()
        concept_sizes = [(np.array(pair_to_concept) == c).sum() for c in range(n_concepts)]
        decomposition_method = "nmf"
        
    elif method == "ica":
        # Independent Component Analysis
        ica = FastICA(n_components=n_concepts, random_state=42, max_iter=500)
        S = ica.fit_transform(diff_vectors)  # [n_pairs, n_concepts]
        A = ica.mixing_  # [hidden_dim, n_concepts]
        
        concept_directions = [torch.tensor(A[:, c]) for c in range(n_concepts)]
        concept_directions = [d / (d.norm() + 1e-8) for d in concept_directions]
        
        # Soft assignment based on source magnitude
        pair_concept_scores = np.abs(S)
        pair_concept_scores = pair_concept_scores / (pair_concept_scores.sum(axis=1, keepdims=True) + 1e-8)
        pair_to_concept = np.abs(S).argmax(axis=1).tolist()
        concept_sizes = [(np.array(pair_to_concept) == c).sum() for c in range(n_concepts)]
        decomposition_method = "ica"
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # =========================================================================
    # Step 3: Validate each concept
    # =========================================================================
    concept_coherences = []
    concept_validities = []
    
    for c in range(n_concepts):
        mask = np.array(pair_to_concept) == c
        if mask.sum() >= min_concept_size:
            pos_c = pos_activations[mask]
            neg_c = neg_activations[mask]
            
            coherence, _ = compute_concept_coherence(pos_c, neg_c)
            concept_coherences.append(coherence)
            
            # Valid if coherent and stable
            stability, _ = compute_concept_stability(pos_c, neg_c)
            concept_validities.append(coherence > 0.3 and stability > 0.7)
        else:
            concept_coherences.append(0.0)
            concept_validities.append(False)
    
    if verbose:
        print(f"  Concept coherences: {[f'{c:.2f}' for c in concept_coherences]}")
        print(f"  Valid concepts: {sum(concept_validities)}/{n_concepts}")
    
    # =========================================================================
    # Step 4: Analyze relationships between concepts
    # =========================================================================
    concept_relationships = []
    
    for i in range(n_concepts):
        for j in range(i + 1, n_concepts):
            # Cosine similarity between directions
            sim = torch.nn.functional.cosine_similarity(
                concept_directions[i].unsqueeze(0).float(),
                concept_directions[j].unsqueeze(0).float()
            ).item()
            
            # Determine relationship type
            if abs(sim) < 0.2:
                relationship = "independent"
            elif sim > 0.5:
                relationship = "correlated"
            elif sim < -0.5:
                relationship = "conflicting"
            elif 0.2 <= sim <= 0.5:
                relationship = "weakly_correlated"
            else:  # -0.5 <= sim <= -0.2
                relationship = "weakly_conflicting"
            
            concept_relationships.append({
                "concept_i": i,
                "concept_j": j,
                "similarity": float(sim),
                "relationship": relationship,
            })
            
            if verbose:
                print(f"  Concept {i} <-> Concept {j}: {relationship} (sim={sim:.2f})")
    
    # =========================================================================
    # Step 5: Compute reconstruction quality
    # =========================================================================
    # How well do the concepts explain the data?
    if n_concepts > 1:
        # Reconstruct each diff as weighted sum of concept directions
        directions_matrix = torch.stack(concept_directions).T.numpy()  # [hidden_dim, n_concepts]
        
        # Project diffs onto concept directions
        projections = diff_vectors @ directions_matrix  # [n_pairs, n_concepts]
        reconstructed = projections @ directions_matrix.T  # [n_pairs, hidden_dim]
        
        reconstruction_error = float(np.mean((diff_vectors - reconstructed) ** 2))
        total_variance = float(np.var(diff_vectors))
        explained_variance = 1.0 - (reconstruction_error / (total_variance + 1e-8))
        explained_variance = max(0.0, min(1.0, explained_variance))
    else:
        explained_variance = 1.0
        reconstruction_error = 0.0
    
    if verbose:
        print(f"  Explained variance: {explained_variance:.1%}")
    
    return ConceptDecomposition(
        n_concepts=n_concepts,
        decomposition_method=decomposition_method,
        concept_directions=concept_directions,
        concept_sizes=concept_sizes,
        concept_coherences=concept_coherences,
        concept_validities=concept_validities,
        concept_relationships=concept_relationships,
        pair_to_concept=pair_to_concept,
        pair_concept_scores=pair_concept_scores,
        total_explained_variance=explained_variance,
        reconstruction_error=reconstruction_error,
    )


def find_mixed_pairs(
    decomposition: ConceptDecomposition,
    threshold: float = 0.4,
) -> List[int]:
    """
    Find pairs that belong to multiple concepts (mixed/ambiguous pairs).
    
    These pairs might represent:
    - Overlapping concepts
    - Transitions between concepts
    - Noise
    
    Args:
        decomposition: Result from decompose_into_concepts
        threshold: Max score for primary concept to be considered "mixed"
        
    Returns:
        List of pair indices that are mixed
    """
    scores = decomposition.pair_concept_scores
    max_scores = scores.max(axis=1)
    
    # Mixed if no concept dominates
    mixed_indices = np.where(max_scores < threshold)[0].tolist()
    
    return mixed_indices


def get_pure_concept_pairs(
    decomposition: ConceptDecomposition,
    concept_idx: int,
    threshold: float = 0.7,
) -> List[int]:
    """
    Get pairs that strongly belong to a specific concept.
    
    Args:
        decomposition: Result from decompose_into_concepts
        concept_idx: Which concept
        threshold: Min score to be considered "pure"
        
    Returns:
        List of pair indices that purely represent this concept
    """
    scores = decomposition.pair_concept_scores
    
    if concept_idx >= scores.shape[1]:
        return []
    
    concept_scores = scores[:, concept_idx]
    pure_indices = np.where(concept_scores >= threshold)[0].tolist()
    
    return pure_indices


def recommend_per_concept_steering(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    decomposition: ConceptDecomposition,
    verbose: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Recommend steering approach for each detected concept.
    
    Different concepts may need different steering methods:
    - Pure concept with high coherence -> CAA
    - Mixed concept -> PRISM (multi-direction)
    - Conflicting concepts -> need separate handling
    
    Args:
        pos_activations: Original positive activations
        neg_activations: Original negative activations  
        decomposition: Concept decomposition result
        verbose: Print progress
        
    Returns:
        Dict mapping concept_idx to recommendation
    """
    recommendations = {}
    
    for c in range(decomposition.n_concepts):
        if verbose:
            print(f"\nConcept {c}:")
        
        # Get pure pairs for this concept
        pure_pairs = get_pure_concept_pairs(decomposition, c, threshold=0.6)
        
        if len(pure_pairs) < 10:
            recommendations[c] = {
                "method": "SKIP",
                "reason": f"Too few pure pairs ({len(pure_pairs)})",
                "n_pairs": len(pure_pairs),
            }
            if verbose:
                print(f"  SKIP: only {len(pure_pairs)} pure pairs")
            continue
        
        # Analyze this concept's pairs
        mask = torch.tensor([i in pure_pairs for i in range(len(pos_activations))])
        pos_c = pos_activations[mask]
        neg_c = neg_activations[mask]
        
        # Quick analysis
        coherence, _ = compute_concept_coherence(pos_c, neg_c)
        signal = compute_linear_probe_accuracy(pos_c, neg_c)
        
        # Check for conflicts with other concepts
        has_conflict = any(
            rel["concept_i"] == c or rel["concept_j"] == c
            for rel in decomposition.concept_relationships
            if rel["relationship"] == "conflicting"
        )
        
        # Recommend method
        if signal < 0.6:
            method = "NO_METHOD"
            reason = "No signal"
        elif has_conflict:
            method = "PRISM"
            reason = "Conflicting with other concepts"
        elif coherence > 0.5:
            method = "CAA"
            reason = f"High coherence ({coherence:.2f})"
        elif coherence > 0.3:
            method = "PRISM"
            reason = f"Moderate coherence ({coherence:.2f})"
        else:
            method = "TITAN"
            reason = f"Low coherence ({coherence:.2f})"
        
        recommendations[c] = {
            "method": method,
            "reason": reason,
            "n_pairs": len(pure_pairs),
            "coherence": coherence,
            "signal": signal,
            "direction": decomposition.concept_directions[c],
        }
        
        if verbose:
            print(f"  Method: {method}")
            print(f"  Reason: {reason}")
            print(f"  Pairs: {len(pure_pairs)}, Coherence: {coherence:.2f}, Signal: {signal:.2f}")
    
    return recommendations


def compute_geometry_metrics(
    cached: CachedActivations,
    layers: List[int],
) -> GeometryTestResult:
    """
    Compute geometry metrics for a layer combination from cached activations.
    
    Uses the comprehensive detect_geometry_structure() to get scores for:
    - linear, cone, cluster, manifold, sparse, bimodal, orthogonal
    
    Args:
        cached: Cached activations with all layers
        layers: Layer indices (0-based) to analyze
        
    Returns:
        GeometryTestResult with all structure scores
    """
    from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
        detect_geometry_structure,
        GeometryAnalysisConfig,
    )
    
    # Stack positive and negative activations for specified layers
    # Convert 0-based indices to 1-based layer names used in cache
    pos_acts_list = []
    neg_acts_list = []
    
    for layer_idx in layers:
        layer_name = str(layer_idx + 1)  # Convert 0-based to 1-based
        try:
            pos = cached.get_positive_activations(layer_name)  # [num_pairs, hidden_size]
            neg = cached.get_negative_activations(layer_name)  # [num_pairs, hidden_size]
            pos_acts_list.append(pos)
            neg_acts_list.append(neg)
        except (KeyError, IndexError):
            continue
    
    if not pos_acts_list:
        return GeometryTestResult(
            benchmark=cached.benchmark,
            strategy=cached.strategy.value,
            layers=layers,
            signal_strength=0.5,
            has_signal=False,
            linear_probe_accuracy=0.5,
            is_linear=False,
            # Nonlinear metrics
            knn_accuracy_k5=0.5,
            knn_accuracy_k10=0.5,
            knn_accuracy_k20=0.5,
            knn_pca_accuracy=0.5,
            knn_umap_accuracy=0.5,
            knn_pacmap_accuracy=0.5,
            mlp_probe_accuracy=0.5,
            best_nonlinear_accuracy=0.5,
            mmd_rbf=0.0,
            local_dim_pos=0.0,
            local_dim_neg=0.0,
            local_dim_ratio=1.0,
            fisher_max=0.0,
            fisher_gini=0.0,
            fisher_top10_ratio=0.0,
            num_dims_fisher_above_1=0,
            density_ratio=1.0,
            # Structure scores
            best_structure="error",
            best_score=0.0,
            linear_score=0.0,
            cone_score=0.0,
            orthogonal_score=0.0,
            manifold_score=0.0,
            sparse_score=0.0,
            cluster_score=0.0,
            bimodal_score=0.0,
            cohens_d=0.0,
            variance_explained=0.0,
            within_class_consistency=0.0,
            raw_mean_cosine_similarity=0.0,
            positive_correlation_fraction=0.0,
            near_zero_fraction=0.0,
            pca_top2_variance=0.0,
            local_nonlinearity=0.0,
            gini_coefficient=0.0,
            active_fraction=0.0,
            top_10_contribution=0.0,
            best_silhouette=0.0,
            best_k=0,
            # Multi-direction
            multi_dir_accuracy_k1=0.5,
            multi_dir_accuracy_k2=0.5,
            multi_dir_accuracy_k3=0.5,
            multi_dir_accuracy_k5=0.5,
            multi_dir_accuracy_k10=0.5,
            multi_dir_min_k_for_good=-1,
            multi_dir_saturation_k=1,
            multi_dir_gain=0.0,
            # Steerability (default/error values)
            diff_mean_alignment=0.0,
            pct_positive_alignment=0.5,
            steering_vector_norm_ratio=0.0,
            cluster_direction_angle=90.0,
            per_cluster_alignment_k2=0.0,
            spherical_silhouette_k2=0.0,
            effective_steering_dims=1,
            steerability_score=0.0,
            recommended_method="error: no activations",
        )
    
    # Concatenate across layers: [num_pairs, hidden_size * num_layers]
    pos_activations = torch.cat(pos_acts_list, dim=-1)
    neg_activations = torch.cat(neg_acts_list, dim=-1)
    
    # Convert to float32 for geometry analysis (bf16/float16 can cause dtype mismatches)
    pos_activations = pos_activations.float()
    neg_activations = neg_activations.float()
    
    # Run comprehensive geometry detection
    config = GeometryAnalysisConfig(
        num_components=5,
        optimization_steps=50,  # Reduced for speed since we're testing many combos
    )
    
    try:
        result = detect_geometry_structure(pos_activations, neg_activations, config)
        
        # Step 1: Compute ALL probe accuracies first
        signal_strength = compute_signal_strength(pos_activations, neg_activations)  # MLP CV
        linear_probe_accuracy = compute_linear_probe_accuracy(pos_activations, neg_activations)
        
        # Compute nonlinear signal metrics
        knn_k5 = compute_knn_accuracy(pos_activations, neg_activations, k=5)
        knn_k10 = compute_knn_accuracy(pos_activations, neg_activations, k=10)
        knn_k20 = compute_knn_accuracy(pos_activations, neg_activations, k=20)
        knn_pca = compute_knn_pca_accuracy(pos_activations, neg_activations, k=10, n_components=50)
        knn_umap = compute_knn_umap_accuracy(pos_activations, neg_activations, k=10, n_components=10)
        knn_pacmap = compute_knn_pacmap_accuracy(pos_activations, neg_activations, k=10, n_components=10)
        mlp_probe = compute_mlp_probe_accuracy(pos_activations, neg_activations, hidden_size=64)
        mmd = compute_mmd_rbf(pos_activations, neg_activations)
        local_dim_pos, local_dim_neg, local_dim_ratio = compute_local_intrinsic_dims(pos_activations, neg_activations)
        fisher_stats = compute_fisher_per_dimension(pos_activations, neg_activations)
        density_rat = compute_density_ratio(pos_activations, neg_activations)
        
        # Multi-direction analysis: how many directions needed?
        multi_dir_results = compute_multi_direction_accuracy(pos_activations, neg_activations)
        
        # Steerability metrics: predict whether CAA steering will work
        steerability = compute_steerability_metrics(pos_activations, neg_activations)
        
        # ICD (Intrinsic Concept Dimensionality) - measures effective rank of differences
        icd_results = compute_icd(pos_activations, neg_activations)
        
        # Use MAXIMUM of all nonlinear probes to detect signal
        best_nonlinear = max(knn_k10, knn_pca, knn_umap, knn_pacmap, mlp_probe)
        
        # Helper to safely get detail from structure analysis
        def get_detail(struct_name: str, key: str, default=0.0):
            if struct_name in result.all_scores:
                return result.all_scores[struct_name].details.get(key, default)
            return default
        
        # Get cohens_d for quality metric
        cohens_d_value = get_detail("linear", "cohens_d", 0.0)
        
        # Use new comprehensive recommendation function
        rec_result = compute_recommendation(
            linear_probe_accuracy=linear_probe_accuracy,
            best_nonlinear_accuracy=best_nonlinear,
            knn_umap_accuracy=knn_umap,
            knn_pacmap_accuracy=knn_pacmap,
            icd=icd_results["icd"],
            icd_top1_variance=icd_results["top1_variance"],
            diff_mean_alignment=steerability["diff_mean_alignment"],
            steerability_score=steerability["steerability_score"],
            effective_steering_dims=steerability["effective_steering_dims"],
            multi_dir_gain=multi_dir_results["gain_from_multi"],
            spherical_silhouette_k2=steerability["spherical_silhouette_k2"],
            cluster_direction_angle=steerability["cluster_direction_angle"],
            cohens_d=cohens_d_value,
            signal_above_baseline=0.0,  # Will be computed later if needed
        )
        
        has_signal = rec_result["signal_exists"]
        is_linear = rec_result["signal_type"] == "LINEAR"
        recommendation = rec_result["recommended_method"]
        
        return GeometryTestResult(
            benchmark=cached.benchmark,
            strategy=cached.strategy.value,
            layers=layers,
            signal_strength=signal_strength,
            has_signal=has_signal,
            linear_probe_accuracy=linear_probe_accuracy,
            is_linear=is_linear,
            # Nonlinear metrics
            knn_accuracy_k5=knn_k5,
            knn_accuracy_k10=knn_k10,
            knn_accuracy_k20=knn_k20,
            knn_pca_accuracy=knn_pca,
            knn_umap_accuracy=knn_umap,
            knn_pacmap_accuracy=knn_pacmap,
            mlp_probe_accuracy=mlp_probe,
            best_nonlinear_accuracy=best_nonlinear,
            mmd_rbf=mmd,
            local_dim_pos=local_dim_pos,
            local_dim_neg=local_dim_neg,
            local_dim_ratio=local_dim_ratio,
            fisher_max=fisher_stats["fisher_max"],
            fisher_gini=fisher_stats["fisher_gini"],
            fisher_top10_ratio=fisher_stats["fisher_top10_ratio"],
            num_dims_fisher_above_1=fisher_stats["num_dims_fisher_above_1"],
            density_ratio=density_rat,
            # Structure scores
            best_structure=result.best_structure.value,
            best_score=result.best_score,
            linear_score=result.all_scores.get("linear", type('', (), {'score': 0.0})()).score,
            cone_score=result.all_scores.get("cone", type('', (), {'score': 0.0})()).score,
            orthogonal_score=result.all_scores.get("orthogonal", type('', (), {'score': 0.0})()).score,
            manifold_score=result.all_scores.get("manifold", type('', (), {'score': 0.0})()).score,
            sparse_score=result.all_scores.get("sparse", type('', (), {'score': 0.0})()).score,
            cluster_score=result.all_scores.get("cluster", type('', (), {'score': 0.0})()).score,
            bimodal_score=result.all_scores.get("bimodal", type('', (), {'score': 0.0})()).score,
            # Linear details
            cohens_d=get_detail("linear", "cohens_d", 0.0),
            variance_explained=get_detail("linear", "variance_explained", 0.0),
            within_class_consistency=get_detail("linear", "within_class_consistency", 0.0),
            # Cone details
            raw_mean_cosine_similarity=get_detail("cone", "raw_mean_cosine_similarity", 0.0),
            positive_correlation_fraction=get_detail("cone", "positive_correlation_fraction", 0.0),
            # Orthogonal details
            near_zero_fraction=get_detail("orthogonal", "near_zero_fraction", 0.0),
            # Manifold details
            pca_top2_variance=get_detail("manifold", "pca_top2_variance", 0.0),
            local_nonlinearity=get_detail("manifold", "local_nonlinearity", 0.0),
            # Sparse details
            gini_coefficient=get_detail("sparse", "gini_coefficient", 0.0),
            active_fraction=get_detail("sparse", "active_fraction", 0.0),
            top_10_contribution=get_detail("sparse", "top_10_contribution", 0.0),
            # Cluster details
            best_silhouette=get_detail("cluster", "best_silhouette", 0.0),
            best_k=int(get_detail("cluster", "best_k", 2)),
            # Multi-direction analysis
            multi_dir_accuracy_k1=multi_dir_results["accuracy_by_k"].get(1, 0.5),
            multi_dir_accuracy_k2=multi_dir_results["accuracy_by_k"].get(2, 0.5),
            multi_dir_accuracy_k3=multi_dir_results["accuracy_by_k"].get(3, 0.5),
            multi_dir_accuracy_k5=multi_dir_results["accuracy_by_k"].get(5, 0.5),
            multi_dir_accuracy_k10=multi_dir_results["accuracy_by_k"].get(10, 0.5),
            multi_dir_min_k_for_good=multi_dir_results["min_k_for_good"],
            multi_dir_saturation_k=multi_dir_results["saturation_k"],
            multi_dir_gain=multi_dir_results["gain_from_multi"],
            # Steerability metrics
            diff_mean_alignment=steerability["diff_mean_alignment"],
            pct_positive_alignment=steerability["pct_positive_alignment"],
            steering_vector_norm_ratio=steerability["steering_vector_norm_ratio"],
            cluster_direction_angle=steerability["cluster_direction_angle"],
            per_cluster_alignment_k2=steerability["per_cluster_alignment_k2"],
            spherical_silhouette_k2=steerability["spherical_silhouette_k2"],
            effective_steering_dims=steerability["effective_steering_dims"],
            steerability_score=steerability["steerability_score"],
            # ICD metrics
            icd=icd_results["icd"],
            icd_top1_variance=icd_results["top1_variance"],
            icd_top5_variance=icd_results["top5_variance"],
            # Recommendation based on signal analysis
            recommended_method=recommendation,
        )
    except Exception as e:
        return GeometryTestResult(
            benchmark=cached.benchmark,
            strategy=cached.strategy.value,
            layers=layers,
            signal_strength=0.5,
            has_signal=False,
            linear_probe_accuracy=0.5,
            is_linear=False,
            # Nonlinear metrics
            knn_accuracy_k5=0.5,
            knn_accuracy_k10=0.5,
            knn_accuracy_k20=0.5,
            knn_pca_accuracy=0.5,
            knn_umap_accuracy=0.5,
            knn_pacmap_accuracy=0.5,
            mlp_probe_accuracy=0.5,
            best_nonlinear_accuracy=0.5,
            mmd_rbf=0.0,
            local_dim_pos=0.0,
            local_dim_neg=0.0,
            local_dim_ratio=1.0,
            fisher_max=0.0,
            fisher_gini=0.0,
            fisher_top10_ratio=0.0,
            num_dims_fisher_above_1=0,
            density_ratio=1.0,
            # Structure scores
            best_structure="error",
            best_score=0.0,
            linear_score=0.0,
            cone_score=0.0,
            orthogonal_score=0.0,
            manifold_score=0.0,
            sparse_score=0.0,
            cluster_score=0.0,
            bimodal_score=0.0,
            cohens_d=0.0,
            variance_explained=0.0,
            within_class_consistency=0.0,
            raw_mean_cosine_similarity=0.0,
            positive_correlation_fraction=0.0,
            near_zero_fraction=0.0,
            pca_top2_variance=0.0,
            local_nonlinearity=0.0,
            gini_coefficient=0.0,
            active_fraction=0.0,
            top_10_contribution=0.0,
            best_silhouette=0.0,
            best_k=0,
            # Multi-direction
            multi_dir_accuracy_k1=0.5,
            multi_dir_accuracy_k2=0.5,
            multi_dir_accuracy_k3=0.5,
            multi_dir_accuracy_k5=0.5,
            multi_dir_accuracy_k10=0.5,
            multi_dir_min_k_for_good=-1,
            multi_dir_saturation_k=1,
            multi_dir_gain=0.0,
            # Steerability (default/error values)
            diff_mean_alignment=0.0,
            pct_positive_alignment=0.5,
            steering_vector_norm_ratio=0.0,
            cluster_direction_angle=90.0,
            per_cluster_alignment_k2=0.0,
            spherical_silhouette_k2=0.0,
            effective_steering_dims=1,
            steerability_score=0.0,
            recommended_method=f"error: {str(e)}",
        )


class GeometryRunner:
    """
    Runs geometry search across the search space.
    
    Uses activation caching for efficiency:
    1. Extract ALL layers once per (benchmark, strategy)
    2. Test all layer combinations from cache
    3. Compare against nonsense baseline (random tokens)
    """
    
    def __init__(
        self,
        search_space: GeometrySearchSpace,
        model: "WisentModel",
        cache_dir: Optional[str] = None,
    ):
        self.search_space = search_space
        self.model = model
        self.cache_dir = cache_dir or f"/tmp/wisent_geometry_cache_{model.model_name.replace('/', '_')}"
        self.cache = ActivationCache(self.cache_dir)
        # NEW: Raw activation cache (stores full sequences, shared between strategies in same family)
        self.raw_cache = RawActivationCache(self.cache_dir)
        # Cache for nonsense baseline activations per (n_pairs, layer)
        self._nonsense_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    
    def _get_nonsense_cache_path(self, n_pairs: int, layer: int) -> Path:
        """Get disk cache path for nonsense baseline."""
        cache_dir = Path(self.cache_dir) / "nonsense_baseline"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_prefix = self.model.model_name.replace("/", "_")
        return cache_dir / f"{model_prefix}_n{n_pairs}_layer{layer}.pt"
    
    def get_nonsense_baseline(
        self,
        n_pairs: int,
        layer: int,
        device: str = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get or generate nonsense baseline activations.
        
        Caches results both in memory and on disk so we only generate once 
        per (n_pairs, layer) combination. This ensures fair comparison: 
        same number of pairs as the benchmark.
        
        Args:
            n_pairs: Number of nonsense pairs to generate (should match benchmark size)
            layer: Which layer to extract from
            device: Device to use (default: model's device)
            
        Returns:
            Tuple of (nonsense_pos, nonsense_neg) tensors
        """
        cache_key = (n_pairs, layer)
        
        # Check memory cache first
        if cache_key in self._nonsense_cache:
            return self._nonsense_cache[cache_key]
        
        # Check disk cache
        cache_path = self._get_nonsense_cache_path(n_pairs, layer)
        if cache_path.exists():
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                nonsense_pos = cached["positive"]
                nonsense_neg = cached["negative"]
                self._nonsense_cache[cache_key] = (nonsense_pos, nonsense_neg)
                return nonsense_pos, nonsense_neg
            except Exception:
                # Corrupted cache, regenerate
                pass
        
        # Generate new nonsense activations
        device = device or str(self.model.hf_model.device)
        
        nonsense_pos, nonsense_neg = generate_nonsense_activations(
            model=self.model.hf_model,
            tokenizer=self.model.tokenizer,
            n_pairs=n_pairs,
            layer=layer,
            device=device,
        )
        
        # Save to memory cache
        self._nonsense_cache[cache_key] = (nonsense_pos, nonsense_neg)
        
        # Save to disk cache
        try:
            torch.save({
                "positive": nonsense_pos.cpu(),
                "negative": nonsense_neg.cpu(),
                "n_pairs": n_pairs,
                "layer": layer,
                "model": self.model.model_name,
            }, cache_path)
        except Exception:
            pass  # Disk cache is optional
        
        return nonsense_pos, nonsense_neg
    
    def clear_nonsense_cache(self, disk: bool = False) -> None:
        """
        Clear the nonsense baseline cache.
        
        Args:
            disk: If True, also clear disk cache
        """
        self._nonsense_cache.clear()
        
        if disk:
            cache_dir = Path(self.cache_dir) / "nonsense_baseline"
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
    
    def run(
        self,
        benchmarks: Optional[List[str]] = None,
        strategies: Optional[List[ExtractionStrategy]] = None,
        max_layer_combo_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> GeometrySearchResults:
        """
        Run the geometry search.
        
        Args:
            benchmarks: Benchmarks to test (default: all from search space)
            strategies: Strategies to test (default: all from search space)
            max_layer_combo_size: Override max layer combo size
            show_progress: Print progress
            
        Returns:
            GeometrySearchResults with all test results
        """
        benchmarks = benchmarks or self.search_space.benchmarks
        strategies = strategies or self.search_space.strategies
        max_combo = max_layer_combo_size or self.search_space.config.max_layer_combo_size
        
        # Get layer combinations
        num_layers = self.model.num_layers
        layer_combos = get_layer_combinations(num_layers, max_combo)
        
        results = GeometrySearchResults(
            model_name=self.model.model_name,
            config=self.search_space.config,
        )
        
        start_time = time.time()
        extraction_time = 0.0
        test_time = 0.0
        
        total_extractions = len(benchmarks) * len(strategies)
        extraction_count = 0
        
        for benchmark in benchmarks:
            for strategy in strategies:
                extraction_count += 1
                
                if show_progress:
                    print(f"\n[{extraction_count}/{total_extractions}] {benchmark} / {strategy.value}")
                
                # Get or create cached activations
                extract_start = time.time()
                try:
                    cached = self._get_cached_activations(benchmark, strategy, show_progress)
                except Exception as e:
                    if show_progress:
                        print(f"  SKIP: {e}")
                    continue
                extraction_time += time.time() - extract_start
                
                # Test all layer combinations
                test_start = time.time()
                for combo in layer_combos:
                    result = compute_geometry_metrics(cached, combo)
                    results.add_result(result)
                test_time += time.time() - test_start
                
                results.benchmarks_tested = len(set(r.benchmark for r in results.results))
                results.strategies_tested = len(set(r.strategy for r in results.results))
                results.layer_combos_tested = len(results.results)
                
                if show_progress:
                    print(f"  Tested {len(layer_combos)} layer combos")
        
        results.total_time_seconds = time.time() - start_time
        results.extraction_time_seconds = extraction_time
        results.test_time_seconds = test_time
        
        return results
    
    def _get_cached_activations(
        self,
        benchmark: str,
        strategy: ExtractionStrategy,
        show_progress: bool = True,
    ) -> CachedActivations:
        """
        Get cached activations, extracting if necessary.
        
        Uses raw activation cache to share data between strategies in the same
        text family (e.g., chat_last, chat_mean, chat_first all share same forward pass).
        """
        # Check legacy cache first (for backward compatibility)
        if self.cache.has(self.model.model_name, benchmark, strategy):
            if show_progress:
                print(f"  Loading from cache...")
            return self.cache.get(self.model.model_name, benchmark, strategy)
        
        # Check raw cache for this text family
        text_family = get_strategy_text_family(strategy)
        if self.raw_cache.has(self.model.model_name, benchmark, text_family):
            if show_progress:
                print(f"  Loading from raw cache ({text_family} family)...")
            raw_cached = self.raw_cache.get(self.model.model_name, benchmark, text_family)
            # Convert to CachedActivations for requested strategy
            cached = raw_cached.to_cached_activations(strategy, self.model.tokenizer)
            # Save to legacy cache for faster future access
            self.cache.put(cached)
            return cached
        
        # Need to extract - load pairs first
        if show_progress:
            print(f"  Loading pairs...")
        
        pairs = self._load_pairs(benchmark)
        
        if show_progress:
            print(f"  Extracting raw activations for {len(pairs)} pairs ({text_family} family)...")
        
        # Collect RAW activations (full sequences) - shared for all strategies in family
        raw_cached = collect_and_cache_raw_activations(
            model=self.model,
            pairs=pairs,
            benchmark=benchmark,
            strategy=strategy,  # Determines text family
            cache=self.raw_cache,
            show_progress=show_progress,
        )
        
        # Convert to CachedActivations for requested strategy
        cached = raw_cached.to_cached_activations(strategy, self.model.tokenizer)
        # Save to legacy cache for faster future access
        self.cache.put(cached)
        
        return cached
    
    def _load_pairs(self, benchmark: str) -> List:
        """Load contrastive pairs for a benchmark."""
        from lm_eval.tasks import TaskManager
        from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
        
        tm = TaskManager()
        try:
            task_dict = tm.load_task_or_group([benchmark])
            task = list(task_dict.values())[0]
        except Exception:
            task = None
        
        # pairs_per_benchmark <= 0 means "use all available"
        limit = self.search_space.config.pairs_per_benchmark
        if limit <= 0:
            limit = None  # No limit
        
        pairs = lm_build_contrastive_pairs(
            benchmark, 
            task, 
            limit=limit
        )
        
        # Random sample if we have more pairs than needed (only if limit is set)
        if limit and len(pairs) > limit:
            random.seed(self.search_space.config.random_seed)
            pairs = random.sample(pairs, limit)
        
        return pairs


def analyze_with_nonsense_baseline(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    nonsense_pos: torch.Tensor,
    nonsense_neg: torch.Tensor,
    benchmark_name: str = "unknown",
) -> Dict[str, Any]:
    """
    Analyze activations and compare against nonsense baseline.
    
    This is a standalone function for quick analysis without full GeometryRunner.
    
    Args:
        pos_activations: Positive class activations from benchmark
        neg_activations: Negative class activations from benchmark
        nonsense_pos: Positive activations from random tokens
        nonsense_neg: Negative activations from random tokens
        benchmark_name: Name for reporting
        
    Returns:
        Dict with ICD, baseline comparison, and verdict
    """
    # Ensure same number of samples for fair comparison
    n_real = min(len(pos_activations), len(neg_activations))
    n_nonsense = min(len(nonsense_pos), len(nonsense_neg))
    
    if n_real != n_nonsense:
        # Warn but continue with smaller size
        n = min(n_real, n_nonsense)
        pos_activations = pos_activations[:n]
        neg_activations = neg_activations[:n]
        nonsense_pos = nonsense_pos[:n]
        nonsense_neg = nonsense_neg[:n]
    
    # Compute ICD for both
    real_icd = compute_icd(pos_activations, neg_activations)
    nonsense_icd = compute_icd(nonsense_pos, nonsense_neg)
    
    # Compute baseline comparison
    baseline = compute_nonsense_baseline(
        pos_activations, neg_activations,
        nonsense_pos, nonsense_neg
    )
    
    # Determine verdict
    icd_ratio = nonsense_icd["icd"] / max(real_icd["icd"], 1e-6)
    
    if baseline["has_real_signal"]:
        if real_icd["icd"] < nonsense_icd["icd"] * 0.5:
            verdict = "STRONG_CONCEPT"  # Low ICD + high accuracy
            explanation = f"Concentrated concept (ICD {icd_ratio:.1f}x lower than noise) with {baseline['real_accuracy']:.0%} accuracy"
        else:
            verdict = "DIFFUSE_CONCEPT"  # High ICD but still separable
            explanation = f"Diffuse but separable concept ({baseline['real_accuracy']:.0%} accuracy, {baseline['signal_above_baseline']:.0%} above baseline)"
    else:
        if baseline["real_accuracy"] < 0.55:
            verdict = "NO_SIGNAL"
            explanation = f"No extractable signal ({baseline['real_accuracy']:.0%} accuracy ≈ chance)"
        else:
            verdict = "WEAK_SIGNAL"
            explanation = f"Weak signal ({baseline['real_accuracy']:.0%} accuracy, only {baseline['signal_above_baseline']:.0%} above baseline)"
    
    return {
        "benchmark": benchmark_name,
        "n_pairs": n_real,
        "real_icd": real_icd,
        "nonsense_icd": nonsense_icd,
        "icd_ratio": icd_ratio,
        "baseline_comparison": baseline,
        "verdict": verdict,
        "explanation": explanation,
    }


# Type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wisent.core.models.wisent_model import WisentModel
