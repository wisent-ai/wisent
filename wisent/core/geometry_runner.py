"""
Geometry search runner.

Runs geometry tests across the search space using cached activations.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import torch

import numpy as np

from wisent.core.geometry_search_space import GeometrySearchSpace, GeometrySearchConfig
from wisent.core.activations.extraction_strategy import ExtractionStrategy
from wisent.core.activations.activation_cache import (
    ActivationCache,
    CachedActivations,
    collect_and_cache_activations,
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
        
        # 6. Steerability score (weighted combination)
        # High alignment + low cluster angle + high norm ratio = good steerability
        steerability_score = (
            0.5 * max(0, diff_mean_alignment) +  # Main predictor
            0.2 * pct_positive_alignment +
            0.15 * min(1.0, steering_vector_norm_ratio) +
            0.15 * (1 - cluster_direction_angle / 180)  # Lower angle = better
        )
        steerability_score = float(np.clip(steerability_score, 0, 1))
        
        return {
            "diff_mean_alignment": diff_mean_alignment,
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
        
        # Step 2: Signal detection and classification (matching paper methodology)
        # Thresholds from paper: tau_exist = 0.6, tau_gap = 0.15
        tau_exist = 0.6
        tau_gap = 0.15
        
        # Use MAXIMUM of all nonlinear probes to detect signal
        # This addresses curse of dimensionality: raw k-NN may fail in high-d,
        # but k-NN on PCA/UMAP/PaCMAP features or MLP should still find nonlinear structure
        best_nonlinear = max(knn_k10, knn_pca, knn_umap, knn_pacmap, mlp_probe)
        
        # Signal exists if ANY nonlinear method can separate classes above chance
        has_signal = best_nonlinear >= tau_exist
        
        # Step 3: Determine if signal is linear or nonlinear
        # NO_SIGNAL: best_nonlinear < 0.6 (no separable signal by any method)
        # LINEAR: best_nonlinear >= 0.6 AND linear >= best_nonlinear - 0.15 (linear methods work)
        # NONLINEAR: best_nonlinear >= 0.6 AND linear < best_nonlinear - 0.15 (linear methods fail but nonlinear works)
        if not has_signal:
            is_linear = False
            recommendation = "NO_SIGNAL"
        elif linear_probe_accuracy >= best_nonlinear - tau_gap:
            is_linear = True
            recommendation = "CAA"  # Linear signal -> use Contrastive Activation Addition
        else:
            is_linear = False
            recommendation = "NONLINEAR"  # Nonlinear signal -> need different method (Truth Forest, NL-ITI, etc.)
        
        # Helper to safely get detail
        def get_detail(struct_name: str, key: str, default=0.0):
            if struct_name in result.all_scores:
                return result.all_scores[struct_name].details.get(key, default)
            return default
        
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
        """Get cached activations, extracting if necessary."""
        # Check cache
        if self.cache.has(self.model.model_name, benchmark, strategy):
            if show_progress:
                print(f"  Loading from cache...")
            return self.cache.get(self.model.model_name, benchmark, strategy)
        
        # Need to extract - load pairs first
        if show_progress:
            print(f"  Loading pairs...")
        
        pairs = self._load_pairs(benchmark)
        
        if show_progress:
            print(f"  Extracting activations for {len(pairs)} pairs...")
        
        return collect_and_cache_activations(
            model=self.model,
            pairs=pairs,
            benchmark=benchmark,
            strategy=strategy,
            cache=self.cache,
            show_progress=show_progress,
        )
    
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
