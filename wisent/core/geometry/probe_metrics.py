"""
Probe-based metrics for measuring signal separability in activation space.

These metrics use various classifiers (linear, MLP, k-NN) to measure
how separable positive and negative activations are.
"""

import torch
import numpy as np
from typing import Tuple


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
            return 0.5
        
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
            alpha=0.01,
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
        
        actual_components = min(n_components, len(X) - 1, X.shape[1])
        
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
    
    UMAP preserves nonlinear structure better than PCA.
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
        
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, X_umap, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except ImportError:
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
    
    PaCMAP preserves both local AND global structure better than UMAP.
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
        
        reducer = pacmap.PaCMAP(
            n_components=n_components,
            n_neighbors=min(10, len(X) // 4),
            MN_ratio=0.5,
            FP_ratio=2.0,
            random_state=42,
        )
        X_pacmap = reducer.fit_transform(X)
        
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, X_pacmap, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except ImportError:
        return 0.5
    except Exception:
        return 0.5
