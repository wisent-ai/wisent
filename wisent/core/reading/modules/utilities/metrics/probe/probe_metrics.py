"""Probe-based metrics for measuring signal separability in activation space."""
from __future__ import annotations
import torch
import numpy as np
from wisent.core.utils.config_tools.constants import (
    DEFAULT_RANDOM_SEED, VIZ_N_NEIGHBORS,
    VIZ_MIN_DIST, VIZ_N_NEIGHBORS_TRIMAP, VIZ_PCA_COMPONENTS,
    VIZ_N_COMPONENTS_2D, BINARY_CLASS_NEGATIVE, BINARY_CLASS_POSITIVE,
    N_COMPONENTS_2D,
)


def _prepare_data(pos_activations, neg_activations, min_per_class: int):
    """Shared data preparation for all probe metrics."""
    n_pos, n_neg = len(pos_activations), len(neg_activations)
    if n_pos < min_per_class or n_neg < min_per_class:
        return None, None, BINARY_CLASS_NEGATIVE, BINARY_CLASS_NEGATIVE
    X = torch.cat([pos_activations, neg_activations], dim=BINARY_CLASS_NEGATIVE).float().cpu().numpy()
    y = np.array([BINARY_CLASS_POSITIVE] * n_pos + [BINARY_CLASS_NEGATIVE] * n_neg)
    return X, y, n_pos, n_neg


def _cv_score(clf, X, y, n_folds, n_pos, n_neg, *, blend_default: float):
    """Run cross-validated scoring."""
    from sklearn.model_selection import cross_val_score
    folds = min(n_folds, min(n_pos, n_neg))
    if folds < N_COMPONENTS_2D:
        return blend_default
    scores = cross_val_score(clf, X, y, cv=folds, scoring='accuracy')
    return float(scores.mean())


def compute_signal_strength(
    pos_activations: torch.Tensor, neg_activations: torch.Tensor, n_folds: int,
    *, probe_min_per_class: int, probe_small_hidden: int,
    blend_default: float,
) -> float:
    """Compute signal strength via MLP cross-validation accuracy."""
    try:
        from sklearn.neural_network import MLPClassifier
        X, y, n_pos, n_neg = _prepare_data(pos_activations, neg_activations, probe_min_per_class)
        if X is None:
            return blend_default
        clf = MLPClassifier(hidden_layer_sizes=(probe_small_hidden,), random_state=DEFAULT_RANDOM_SEED)
        return _cv_score(clf, X, y, n_folds, n_pos, n_neg, blend_default=blend_default)
    except Exception:
        return blend_default


def compute_linear_probe_accuracy(
    pos_activations: torch.Tensor, neg_activations: torch.Tensor, n_folds: int,
    *, probe_min_per_class: int,
    blend_default: float,
) -> float:
    """Compute linear probe cross-validation accuracy."""
    try:
        from sklearn.linear_model import LogisticRegression
        X, y, n_pos, n_neg = _prepare_data(pos_activations, neg_activations, probe_min_per_class)
        if X is None:
            return blend_default
        clf = LogisticRegression(solver='lbfgs')
        return _cv_score(clf, X, y, n_folds, n_pos, n_neg, blend_default=blend_default)
    except Exception:
        return blend_default


def compute_mlp_probe_accuracy(
    pos_activations: torch.Tensor, neg_activations: torch.Tensor, n_folds: int,
    *,
    probe_min_per_class: int,
    probe_mlp_hidden: int,
    probe_mlp_alpha: float,
    probe_validation_fraction: float,
    blend_default: float,
) -> float:
    """Compute MLP probe cross-validation accuracy."""
    try:
        from sklearn.neural_network import MLPClassifier
        X, y, n_pos, n_neg = _prepare_data(pos_activations, neg_activations, probe_min_per_class)
        if X is None:
            return blend_default
        clf = MLPClassifier(
            hidden_layer_sizes=(probe_mlp_hidden,), early_stopping=True,
            validation_fraction=probe_validation_fraction,
            random_state=DEFAULT_RANDOM_SEED, alpha=probe_mlp_alpha,
        )
        return _cv_score(clf, X, y, n_folds, n_pos, n_neg, blend_default=blend_default)
    except Exception:
        return blend_default


def compute_knn_accuracy(
    pos_activations: torch.Tensor, neg_activations: torch.Tensor,
    k: int, n_folds: int,
    *, knn_min_class_offset: int,
    blend_default: float,
) -> float:
    """Compute k-NN cross-validation accuracy. Measures local separability."""
    try:
        from sklearn.neighbors import KNeighborsClassifier
        X, y, n_pos, n_neg = _prepare_data(
            pos_activations, neg_activations, min_per_class=k + knn_min_class_offset,
        )
        if X is None:
            return blend_default
        clf = KNeighborsClassifier(n_neighbors=k)
        return _cv_score(clf, X, y, n_folds, n_pos, n_neg, blend_default=blend_default)
    except Exception:
        return blend_default


def compute_knn_pca_accuracy(
    pos_activations: torch.Tensor, neg_activations: torch.Tensor,
    *,
    probe_knn_k: int,
    knn_min_class_offset: int,
    feature_dim_index: int,
    cv_folds: int,
    blend_default: float,
) -> float:
    """Compute k-NN accuracy on PCA-reduced features."""
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        X, y, n_pos, n_neg = _prepare_data(
            pos_activations, neg_activations, min_per_class=probe_knn_k + knn_min_class_offset,
        )
        if X is None:
            return blend_default
        actual_components = min(
            VIZ_PCA_COMPONENTS, len(X) - knn_min_class_offset, X.shape[feature_dim_index],
        )
        clf = Pipeline([
            ('pca', PCA(n_components=actual_components)),
            ('knn', KNeighborsClassifier(n_neighbors=probe_knn_k)),
        ])
        return _cv_score(clf, X, y, cv_folds, n_pos, n_neg, blend_default=blend_default)
    except Exception:
        return blend_default


def compute_knn_umap_accuracy(
    pos_activations: torch.Tensor, neg_activations: torch.Tensor,
    *,
    neighbors_viability_min: int,
    pacmap_neighbors_divisor: int,
    probe_knn_k: int,
    knn_min_class_offset: int,
    cv_folds: int,
    blend_default: float,
) -> float:
    """Compute k-NN accuracy on UMAP-reduced features."""
    try:
        import umap
        from sklearn.neighbors import KNeighborsClassifier
        X, y, n_pos, n_neg = _prepare_data(
            pos_activations, neg_activations, min_per_class=probe_knn_k + knn_min_class_offset,
        )
        if X is None:
            return blend_default
        umap_n_neighbors = min(VIZ_N_NEIGHBORS, len(X) // pacmap_neighbors_divisor)
        if umap_n_neighbors < neighbors_viability_min:
            return blend_default
        reducer = umap.UMAP(
            n_components=probe_knn_k, n_neighbors=umap_n_neighbors,
            min_dist=VIZ_MIN_DIST, random_state=DEFAULT_RANDOM_SEED,
        )
        X_reduced = reducer.fit_transform(X)
        clf = KNeighborsClassifier(n_neighbors=probe_knn_k)
        return _cv_score(clf, X_reduced, y, cv_folds, n_pos, n_neg, blend_default=blend_default)
    except ImportError:
        return blend_default
    except Exception:
        return blend_default


def compute_knn_pacmap_accuracy(
    pos_activations: torch.Tensor, neg_activations: torch.Tensor,
    *,
    neighbors_viability_min: int,
    pacmap_neighbors_divisor: int,
    probe_knn_k: int,
    knn_min_class_offset: int,
    cv_folds: int,
    blend_default: float,
) -> float:
    """Compute k-NN accuracy on PaCMAP-reduced features."""
    try:
        import pacmap
        from sklearn.neighbors import KNeighborsClassifier
        X, y, n_pos, n_neg = _prepare_data(
            pos_activations, neg_activations, min_per_class=probe_knn_k + knn_min_class_offset,
        )
        if X is None:
            return blend_default
        pac_neighbors = min(VIZ_N_NEIGHBORS_TRIMAP, len(X) // pacmap_neighbors_divisor)
        if pac_neighbors < neighbors_viability_min:
            return blend_default
        reducer = pacmap.PaCMAP(n_components=VIZ_N_COMPONENTS_2D, n_neighbors=pac_neighbors)
        X_reduced = reducer.fit_transform(X)
        clf = KNeighborsClassifier(n_neighbors=probe_knn_k)
        return _cv_score(clf, X_reduced, y, cv_folds, n_pos, n_neg, blend_default=blend_default)
    except ImportError:
        return blend_default
    except Exception:
        return blend_default
