"""Probe-based metrics for measuring signal separability in activation space."""
from __future__ import annotations
import torch
import numpy as np
from wisent.core.utils.config_tools.constants import (
    PROBE_MIN_PER_CLASS, PROBE_SMALL_HIDDEN, PROBE_MLP_HIDDEN,
    PROBE_MLP_ALPHA, PROBE_VALIDATION_FRACTION, CV_FOLDS,
    DEFAULT_RANDOM_SEED, BLEND_DEFAULT, VIZ_N_NEIGHBORS,
    VIZ_MIN_DIST, VIZ_N_NEIGHBORS_TRIMAP, VIZ_PCA_COMPONENTS,
    VIZ_N_COMPONENTS_2D, PROBE_KNN_K,
)


def _prepare_data(pos_activations, neg_activations, min_per_class=PROBE_MIN_PER_CLASS):
    """Shared data preparation for all probe metrics."""
    n_pos, n_neg = len(pos_activations), len(neg_activations)
    if n_pos < min_per_class or n_neg < min_per_class:
        return None, None, 0, 0
    X = torch.cat([pos_activations, neg_activations], dim=0).float().cpu().numpy()
    y = np.array([1] * n_pos + [0] * n_neg)
    return X, y, n_pos, n_neg


def _cv_score(clf, X, y, n_folds, n_pos, n_neg):
    """Run cross-validated scoring."""
    from sklearn.model_selection import cross_val_score
    folds = min(n_folds, min(n_pos, n_neg))
    if folds < 2:
        return BLEND_DEFAULT
    scores = cross_val_score(clf, X, y, cv=folds, scoring='accuracy')
    return float(scores.mean())


def compute_signal_strength(
    pos_activations: torch.Tensor, neg_activations: torch.Tensor, n_folds: int = CV_FOLDS,
) -> float:
    """Compute signal strength using MLP cross-validation accuracy.
    Measures whether there is ANY extractable signal (linear or nonlinear).
    Returns BLEND_DEFAULT for no signal, >0.7 for signal exists."""
    try:
        from sklearn.neural_network import MLPClassifier
        X, y, n_pos, n_neg = _prepare_data(pos_activations, neg_activations)
        if X is None:
            return BLEND_DEFAULT
        clf = MLPClassifier(hidden_layer_sizes=(PROBE_SMALL_HIDDEN,), random_state=DEFAULT_RANDOM_SEED)
        return _cv_score(clf, X, y, n_folds, n_pos, n_neg)
    except Exception:
        return BLEND_DEFAULT


def compute_linear_probe_accuracy(
    pos_activations: torch.Tensor, neg_activations: torch.Tensor, n_folds: int = CV_FOLDS,
) -> float:
    """Compute linear probe cross-validation accuracy.
    If signal_strength is high but linear_probe is low, the signal is nonlinear."""
    try:
        from sklearn.linear_model import LogisticRegression
        X, y, n_pos, n_neg = _prepare_data(pos_activations, neg_activations)
        if X is None:
            return BLEND_DEFAULT
        clf = LogisticRegression(solver='lbfgs')
        return _cv_score(clf, X, y, n_folds, n_pos, n_neg)
    except Exception:
        return BLEND_DEFAULT


def compute_mlp_probe_accuracy(
    pos_activations: torch.Tensor, neg_activations: torch.Tensor,
    hidden_size: int = PROBE_MLP_HIDDEN, n_folds: int = CV_FOLDS,
) -> float:
    """Compute MLP probe cross-validation accuracy.
    Provides a nonlinear baseline more robust than k-NN in high dimensions."""
    try:
        from sklearn.neural_network import MLPClassifier
        X, y, n_pos, n_neg = _prepare_data(pos_activations, neg_activations)
        if X is None:
            return BLEND_DEFAULT
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden_size,), early_stopping=True,
            validation_fraction=PROBE_VALIDATION_FRACTION,
            random_state=DEFAULT_RANDOM_SEED, alpha=PROBE_MLP_ALPHA,
        )
        return _cv_score(clf, X, y, n_folds, n_pos, n_neg)
    except Exception:
        return BLEND_DEFAULT


def compute_knn_accuracy(
    pos_activations: torch.Tensor, neg_activations: torch.Tensor,
    k: int = PROBE_KNN_K, n_folds: int = CV_FOLDS,
) -> float:
    """Compute k-NN cross-validation accuracy. Measures local separability."""
    try:
        from sklearn.neighbors import KNeighborsClassifier
        X, y, n_pos, n_neg = _prepare_data(pos_activations, neg_activations, min_per_class=k + 1)
        if X is None:
            return BLEND_DEFAULT
        clf = KNeighborsClassifier(n_neighbors=k)
        return _cv_score(clf, X, y, n_folds, n_pos, n_neg)
    except Exception:
        return BLEND_DEFAULT


def compute_knn_pca_accuracy(
    pos_activations: torch.Tensor, neg_activations: torch.Tensor,
    k: int = PROBE_KNN_K, n_components: int = VIZ_PCA_COMPONENTS, n_folds: int = CV_FOLDS,
) -> float:
    """Compute k-NN accuracy on PCA-reduced features."""
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        X, y, n_pos, n_neg = _prepare_data(pos_activations, neg_activations, min_per_class=k + 1)
        if X is None:
            return BLEND_DEFAULT
        actual_components = min(n_components, len(X) - 1, X.shape[1])
        clf = Pipeline([
            ('pca', PCA(n_components=actual_components)),
            ('knn', KNeighborsClassifier(n_neighbors=k)),
        ])
        return _cv_score(clf, X, y, n_folds, n_pos, n_neg)
    except Exception:
        return BLEND_DEFAULT


def compute_knn_umap_accuracy(
    pos_activations: torch.Tensor, neg_activations: torch.Tensor,
    k: int = PROBE_KNN_K, n_components: int = PROBE_KNN_K, n_folds: int = CV_FOLDS,
) -> float:
    """Compute k-NN accuracy on UMAP-reduced features."""
    try:
        import umap
        from sklearn.neighbors import KNeighborsClassifier
        X, y, n_pos, n_neg = _prepare_data(pos_activations, neg_activations, min_per_class=k + 1)
        if X is None:
            return BLEND_DEFAULT
        umap_n_neighbors = min(VIZ_N_NEIGHBORS, len(X) // 4)
        if umap_n_neighbors < 2:
            return BLEND_DEFAULT
        reducer = umap.UMAP(
            n_components=n_components, n_neighbors=umap_n_neighbors,
            min_dist=VIZ_MIN_DIST, random_state=DEFAULT_RANDOM_SEED,
        )
        X_reduced = reducer.fit_transform(X)
        from sklearn.neighbors import KNeighborsClassifier as KNC
        clf = KNC(n_neighbors=k)
        return _cv_score(clf, X_reduced, y, n_folds, n_pos, n_neg)
    except ImportError:
        return BLEND_DEFAULT
    except Exception:
        return BLEND_DEFAULT


def compute_knn_pacmap_accuracy(
    pos_activations: torch.Tensor, neg_activations: torch.Tensor,
    k: int = PROBE_KNN_K, n_components: int = VIZ_N_COMPONENTS_2D, n_folds: int = CV_FOLDS,
) -> float:
    """Compute k-NN accuracy on PaCMAP-reduced features."""
    try:
        import pacmap
        from sklearn.neighbors import KNeighborsClassifier
        X, y, n_pos, n_neg = _prepare_data(pos_activations, neg_activations, min_per_class=k + 1)
        if X is None:
            return BLEND_DEFAULT
        pac_neighbors = min(VIZ_N_NEIGHBORS_TRIMAP, len(X) // 4)
        if pac_neighbors < 2:
            return BLEND_DEFAULT
        reducer = pacmap.PaCMAP(n_components=n_components, n_neighbors=pac_neighbors)
        X_reduced = reducer.fit_transform(X)
        clf = KNeighborsClassifier(n_neighbors=k)
        return _cv_score(clf, X_reduced, y, n_folds, n_pos, n_neg)
    except ImportError:
        return BLEND_DEFAULT
    except Exception:
        return BLEND_DEFAULT
