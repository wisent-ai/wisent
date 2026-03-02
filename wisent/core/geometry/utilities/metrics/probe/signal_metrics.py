"""Signal metrics for Zwiad Step 1: Signal Test."""

from typing import Dict, List
import torch
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from wisent.core.constants import DEFAULT_RANDOM_SEED, N_JOBS_SINGLE
from wisent.core import constants as _C


def _adaptive_n_permutations(n_samples: int) -> int:
    """Adaptive number of permutations: fewer for large datasets."""
    return max(_C.ADAPTIVE_PERMUTATIONS_MIN, min(_C.ADAPTIVE_PERMUTATIONS_MAX, _C.ADAPTIVE_PERMUTATIONS_DIVISOR // n_samples))


def _adaptive_k(n_samples: int) -> int:
    """Adaptive k for KNN: sqrt(n) clamped to [3, n//4]."""
    k = int(np.sqrt(n_samples))
    return max(_C.KNN_ADAPTIVE_K_MIN, min(k, n_samples // _C.KNN_ADAPTIVE_K_DIVISOR))


def _adaptive_cv(n_samples: int) -> int:
    """Adaptive CV folds: ensure at least 10 samples per fold."""
    return max(_C.ADAPTIVE_CV_MIN_FOLDS, min(_C.ADAPTIVE_CV_MAX_FOLDS, n_samples // _C.ADAPTIVE_CV_SAMPLES_PER_FOLD))


def _adaptive_pca_components(n_samples: int, n_features: int) -> int:
    """Adaptive PCA: min(n_samples/2, n_features, sample-based cap)."""
    max_possible = min(n_samples - 1, n_features)
    target = n_samples // 2
    return max(2, min(target, max_possible))


def _adaptive_manifold_components(n_samples: int) -> int:
    """Adaptive UMAP/PaCMAP components: log2(n_samples) clamped."""
    return max(2, min(int(np.log2(n_samples)), _C.ADAPTIVE_MANIFOLD_MAX_COMPONENTS))


def _adaptive_mlp_hidden(n_features: int) -> int:
    """Adaptive MLP hidden size: sqrt(n_features) clamped."""
    hidden = int(np.sqrt(n_features))
    return max(_C.ADAPTIVE_MLP_HIDDEN_MIN, min(hidden, _C.ADAPTIVE_MLP_HIDDEN_MAX))


def compute_signal_metrics(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    metric_keys: List[str],
) -> Dict[str, float]:
    """Compute geometry-agnostic signal metrics with adaptive parameters."""
    pos_np = pos_activations.cpu().numpy() if isinstance(pos_activations, torch.Tensor) else pos_activations
    neg_np = neg_activations.cpu().numpy() if isinstance(neg_activations, torch.Tensor) else neg_activations

    X = np.vstack([pos_np, neg_np])
    y = np.array([1] * len(pos_np) + [0] * len(neg_np))

    n_samples, n_features = X.shape
    k = _adaptive_k(n_samples)
    cv = _adaptive_cv(n_samples)
    pca_components = _adaptive_pca_components(n_samples, n_features)
    manifold_components = _adaptive_manifold_components(n_samples)
    mlp_hidden = _adaptive_mlp_hidden(n_features)

    results = {}

    if "knn_accuracy" in metric_keys:
        results["knn_accuracy"] = _knn_accuracy(X, y, k=k, cv=cv)
    if "knn_pca_accuracy" in metric_keys:
        results["knn_pca_accuracy"] = _knn_pca_accuracy(X, y, n_components=pca_components, k=k, cv=cv)
    if "mlp_probe_accuracy" in metric_keys:
        results["mlp_probe_accuracy"] = _mlp_accuracy(X, y, hidden=mlp_hidden, cv=cv)
    if "knn_umap_accuracy" in metric_keys:
        results["knn_umap_accuracy"] = _knn_umap_accuracy(X, y, n_components=manifold_components, k=k, cv=cv)
    if "knn_pacmap_accuracy" in metric_keys:
        results["knn_pacmap_accuracy"] = _knn_pacmap_accuracy(X, y, n_components=manifold_components, k=k, cv=cv)

    return results


def _knn_accuracy(X: np.ndarray, y: np.ndarray, k: int, cv: int) -> float:
    """KNN cross-validated accuracy."""
    clf = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return float(scores.mean())


def _knn_pca_accuracy(X: np.ndarray, y: np.ndarray, n_components: int, k: int, cv: int) -> float:
    """KNN accuracy after PCA dimensionality reduction."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    clf = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(clf, X_pca, y, cv=cv, scoring="accuracy")
    return float(scores.mean())


def _mlp_accuracy(X: np.ndarray, y: np.ndarray, hidden: int, cv: int) -> float:
    """MLP probe cross-validated accuracy."""
    # early_stopping requires internal validation split; needs >=2 samples per class
    # per fold. With n samples and cv folds, each fold has n*(cv-1)/cv training samples.
    # Internal split takes 10%, so need >=20 training samples per fold for safety.
    n_per_fold = len(X) * (cv - 1) // cv
    use_early_stopping = n_per_fold >= _C.MLP_EARLY_STOPPING_MIN_SAMPLES
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), random_state=DEFAULT_RANDOM_SEED, early_stopping=use_early_stopping, max_iter=_C.MLP_PROBE_MAX_ITER)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return float(scores.mean())


def _knn_umap_accuracy(X: np.ndarray, y: np.ndarray, n_components: int, k: int, cv: int) -> float:
    """KNN accuracy after UMAP dimensionality reduction."""
    try:
        import umap
        umap_neighbors = max(_C.UMAP_NEIGHBORS_MIN, min(_C.UMAP_NEIGHBORS_MAX, len(y) // _C.UMAP_NEIGHBORS_DIVISOR))
        reducer = umap.UMAP(n_components=n_components, n_neighbors=umap_neighbors, random_state=DEFAULT_RANDOM_SEED, n_jobs=N_JOBS_SINGLE)
        X_umap = reducer.fit_transform(X)
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, X_umap, y, cv=cv, scoring="accuracy")
        return float(scores.mean())
    except ImportError:
        return 0.0


def _knn_pacmap_accuracy(X: np.ndarray, y: np.ndarray, n_components: int, k: int, cv: int) -> float:
    """KNN accuracy after PaCMAP dimensionality reduction."""
    try:
        import pacmap
        pacmap_neighbors = max(_C.PACMAP_NEIGHBORS_MIN, min(_C.PACMAP_NEIGHBORS_MAX, len(y) // _C.PACMAP_NEIGHBORS_DIVISOR))
        reducer = pacmap.PaCMAP(n_components=n_components, n_neighbors=pacmap_neighbors, random_state=DEFAULT_RANDOM_SEED)
        X_pacmap = reducer.fit_transform(X)
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, X_pacmap, y, cv=cv, scoring="accuracy")
        return float(scores.mean())
    except ImportError:
        return 0.0
