"""Signal metrics for Zwiad Step 1: Signal Test."""

from typing import Dict, List
import torch
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from wisent.core.utils.config_tools.constants import DEFAULT_RANDOM_SEED, N_JOBS_SINGLE


def _adaptive_n_permutations(n_samples: int, perm_min: int, perm_max: int, perm_divisor: int) -> int:
    """Adaptive number of permutations: fewer for large datasets."""
    return max(perm_min, min(perm_max, perm_divisor // n_samples))


def _adaptive_k(n_samples: int, k_min: int, k_divisor: int) -> int:
    """Adaptive k for KNN: sqrt(n) clamped to [k_min, n//k_divisor]."""
    k = int(np.sqrt(n_samples))
    return max(k_min, min(k, n_samples // k_divisor))


def _adaptive_cv(n_samples: int, cv_min: int, cv_max: int, samples_per_fold: int) -> int:
    """Adaptive CV folds: ensure at least samples_per_fold per fold."""
    return max(cv_min, min(cv_max, n_samples // samples_per_fold))


def _adaptive_pca_components(n_samples: int, n_features: int) -> int:
    """Adaptive PCA: min(n_samples/2, n_features, sample-based cap)."""
    max_possible = min(n_samples - 1, n_features)
    target = n_samples // 2
    return max(2, min(target, max_possible))


def _adaptive_manifold_components(n_samples: int, max_components: int) -> int:
    """Adaptive UMAP/PaCMAP components: log2(n_samples) clamped."""
    return max(2, min(int(np.log2(n_samples)), max_components))


def _adaptive_mlp_hidden(n_features: int, hidden_min: int, hidden_max: int) -> int:
    """Adaptive MLP hidden size: sqrt(n_features) clamped."""
    hidden = int(np.sqrt(n_features))
    return max(hidden_min, min(hidden, hidden_max))


def compute_signal_metrics(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    metric_keys: List[str],
    mlp_early_stopping_min_samples: int = None,
    mlp_probe_max_iter: int = None,
    adaptive_config: dict = None,
) -> Dict[str, float]:
    """Compute geometry-agnostic signal metrics with adaptive parameters."""
    if adaptive_config is None:
        raise ValueError("adaptive_config is required")
    ac = adaptive_config
    pos_np = pos_activations.cpu().numpy() if isinstance(pos_activations, torch.Tensor) else pos_activations
    neg_np = neg_activations.cpu().numpy() if isinstance(neg_activations, torch.Tensor) else neg_activations

    X = np.vstack([pos_np, neg_np])
    y = np.array([1] * len(pos_np) + [0] * len(neg_np))

    n_samples, n_features = X.shape
    k = _adaptive_k(n_samples, k_min=ac["knn_k_min"], k_divisor=ac["knn_k_divisor"])
    cv = _adaptive_cv(n_samples, cv_min=ac["cv_min_folds"], cv_max=ac["cv_max_folds"], samples_per_fold=ac["cv_samples_per_fold"])
    pca_components = _adaptive_pca_components(n_samples, n_features)
    manifold_components = _adaptive_manifold_components(n_samples, max_components=ac["manifold_max_components"])
    mlp_hidden = _adaptive_mlp_hidden(n_features, hidden_min=ac["mlp_hidden_min"], hidden_max=ac["mlp_hidden_max"])

    results = {}

    if "knn_accuracy" in metric_keys:
        results["knn_accuracy"] = _knn_accuracy(X, y, k=k, cv=cv)
    if "knn_pca_accuracy" in metric_keys:
        results["knn_pca_accuracy"] = _knn_pca_accuracy(X, y, n_components=pca_components, k=k, cv=cv)
    if "mlp_probe_accuracy" in metric_keys:
        results["mlp_probe_accuracy"] = _mlp_accuracy(
            X, y, hidden=mlp_hidden, cv=cv,
            mlp_early_stopping_min_samples=mlp_early_stopping_min_samples,
            mlp_probe_max_iter=mlp_probe_max_iter)
    if "knn_umap_accuracy" in metric_keys:
        results["knn_umap_accuracy"] = _knn_umap_accuracy(X, y, n_components=manifold_components, k=k, cv=cv, umap_min=ac["umap_neighbors_min"], umap_max=ac["umap_neighbors_max"], umap_divisor=ac["umap_neighbors_divisor"])
    if "knn_pacmap_accuracy" in metric_keys:
        results["knn_pacmap_accuracy"] = _knn_pacmap_accuracy(X, y, n_components=manifold_components, k=k, cv=cv, pac_min=ac["pacmap_neighbors_min"], pac_max=ac["pacmap_neighbors_max"], pac_divisor=ac["pacmap_neighbors_divisor"])

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


def _mlp_accuracy(
    X: np.ndarray, y: np.ndarray, hidden: int, cv: int,
    mlp_early_stopping_min_samples: int, mlp_probe_max_iter: int,
) -> float:
    """MLP probe cross-validated accuracy."""
    # early_stopping requires internal validation split; needs >=2 samples per class
    # per fold. With n samples and cv folds, each fold has n*(cv-1)/cv training samples.
    # Internal split takes 10%, so need >=20 training samples per fold for safety.
    n_per_fold = len(X) * (cv - 1) // cv
    use_early_stopping = n_per_fold >= mlp_early_stopping_min_samples
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), random_state=DEFAULT_RANDOM_SEED, early_stopping=use_early_stopping, max_iter=mlp_probe_max_iter)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return float(scores.mean())


def _knn_umap_accuracy(X: np.ndarray, y: np.ndarray, n_components: int, k: int, cv: int, umap_min: int = None, umap_max: int = None, umap_divisor: int = None) -> float:
    """KNN accuracy after UMAP dimensionality reduction."""
    try:
        import umap
        umap_neighbors = max(umap_min, min(umap_max, len(y) // umap_divisor))
        reducer = umap.UMAP(n_components=n_components, n_neighbors=umap_neighbors, random_state=DEFAULT_RANDOM_SEED, n_jobs=N_JOBS_SINGLE)
        X_umap = reducer.fit_transform(X)
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, X_umap, y, cv=cv, scoring="accuracy")
        return float(scores.mean())
    except ImportError:
        return 0.0


def _knn_pacmap_accuracy(X: np.ndarray, y: np.ndarray, n_components: int, k: int, cv: int, pac_min: int = None, pac_max: int = None, pac_divisor: int = None) -> float:
    """KNN accuracy after PaCMAP dimensionality reduction."""
    try:
        import pacmap
        pacmap_neighbors = max(pac_min, min(pac_max, len(y) // pac_divisor))
        reducer = pacmap.PaCMAP(n_components=n_components, n_neighbors=pacmap_neighbors, random_state=DEFAULT_RANDOM_SEED)
        X_pacmap = reducer.fit_transform(X)
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, X_pacmap, y, cv=cv, scoring="accuracy")
        return float(scores.mean())
    except ImportError:
        return 0.0
