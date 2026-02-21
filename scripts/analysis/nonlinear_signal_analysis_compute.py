"""
Compute functions for nonlinear signal analysis.

Contains Fisher per-dimension, Gini coefficient, kNN accuracy, MMD,
local intrinsic dimension, silhouette, and density ratio computations.
"""
import numpy as np
from typing import List
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import rbf_kernel


@dataclass
class NonlinearSignalResult:
    """Result of nonlinear signal analysis for one layer."""
    layer: int
    fisher_mean: float
    fisher_max: float
    fisher_gini: float
    fisher_top10_ratio: float
    num_dims_fisher_above_1: int
    knn_accuracy_k5: float
    knn_accuracy_k10: float
    knn_accuracy_k20: float
    mmd_rbf: float
    mmd_linear: float
    local_dim_pos: float
    local_dim_neg: float
    local_dim_ratio: float
    silhouette_score: float
    density_ratio: float


@dataclass
class BenchmarkNonlinearAnalysis:
    """Full nonlinear analysis for a benchmark."""
    benchmark: str
    model: str
    strategy: str
    num_pairs: int
    per_layer_results: List[NonlinearSignalResult]
    best_layer_knn: int
    best_knn_accuracy: float
    best_layer_mmd: int
    best_mmd: float


def compute_fisher_per_dimension(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Compute Fisher ratio for each dimension independently."""
    n_dims = pos.shape[1]
    fishers = np.zeros(n_dims)
    for d in range(n_dims):
        pos_d = pos[:, d]
        neg_d = neg[:, d]
        between_var = (pos_d.mean() - neg_d.mean()) ** 2
        within_var = (pos_d.var() + neg_d.var()) / 2
        if within_var > 1e-10:
            fishers[d] = between_var / within_var
    return fishers


def compute_gini(values: np.ndarray) -> float:
    """Compute Gini coefficient - measures concentration."""
    values = np.abs(values)
    if values.sum() < 1e-10:
        return 0.0
    values = np.sort(values)
    n = len(values)
    return (2 * np.sum((np.arange(1, n+1) * values)) / (n * values.sum())) - (n + 1) / n


def compute_knn_accuracy(pos: np.ndarray, neg: np.ndarray, k: int = 5) -> float:
    """Compute k-NN cross-validation accuracy."""
    X = np.vstack([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg))
    if len(X) < k + 1:
        return 0.5
    knn = KNeighborsClassifier(n_neighbors=k)
    try:
        scores = cross_val_score(knn, X, y, cv=min(5, len(X) // 2))
        return float(scores.mean())
    except:
        return 0.5


def compute_mmd_rbf(pos: np.ndarray, neg: np.ndarray, gamma: float = None) -> float:
    """Compute Maximum Mean Discrepancy with RBF kernel."""
    if gamma is None:
        all_data = np.vstack([pos, neg])
        dists = cdist(all_data, all_data, 'euclidean')
        gamma = 1.0 / (2 * np.median(dists[dists > 0]) ** 2 + 1e-10)
    K_pp = rbf_kernel(pos, pos, gamma=gamma)
    K_nn = rbf_kernel(neg, neg, gamma=gamma)
    K_pn = rbf_kernel(pos, neg, gamma=gamma)
    m, n = len(pos), len(neg)
    mmd = (K_pp.sum() / (m * m) + K_nn.sum() / (n * n) - 2 * K_pn.sum() / (m * n))
    return float(max(0, mmd))


def compute_mmd_linear(pos: np.ndarray, neg: np.ndarray) -> float:
    """Compute MMD with linear kernel (just mean difference)."""
    mean_diff = pos.mean(axis=0) - neg.mean(axis=0)
    return float(np.linalg.norm(mean_diff))


def estimate_local_intrinsic_dim(X: np.ndarray, k: int = 10) -> float:
    """Estimate local intrinsic dimensionality using MLE method."""
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
        if len(log_ratios) > 0:
            dim_est = -(k - 1) / log_ratios.sum() if log_ratios.sum() < 0 else X.shape[1]
            dims.append(min(dim_est, X.shape[1]))
    return float(np.median(dims)) if dims else float(X.shape[1])


def compute_silhouette(pos: np.ndarray, neg: np.ndarray) -> float:
    """Compute silhouette score for pos/neg clustering."""
    from sklearn.metrics import silhouette_score
    X = np.vstack([pos, neg])
    labels = np.array([1] * len(pos) + [0] * len(neg))
    if len(np.unique(labels)) < 2:
        return 0.0
    try:
        return float(silhouette_score(X, labels))
    except:
        return 0.0


def compute_density_ratio(pos: np.ndarray, neg: np.ndarray) -> float:
    """Compute ratio of average intra-class distances."""
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


def analyze_layer(pos: np.ndarray, neg: np.ndarray, layer: int) -> NonlinearSignalResult:
    """Analyze signal quality for one layer."""
    fishers = compute_fisher_per_dimension(pos, neg)
    sorted_fishers = np.sort(fishers)[::-1]
    top10_sum = sorted_fishers[:10].sum()
    total_sum = fishers.sum() + 1e-10

    return NonlinearSignalResult(
        layer=layer,
        fisher_mean=float(fishers.mean()),
        fisher_max=float(fishers.max()),
        fisher_gini=compute_gini(fishers),
        fisher_top10_ratio=float(top10_sum / total_sum),
        num_dims_fisher_above_1=int((fishers > 1.0).sum()),
        knn_accuracy_k5=compute_knn_accuracy(pos, neg, k=5),
        knn_accuracy_k10=compute_knn_accuracy(pos, neg, k=10),
        knn_accuracy_k20=compute_knn_accuracy(pos, neg, k=20),
        mmd_rbf=compute_mmd_rbf(pos, neg),
        mmd_linear=compute_mmd_linear(pos, neg),
        local_dim_pos=estimate_local_intrinsic_dim(pos),
        local_dim_neg=estimate_local_intrinsic_dim(neg),
        local_dim_ratio=estimate_local_intrinsic_dim(pos) / (estimate_local_intrinsic_dim(neg) + 1e-10),
        silhouette_score=compute_silhouette(pos, neg),
        density_ratio=compute_density_ratio(pos, neg),
    )
