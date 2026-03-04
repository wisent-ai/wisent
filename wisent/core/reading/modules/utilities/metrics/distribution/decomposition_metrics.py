"""Decomposition metrics for Zwiad Step 3: Decomposition Test."""

from typing import List, Tuple, Dict
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from wisent.core import constants as _C


def _adaptive_max_k(n_samples: int, *, decomp_adaptive_k_min: int, decomp_adaptive_k_max: int) -> int:
    """Adaptive max clusters: sqrt(n/half) clamped to range."""
    max_k = int(np.sqrt(n_samples / _C.N_COMPONENTS_2D))
    return max(decomp_adaptive_k_min, min(max_k, decomp_adaptive_k_max))


def _adaptive_min_cluster_size(n_samples: int, *, decomp_min_cluster_size_base: int, decomp_cluster_size_ratio: int) -> int:
    """Adaptive min samples per cluster."""
    return max(decomp_min_cluster_size_base, n_samples // decomp_cluster_size_ratio)


def find_optimal_clustering(
    diff_vectors: torch.Tensor,
    *,
    decomp_min_silhouette: float,
    decomp_pca_dims_max: int,
    decomp_max_silhouette_samples: int,
    decomp_adaptive_k_min: int,
    decomp_adaptive_k_max: int,
    decomp_min_cluster_size_base: int,
    decomp_cluster_size_ratio: int,
    decomp_kmeans_n_init_min: int,
    decomp_kmeans_n_init_max: int,
    decomp_kmeans_scaling_factor: int,
) -> Tuple[int, List[int], float]:
    """Find optimal number of clusters with adaptive parameters."""
    diff_np = diff_vectors.cpu().numpy() if isinstance(diff_vectors, torch.Tensor) else diff_vectors
    n_samples, n_features = diff_np.shape

    pca_dims = min(n_samples - _C.COMBO_OFFSET, n_features, decomp_pca_dims_max)
    if pca_dims < n_features and pca_dims >= _C.N_COMPONENTS_2D:
        _pca = PCA(n_components=pca_dims, random_state=_C.DEFAULT_RANDOM_SEED)
        diff_np = _pca.fit_transform(diff_np)

    if n_samples > decomp_max_silhouette_samples:
        _idx = np.random.RandomState(_C.DEFAULT_RANDOM_SEED).choice(
            n_samples, decomp_max_silhouette_samples, replace=False)
        diff_sub = diff_np[_idx]
        n_sub = decomp_max_silhouette_samples
    else:
        diff_sub = diff_np
        n_sub = n_samples

    max_k = _adaptive_max_k(
        n_sub, decomp_adaptive_k_min=decomp_adaptive_k_min, decomp_adaptive_k_max=decomp_adaptive_k_max)
    min_cluster_size = _adaptive_min_cluster_size(
        n_sub, decomp_min_cluster_size_base=decomp_min_cluster_size_base,
        decomp_cluster_size_ratio=decomp_cluster_size_ratio)
    max_k = min(max_k, n_sub // min_cluster_size)
    if max_k < _C.N_COMPONENTS_2D:
        return _C.COMBO_OFFSET, [_C.BINARY_CLASS_NEGATIVE] * n_samples, _C.SCORE_RANGE_MIN

    n_init = max(decomp_kmeans_n_init_min, min(
        decomp_kmeans_n_init_max, decomp_kmeans_scaling_factor // n_sub + _C.COMBO_OFFSET))
    best_k = _C.COMBO_OFFSET
    best_silhouette = -_C.SCORE_RANGE_MAX

    for k in range(_C.N_COMPONENTS_2D, max_k + _C.COMBO_OFFSET):
        km = KMeans(n_clusters=k, random_state=_C.DEFAULT_RANDOM_SEED, n_init=n_init)
        labels = km.fit_predict(diff_sub)
        cluster_sizes = np.bincount(labels)
        if len(cluster_sizes) < _C.N_COMPONENTS_2D or cluster_sizes.min() < min_cluster_size:
            continue
        sil = silhouette_score(diff_sub, labels)
        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k

    if best_silhouette < decomp_min_silhouette:
        return _C.COMBO_OFFSET, [_C.BINARY_CLASS_NEGATIVE] * n_samples, float(best_silhouette)

    # Refit on full data with best k to get labels for all samples
    final_km = KMeans(n_clusters=best_k, random_state=_C.DEFAULT_RANDOM_SEED, n_init=n_init)
    best_labels = final_km.fit_predict(diff_np).tolist()
    return best_k, best_labels, float(best_silhouette)


def geometry_per_concept(
    pos: torch.Tensor, neg: torch.Tensor, cluster_labels: List[int], n_concepts: int,
    *, decomp_min_concept_samples: int, decomp_threshold_base: float,
    decomp_threshold_min: float, decomp_threshold_max: float,
    cv_min_folds: int, cv_max_folds: int, cv_samples_per_fold: int,
    mlp_hidden_min: int, mlp_hidden_max: int,
    geometry_logistic_c_min: float, geometry_logistic_c_max: float,
) -> Dict[int, Dict[str, float]]:
    """Run geometry test (linear vs nonlinear) per concept cluster."""
    from .geometry_metrics import compute_linear_nonlinear_gap

    pos_np = pos.cpu().numpy() if isinstance(pos, torch.Tensor) else pos
    neg_np = neg.cpu().numpy() if isinstance(neg, torch.Tensor) else neg
    labels = np.array(cluster_labels)

    results = {}

    for concept_id in range(n_concepts):
        mask = labels == concept_id
        if mask.sum() < decomp_min_concept_samples:
            results[concept_id] = {
                "n_samples": int(mask.sum()),
                "linear_accuracy": None,
                "nonlinear_accuracy": None,
                "gap": None,
                "diagnosis": "INSUFFICIENT_SAMPLES",
            }
            continue

        pos_concept = torch.tensor(pos_np[mask])
        neg_concept = torch.tensor(neg_np[mask])

        linear_acc, nonlinear_acc = compute_linear_nonlinear_gap(
            pos_concept, neg_concept, cv_min_folds=cv_min_folds, cv_max_folds=cv_max_folds,
            cv_samples_per_fold=cv_samples_per_fold, mlp_hidden_min=mlp_hidden_min,
            mlp_hidden_max=mlp_hidden_max, geometry_logistic_c_min=geometry_logistic_c_min,
            geometry_logistic_c_max=geometry_logistic_c_max)
        gap = nonlinear_acc - linear_acc

        n_samples = int(mask.sum())
        adaptive_threshold = decomp_threshold_base / np.sqrt(n_samples)
        adaptive_threshold = max(decomp_threshold_min, min(adaptive_threshold, decomp_threshold_max))

        diagnosis = "NONLINEAR" if gap > adaptive_threshold else "LINEAR"

        results[concept_id] = {
            "n_samples": n_samples,
            "linear_accuracy": linear_acc,
            "nonlinear_accuracy": nonlinear_acc,
            "gap": gap,
            "gap_threshold": adaptive_threshold,
            "diagnosis": diagnosis,
        }

    return results


def compute_decomposition_metrics(
    pos: torch.Tensor, neg: torch.Tensor,
    *, decomp_min_silhouette: float, decomp_pca_dims_max: int,
    decomp_max_silhouette_samples: int, decomp_adaptive_k_min: int, decomp_adaptive_k_max: int,
    decomp_min_cluster_size_base: int, decomp_cluster_size_ratio: int,
    decomp_kmeans_n_init_min: int, decomp_kmeans_n_init_max: int, decomp_kmeans_scaling_factor: int,
    decomp_min_concept_samples: int, decomp_threshold_base: float,
    decomp_threshold_min: float, decomp_threshold_max: float,
    decomp_min_concept_samples_chow: int, decomp_cosine_sim_threshold: float,
    cv_min_folds: int, cv_max_folds: int, cv_samples_per_fold: int,
    mlp_hidden_min: int, mlp_hidden_max: int,
    geometry_logistic_c_min: float, geometry_logistic_c_max: float,
) -> Dict:
    """Compute decomposition metrics: clustering and per-concept geometry analysis."""
    diff_vectors = pos - neg
    n_concepts, cluster_labels, silhouette = find_optimal_clustering(
        diff_vectors, decomp_min_silhouette=decomp_min_silhouette,
        decomp_pca_dims_max=decomp_pca_dims_max,
        decomp_max_silhouette_samples=decomp_max_silhouette_samples,
        decomp_adaptive_k_min=decomp_adaptive_k_min, decomp_adaptive_k_max=decomp_adaptive_k_max,
        decomp_min_cluster_size_base=decomp_min_cluster_size_base,
        decomp_cluster_size_ratio=decomp_cluster_size_ratio,
        decomp_kmeans_n_init_min=decomp_kmeans_n_init_min,
        decomp_kmeans_n_init_max=decomp_kmeans_n_init_max,
        decomp_kmeans_scaling_factor=decomp_kmeans_scaling_factor)
    per_concept = geometry_per_concept(
        pos, neg, cluster_labels, n_concepts,
        decomp_min_concept_samples=decomp_min_concept_samples,
        decomp_threshold_base=decomp_threshold_base,
        decomp_threshold_min=decomp_threshold_min, decomp_threshold_max=decomp_threshold_max,
        cv_min_folds=cv_min_folds, cv_max_folds=cv_max_folds,
        cv_samples_per_fold=cv_samples_per_fold, mlp_hidden_min=mlp_hidden_min,
        mlp_hidden_max=mlp_hidden_max, geometry_logistic_c_min=geometry_logistic_c_min,
        geometry_logistic_c_max=geometry_logistic_c_max)
    chow = chow_test_analog(
        pos, neg, cluster_labels, n_concepts,
        decomp_min_concept_samples_chow=decomp_min_concept_samples_chow,
        decomp_cosine_sim_threshold=decomp_cosine_sim_threshold)
    return {
        "n_concepts": n_concepts,
        "silhouette_score": silhouette,
        "cluster_labels": cluster_labels,
        "per_concept_geometry": per_concept,
        "chow_test": chow,
    }


def chow_test_analog(
    pos: torch.Tensor, neg: torch.Tensor, cluster_labels: List[int], n_concepts: int,
    *, decomp_min_concept_samples_chow: int, decomp_cosine_sim_threshold: float,
) -> Dict[str, float]:
    """Chow test analog: do probe coefficients differ across concepts?"""
    from sklearn.linear_model import LogisticRegression

    pos_np = pos.cpu().numpy() if isinstance(pos, torch.Tensor) else pos
    neg_np = neg.cpu().numpy() if isinstance(neg, torch.Tensor) else neg
    labels = np.array(cluster_labels)

    coefficients = []

    for concept_id in range(n_concepts):
        mask = labels == concept_id
        if mask.sum() < decomp_min_concept_samples_chow:
            continue

        X = np.vstack([pos_np[mask], neg_np[mask]])
        y = np.array([1] * mask.sum() + [0] * mask.sum())

        model = LogisticRegression( solver="lbfgs", random_state=_C.DEFAULT_RANDOM_SEED)
        model.fit(X, y)

        coef = model.coef_[0]
        coef_norm = coef / (np.linalg.norm(coef) + _C.ZERO_THRESHOLD)
        coefficients.append(coef_norm)

    if len(coefficients) < 2:
        return {
            "n_concepts_tested": len(coefficients),
            "sufficient_concepts": False,
            "coefficients_differ": None,
        }

    cosine_similarities = []
    for i in range(len(coefficients)):
        for j in range(i + 1, len(coefficients)):
            cos_sim = np.dot(coefficients[i], coefficients[j])
            cosine_similarities.append(cos_sim)

    mean_cos = float(np.mean(cosine_similarities))
    min_cos = float(np.min(cosine_similarities))

    coefficients_differ = min_cos < decomp_cosine_sim_threshold

    return {
        "n_concepts_tested": len(coefficients),
        "sufficient_concepts": True,
        "mean_cosine_similarity": mean_cos,
        "min_cosine_similarity": min_cos,
        "coefficients_differ": coefficients_differ,
        "interpretation": "Directions differ by concept" if coefficients_differ else "Similar direction across concepts",
    }
