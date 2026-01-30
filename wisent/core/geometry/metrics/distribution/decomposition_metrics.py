"""Decomposition metrics for RepScan Step 3: Decomposition Test."""

from typing import List, Tuple, Dict
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _adaptive_max_k(n_samples: int) -> int:
    """Adaptive max clusters: sqrt(n/2) clamped to [2, 15]."""
    max_k = int(np.sqrt(n_samples / 2))
    return max(2, min(max_k, 15))


def _adaptive_min_cluster_size(n_samples: int) -> int:
    """Adaptive min samples per cluster: max(5, n/20)."""
    return max(5, n_samples // 20)


def find_optimal_clustering(
    diff_vectors: torch.Tensor,
    min_silhouette: float = 0.3,
) -> Tuple[int, List[int], float]:
    """
    Find optimal number of clusters with adaptive parameters.

    Uses silhouette score to determine optimal k.
    Only reports multiple concepts if silhouette exceeds min_silhouette threshold.

    Args:
        diff_vectors: Difference vectors (pos - neg) to cluster
        min_silhouette: Minimum silhouette score to accept multiple clusters (default 0.3)

    Returns:
        Tuple of (n_concepts, cluster_labels, best_silhouette)
    """
    diff_np = diff_vectors.cpu().numpy() if isinstance(diff_vectors, torch.Tensor) else diff_vectors

    n_samples, n_features = diff_np.shape
    max_k = _adaptive_max_k(n_samples)
    min_cluster_size = _adaptive_min_cluster_size(n_samples)

    # Adjust max_k based on min_cluster_size constraint
    max_k = min(max_k, n_samples // min_cluster_size)

    if max_k < 2:
        return 1, [0] * n_samples, 0.0

    # Adaptive n_init based on sample size (more samples = fewer inits needed)
    n_init = max(3, min(10, 1000 // n_samples + 1))

    best_k = 1
    best_silhouette = -1.0
    best_labels = [0] * n_samples

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init)
        labels = kmeans.fit_predict(diff_np)

        # Check if any cluster is too small
        cluster_sizes = np.bincount(labels)
        if cluster_sizes.min() < min_cluster_size:
            continue

        sil = silhouette_score(diff_np, labels)
        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k
            best_labels = labels.tolist()

    # Only accept multiple clusters if silhouette meets threshold
    if best_silhouette < min_silhouette:
        return 1, [0] * n_samples, float(best_silhouette)

    return best_k, best_labels, float(best_silhouette)


def geometry_per_concept(
    pos: torch.Tensor,
    neg: torch.Tensor,
    cluster_labels: List[int],
    n_concepts: int,
) -> Dict[int, Dict[str, float]]:
    """
    Run geometry test (linear vs nonlinear) per concept cluster.

    This addresses: "Does the direction differ for violent vs sexual vs illegal?"
    Each concept may have different geometry (linear vs nonlinear encoding).

    Returns:
        Dict mapping concept_id -> {linear_acc, nonlinear_acc, gap, diagnosis}
    """
    from .geometry_metrics import compute_linear_nonlinear_gap

    pos_np = pos.cpu().numpy() if isinstance(pos, torch.Tensor) else pos
    neg_np = neg.cpu().numpy() if isinstance(neg, torch.Tensor) else neg
    labels = np.array(cluster_labels)

    results = {}

    for concept_id in range(n_concepts):
        mask = labels == concept_id
        if mask.sum() < 10:
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

        linear_acc, nonlinear_acc = compute_linear_nonlinear_gap(pos_concept, neg_concept)
        gap = nonlinear_acc - linear_acc

        n_samples = int(mask.sum())
        adaptive_threshold = 0.5 / np.sqrt(n_samples)
        adaptive_threshold = max(0.02, min(adaptive_threshold, 0.1))

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


def chow_test_analog(
    pos: torch.Tensor,
    neg: torch.Tensor,
    cluster_labels: List[int],
    n_concepts: int,
) -> Dict[str, float]:
    """
    Chow test analog: do probe coefficients differ significantly across concepts?

    Fits linear probe per concept and tests if coefficients are significantly different.
    If yes, the "direction" differs by content type.

    Returns:
        Dict with coefficient_variance, f_statistic_analog, diagnosis.
    """
    from sklearn.linear_model import LogisticRegression

    pos_np = pos.cpu().numpy() if isinstance(pos, torch.Tensor) else pos
    neg_np = neg.cpu().numpy() if isinstance(neg, torch.Tensor) else neg
    labels = np.array(cluster_labels)

    coefficients = []

    for concept_id in range(n_concepts):
        mask = labels == concept_id
        if mask.sum() < 20:
            continue

        X = np.vstack([pos_np[mask], neg_np[mask]])
        y = np.array([1] * mask.sum() + [0] * mask.sum())

        model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
        model.fit(X, y)

        coef = model.coef_[0]
        coef_norm = coef / (np.linalg.norm(coef) + 1e-10)
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

    coefficients_differ = min_cos < 0.8

    return {
        "n_concepts_tested": len(coefficients),
        "sufficient_concepts": True,
        "mean_cosine_similarity": mean_cos,
        "min_cosine_similarity": min_cos,
        "coefficients_differ": coefficients_differ,
        "interpretation": "Directions differ by concept" if coefficients_differ else "Similar direction across concepts",
    }
