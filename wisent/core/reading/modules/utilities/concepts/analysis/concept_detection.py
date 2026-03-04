"""
Efficient concept detection algorithms.

Optimized for speed with coarse-to-fine search and K-means screening.
Supports multi-layer concatenation for capturing concepts across all layers.
"""

import numpy as np
import torch
from typing import Dict, Any, Tuple
from wisent.core import constants as _C


def detect_concepts_multilayer(
    activations_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    min_clusters: int,
    *,
    hdbscan_min_cluster_floor: int,
    hdbscan_adaptive_divisor: int,
    hdbscan_min_samples: int,
    kmeans_n_init_small: int,
    spectral_n_neighbors: int,
    elbow_consecutive_decreases: int,
    concept_detection_coarse_k: tuple,
    cumulative_variance_top_n: int,
    use_hdbscan: bool = True,
) -> Dict[str, Any]:
    """
    Detect concepts using concatenated diff vectors from ALL layers.

    This captures concepts that may be clear at different layers,
    rather than clustering at a single layer and potentially missing patterns.

    Args:
        activations_by_layer: Dict mapping layer -> (pos_activations, neg_activations)
        use_hdbscan: Whether to use HDBSCAN for automatic cluster count

    Returns:
        Dict with n_concepts, cluster_labels, silhouette_scores, method
    """
    layers = sorted(activations_by_layer.keys())
    if not layers:
        return {"n_concepts": 1, "silhouette_scores": {}, "cluster_labels": []}

    # Get number of pairs from first layer
    first_pos, first_neg = activations_by_layer[layers[0]]
    n_pairs = min(len(first_pos), len(first_neg))

    if n_pairs < 10:
        return {"n_concepts": 1, "silhouette_scores": {}, "cluster_labels": list(range(n_pairs))}

    # Build concatenated diff vectors: for each pair, concat L2-normed diffs from all layers
    all_layer_diffs = []

    for layer in layers:
        pos, neg = activations_by_layer[layer]
        diff = (pos[:n_pairs] - neg[:n_pairs]).float().cpu().numpy()

        # L2 normalize this layer's diff vectors
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms = np.where(norms < _C.NORM_EPS, 1.0, norms)
        diff_normalized = diff / norms

        all_layer_diffs.append(diff_normalized)

    # Concatenate: shape (n_pairs, hidden_dim * n_layers)
    concatenated = np.concatenate(all_layer_diffs, axis=1)

    # Now cluster on concatenated representation
    if use_hdbscan:
        result = detect_with_hdbscan(concatenated, min_clusters=min_clusters, hdbscan_min_cluster_floor=hdbscan_min_cluster_floor, hdbscan_adaptive_divisor=hdbscan_adaptive_divisor, hdbscan_min_samples=hdbscan_min_samples, kmeans_n_init_small=kmeans_n_init_small, spectral_n_neighbors=spectral_n_neighbors, elbow_consecutive_decreases=elbow_consecutive_decreases, concept_detection_coarse_k=concept_detection_coarse_k, cumulative_variance_top_n=cumulative_variance_top_n)
    else:
        result = detect_with_coarse_fine_search(concatenated, min_clusters=min_clusters, concept_detection_coarse_k=concept_detection_coarse_k, kmeans_n_init_small=kmeans_n_init_small, spectral_n_neighbors=spectral_n_neighbors, elbow_consecutive_decreases=elbow_consecutive_decreases, cumulative_variance_top_n=cumulative_variance_top_n)

    result["n_layers_used"] = len(layers)
    result["layers"] = layers
    return result


def detect_with_hdbscan(diff_normalized: np.ndarray, min_clusters: int, *, hdbscan_min_cluster_floor: int, hdbscan_adaptive_divisor: int, hdbscan_min_samples: int, kmeans_n_init_small: int = None, spectral_n_neighbors: int = None, elbow_consecutive_decreases: int = None, concept_detection_coarse_k: tuple = None, cumulative_variance_top_n: int = None) -> Dict[str, Any]:
    """Use HDBSCAN for automatic cluster count detection."""
    try:
        import hdbscan
        from sklearn.metrics import silhouette_score

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(hdbscan_min_cluster_floor, len(diff_normalized) // hdbscan_adaptive_divisor),
            min_samples=hdbscan_min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(diff_normalized)

        unique_labels = set(labels)
        n_clusters = len([l for l in unique_labels if l >= 0])

        if n_clusters < min_clusters:
            return {"n_concepts": 1, "silhouette_scores": {}, "method": "hdbscan"}

        valid_mask = labels >= 0
        if valid_mask.sum() < 10:
            return {"n_concepts": 1, "silhouette_scores": {}, "method": "hdbscan"}

        score = silhouette_score(diff_normalized[valid_mask], labels[valid_mask])

        return {
            "best_k": n_clusters,
            "best_silhouette": float(score),
            "silhouette_scores": {n_clusters: float(score)},
            "n_concepts": n_clusters,
            "cluster_labels": labels.tolist(),
            "method": "hdbscan",
            "noise_ratio": float((labels == -1).sum() / len(labels)),
        }
    except ImportError:
        return detect_with_coarse_fine_search(diff_normalized, min_clusters=min_clusters, concept_detection_coarse_k=concept_detection_coarse_k, kmeans_n_init_small=kmeans_n_init_small, spectral_n_neighbors=spectral_n_neighbors, elbow_consecutive_decreases=elbow_consecutive_decreases, cumulative_variance_top_n=cumulative_variance_top_n)
    except Exception as e:
        return {"n_concepts": 1, "silhouette_scores": {}, "error": str(e), "method": "hdbscan"}


def detect_with_coarse_fine_search(diff_normalized: np.ndarray, min_clusters: int, concept_detection_coarse_k: tuple, *, kmeans_n_init_small: int, spectral_n_neighbors: int, elbow_consecutive_decreases: int, cumulative_variance_top_n: int) -> Dict[str, Any]:
    """Efficient coarse-to-fine search with K-means screening."""
    from sklearn.cluster import SpectralClustering, KMeans
    from sklearn.metrics import silhouette_score

    n_samples = len(diff_normalized)
    max_k = n_samples // 2
    silhouette_scores = {}

    # Phase 1: Coarse search with fast K-means
    coarse_k_values = list(concept_detection_coarse_k)
    coarse_k_values = [k for k in coarse_k_values if k <= max_k]

    if not coarse_k_values:
        coarse_k_values = [min_clusters]

    kmeans_scores = {}
    for k in coarse_k_values:
        try:
            kmeans = KMeans(n_clusters=k, random_state=_C.DEFAULT_RANDOM_SEED, n_init=kmeans_n_init_small, )
            labels = kmeans.fit_predict(diff_normalized)
            score = silhouette_score(diff_normalized, labels)
            kmeans_scores[k] = float(score)
        except:
            continue

    if not kmeans_scores:
        return {"n_concepts": 1, "silhouette_scores": {}}

    # Find top 3 k values from K-means screening
    sorted_k = sorted(kmeans_scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates = [k for k, _ in sorted_k[:cumulative_variance_top_n]]

    # Phase 2: Refine around top candidates with SpectralClustering
    n_neighbors = min(spectral_n_neighbors, n_samples - 1)
    refine_k_values = set()

    for k in top_candidates:
        for delta in range(-3, 4):
            candidate = k + delta
            if min_clusters <= candidate <= max_k:
                refine_k_values.add(candidate)

    refine_k_values = sorted(refine_k_values)

    consecutive_decreases = 0
    prev_score = None
    best_score = -1
    best_k = min_clusters

    for k in refine_k_values:
        try:
            spectral = SpectralClustering(
                n_clusters=k, random_state=_C.DEFAULT_RANDOM_SEED,
                affinity='nearest_neighbors', n_neighbors=n_neighbors
            )
            labels = spectral.fit_predict(diff_normalized)
            score = silhouette_score(diff_normalized, labels)
            silhouette_scores[k] = float(score)

            if score > best_score:
                best_score = score
                best_k = k

            if prev_score is not None:
                if score < prev_score:
                    consecutive_decreases += 1
                    if consecutive_decreases >= elbow_consecutive_decreases and k > best_k:
                        break
                else:
                    consecutive_decreases = 0

            prev_score = score
        except:
            continue

    if not silhouette_scores:
        best_k = max(kmeans_scores, key=kmeans_scores.get)
        # Re-run K-means with best_k to get labels
        kmeans = KMeans(n_clusters=best_k, random_state=_C.DEFAULT_RANDOM_SEED, n_init=kmeans_n_init_small, )
        best_labels = kmeans.fit_predict(diff_normalized)
        return {
            "best_k": best_k,
            "best_silhouette": kmeans_scores[best_k],
            "silhouette_scores": kmeans_scores,
            "n_concepts": best_k,
            "cluster_labels": best_labels.tolist(),
            "method": "kmeans_fallback",
        }

    # Re-run SpectralClustering with best_k to get final labels
    spectral = SpectralClustering(
        n_clusters=best_k, random_state=_C.DEFAULT_RANDOM_SEED,
        affinity='nearest_neighbors', n_neighbors=n_neighbors
    )
    best_labels = spectral.fit_predict(diff_normalized)

    return {
        "best_k": best_k,
        "best_silhouette": best_score,
        "silhouette_scores": silhouette_scores,
        "kmeans_screening": kmeans_scores,
        "n_concepts": best_k,
        "cluster_labels": best_labels.tolist(),
        "method": "coarse_fine",
    }
