"""
Efficient concept detection algorithms.

Optimized for speed with coarse-to-fine search and K-means screening.
Supports multi-layer concatenation for capturing concepts across all layers.
"""

import numpy as np
import torch
from typing import Dict, Any, Tuple


def detect_concepts_multilayer(
    activations_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    use_hdbscan: bool = False,
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
        norms = np.where(norms < 1e-8, 1.0, norms)
        diff_normalized = diff / norms

        all_layer_diffs.append(diff_normalized)

    # Concatenate: shape (n_pairs, hidden_dim * n_layers)
    concatenated = np.concatenate(all_layer_diffs, axis=1)

    # Now cluster on concatenated representation
    if use_hdbscan:
        result = detect_with_hdbscan(concatenated)
    else:
        result = detect_with_coarse_fine_search(concatenated)

    result["n_layers_used"] = len(layers)
    result["layers"] = layers
    return result


def detect_with_hdbscan(diff_normalized: np.ndarray) -> Dict[str, Any]:
    """Use HDBSCAN for automatic cluster count detection."""
    try:
        import hdbscan
        from sklearn.metrics import silhouette_score

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(5, len(diff_normalized) // 20),
            min_samples=3,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(diff_normalized)

        unique_labels = set(labels)
        n_clusters = len([l for l in unique_labels if l >= 0])

        if n_clusters < 2:
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
        return detect_with_coarse_fine_search(diff_normalized)
    except Exception as e:
        return {"n_concepts": 1, "silhouette_scores": {}, "error": str(e), "method": "hdbscan"}


def detect_with_coarse_fine_search(diff_normalized: np.ndarray) -> Dict[str, Any]:
    """Efficient coarse-to-fine search with K-means screening."""
    from sklearn.cluster import SpectralClustering, KMeans
    from sklearn.metrics import silhouette_score

    n_samples = len(diff_normalized)
    max_k = n_samples // 2
    silhouette_scores = {}

    # Phase 1: Coarse search with fast K-means
    coarse_k_values = [2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200]
    coarse_k_values = [k for k in coarse_k_values if k <= max_k]

    if not coarse_k_values:
        coarse_k_values = [2]

    kmeans_scores = {}
    for k in coarse_k_values:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=100)
            labels = kmeans.fit_predict(diff_normalized)
            score = silhouette_score(diff_normalized, labels)
            kmeans_scores[k] = float(score)
        except:
            continue

    if not kmeans_scores:
        return {"n_concepts": 1, "silhouette_scores": {}}

    # Find top 3 k values from K-means screening
    sorted_k = sorted(kmeans_scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates = [k for k, _ in sorted_k[:3]]

    # Phase 2: Refine around top candidates with SpectralClustering
    n_neighbors = min(10, n_samples - 1)
    refine_k_values = set()

    for k in top_candidates:
        for delta in range(-3, 4):
            candidate = k + delta
            if 2 <= candidate <= max_k:
                refine_k_values.add(candidate)

    refine_k_values = sorted(refine_k_values)

    consecutive_decreases = 0
    prev_score = None
    best_score = -1
    best_k = 2

    for k in refine_k_values:
        try:
            spectral = SpectralClustering(
                n_clusters=k, random_state=42,
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
                    if consecutive_decreases >= 3 and k > best_k:
                        break
                else:
                    consecutive_decreases = 0

            prev_score = score
        except:
            continue

    if not silhouette_scores:
        best_k = max(kmeans_scores, key=kmeans_scores.get)
        # Re-run K-means with best_k to get labels
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=3, max_iter=100)
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
        n_clusters=best_k, random_state=42,
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
