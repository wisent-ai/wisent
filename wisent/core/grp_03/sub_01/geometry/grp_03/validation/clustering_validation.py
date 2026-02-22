"""Clustering quality validation for concept analysis."""
import torch
import numpy as np
from typing import Dict, Any, Optional


def validate_clustering_quality(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_concepts: int = None,
    n_stability_runs: int = 5,
) -> Dict[str, Any]:
    """
    Validate clustering quality with multiple tests.

    Runs several validation checks to ensure clustering results are reliable:
    1. Stability test: Same data, different random seeds -> same clusters?
    2. Silhouette validation: Is the chosen k optimal?
    3. Cluster balance: Are clusters reasonably balanced?

    Args:
        pos_activations: Positive activations
        neg_activations: Negative activations
        n_concepts: Number of concepts (if None, uses auto-detection)
        n_stability_runs: Number of runs for stability test

    Returns:
        Dict with:
            - stability_score: 0-1, how consistent clustering is across runs
            - optimal_k_confidence: 0-1, confidence that k is correct
            - balance_score: 0-1, how balanced cluster sizes are
            - overall_quality: 0-1, combined quality score
            - warnings: List of potential issues
    """
    try:
        from sklearn.cluster import SpectralClustering
        from sklearn.metrics import adjusted_rand_score, silhouette_score

        n_pairs = min(len(pos_activations), len(neg_activations))
        diff_vectors = (pos_activations[:n_pairs] - neg_activations[:n_pairs]).float().cpu().numpy()

        # L2 normalize
        norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
        valid_mask = norms.squeeze() > 1e-8
        diff_normalized = diff_vectors[valid_mask] / norms[valid_mask]

        if len(diff_normalized) < 20:
            return {
                "stability_score": 0.0,
                "optimal_k_confidence": 0.0,
                "balance_score": 0.0,
                "overall_quality": 0.0,
                "warnings": ["Not enough valid samples for quality validation"],
            }

        # Auto-detect n_concepts if not provided
        if n_concepts is None:
            from .concept_analysis import detect_multiple_concepts
            detection = detect_multiple_concepts(pos_activations, neg_activations)
            n_concepts = detection.get("n_concepts", 2)

        n_neighbors = min(10, len(diff_normalized) - 1)
        warnings = []

        # 1. Stability test: run clustering multiple times with different seeds
        all_labels = []
        for seed in range(n_stability_runs):
            spectral = SpectralClustering(
                n_clusters=n_concepts, random_state=seed,
                affinity='nearest_neighbors', n_neighbors=n_neighbors
            )
            labels = spectral.fit_predict(diff_normalized)
            all_labels.append(labels)

        # Compute pairwise ARI between runs
        aris = []
        for i in range(n_stability_runs):
            for j in range(i + 1, n_stability_runs):
                ari = adjusted_rand_score(all_labels[i], all_labels[j])
                aris.append(ari)

        stability_score = float(np.mean(aris)) if aris else 0.0
        if stability_score < 0.8:
            warnings.append(f"Low clustering stability ({stability_score:.2f})")

        # 2. Optimal k confidence: compare silhouette scores for different k
        silhouette_scores = {}
        for k in range(2, min(6, len(diff_normalized) // 5)):
            spectral = SpectralClustering(
                n_clusters=k, random_state=42,
                affinity='nearest_neighbors', n_neighbors=n_neighbors
            )
            labels = spectral.fit_predict(diff_normalized)
            try:
                score = silhouette_score(diff_normalized, labels)
                silhouette_scores[k] = float(score)
            except:
                silhouette_scores[k] = 0.0

        if silhouette_scores:
            best_k = max(silhouette_scores, key=silhouette_scores.get)
            current_silhouette = silhouette_scores.get(n_concepts, 0.0)
            best_silhouette = silhouette_scores[best_k]

            if best_silhouette > 0:
                optimal_k_confidence = current_silhouette / best_silhouette
            else:
                optimal_k_confidence = 0.5

            if best_k != n_concepts:
                warnings.append(f"Better k might be {best_k} (silhouette: {best_silhouette:.3f} vs {current_silhouette:.3f})")
        else:
            optimal_k_confidence = 0.5

        # 3. Cluster balance: check if clusters are reasonably balanced
        final_labels = all_labels[0]  # Use first run
        cluster_sizes = np.bincount(final_labels)
        if len(cluster_sizes) > 1:
            min_size = cluster_sizes.min()
            max_size = cluster_sizes.max()
            balance_score = float(min_size / max_size) if max_size > 0 else 0.0

            if balance_score < 0.2:
                warnings.append(f"Highly imbalanced clusters (ratio: {balance_score:.2f})")
        else:
            balance_score = 1.0

        # Overall quality: weighted combination
        overall_quality = 0.4 * stability_score + 0.3 * optimal_k_confidence + 0.3 * balance_score

        return {
            "stability_score": stability_score,
            "optimal_k_confidence": optimal_k_confidence,
            "balance_score": balance_score,
            "overall_quality": overall_quality,
            "cluster_sizes": cluster_sizes.tolist(),
            "silhouette_scores": silhouette_scores,
            "warnings": warnings,
        }

    except Exception as e:
        return {
            "stability_score": 0.0,
            "optimal_k_confidence": 0.0,
            "balance_score": 0.0,
            "overall_quality": 0.0,
            "warnings": [f"Validation failed: {str(e)}"],
        }
