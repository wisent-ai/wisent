"""
Concept pair assignment and utility functions.

Functions for assigning pairs to concepts and analyzing concept structure.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple


def compute_concept_linear_separability(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    cluster_labels: np.ndarray,
) -> Dict[str, Any]:
    """Check if concepts are linearly separable from each other."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        n_pairs = min(len(pos_activations), len(neg_activations))
        diff_vectors = (pos_activations[:n_pairs] - neg_activations[:n_pairs]).float().cpu().numpy()
        labels = cluster_labels[:n_pairs]

        unique_concepts = np.unique(labels)
        if len(unique_concepts) < 2:
            return {"separability": 1.0, "pairwise_separability": {}, "all_linearly_separable": True}

        pairwise_scores = {}
        for i in range(len(unique_concepts)):
            for j in range(i + 1, len(unique_concepts)):
                c_i, c_j = unique_concepts[i], unique_concepts[j]
                mask = (labels == c_i) | (labels == c_j)
                X = diff_vectors[mask]
                y = (labels[mask] == c_j).astype(int)

                if len(X) < 10 or len(np.unique(y)) < 2:
                    continue

                clf = LogisticRegression( solver='lbfgs')
                try:
                    n_cv = min(5, min(np.sum(y == 0), np.sum(y == 1)))
                    if n_cv >= 2:
                        scores = cross_val_score(clf, X, y, cv=n_cv, scoring='accuracy')
                        acc = float(np.mean(scores))
                    else:
                        clf.fit(X, y)
                        acc = float(clf.score(X, y))
                    pairwise_scores[(int(c_i), int(c_j))] = acc
                except Exception:
                    pairwise_scores[(int(c_i), int(c_j))] = 0.5

        if not pairwise_scores:
            return {"separability": 0.5, "pairwise_separability": {}, "all_linearly_separable": False}

        mean_separability = float(np.mean(list(pairwise_scores.values())))
        all_separable = all(s >= 0.95 for s in pairwise_scores.values())

        return {
            "separability": mean_separability,
            "pairwise_separability": {f"{k[0]}-{k[1]}": v for k, v in pairwise_scores.items()},
            "all_linearly_separable": all_separable,
        }
    except Exception:
        return {"separability": 0.5, "pairwise_separability": {}, "all_linearly_separable": False}


def get_pair_concept_assignments(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_concepts: int = 2,
) -> Dict[str, Any]:
    """Get concept assignment for each pair using KMeans."""
    try:
        from sklearn.cluster import KMeans

        n_pairs = min(len(pos_activations), len(neg_activations))
        diff_vectors = (pos_activations[:n_pairs] - neg_activations[:n_pairs]).float().cpu().numpy()

        norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
        valid_mask = norms.squeeze() > 1e-8
        diff_normalized = diff_vectors[valid_mask] / norms[valid_mask]

        if len(diff_normalized) < n_concepts * 2:
            return {"assignments": [0] * n_pairs, "confidences": [1.0] * n_pairs, "error": "not enough samples"}

        km = KMeans(n_clusters=n_concepts, random_state=42, n_init=10)
        labels = km.fit_predict(diff_normalized)
        distances = km.transform(diff_normalized)

        assigned_distances = distances[np.arange(len(labels)), labels]
        max_dist = distances.max() + 1e-8
        confidences_valid = 1 - (assigned_distances / max_dist)

        valid_indices = np.where(valid_mask)[0]
        assignments = [-1] * n_pairs
        confidences = [0.0] * n_pairs
        distances_to_clusters = [[float('inf')] * n_concepts for _ in range(n_pairs)]

        for i, orig_idx in enumerate(valid_indices):
            assignments[orig_idx] = int(labels[i])
            confidences[orig_idx] = float(confidences_valid[i])
            distances_to_clusters[orig_idx] = distances[i].tolist()

        pairs_by_concept = {c: [] for c in range(n_concepts)}
        for pair_idx, concept in enumerate(assignments):
            if concept >= 0:
                pairs_by_concept[concept].append(pair_idx)

        concept_steering_vectors = []
        for c in range(n_concepts):
            concept_indices = pairs_by_concept[c]
            if concept_indices:
                steering_vec = diff_vectors[concept_indices].mean(axis=0)
                concept_steering_vectors.append(steering_vec)
            else:
                concept_steering_vectors.append(np.zeros(diff_vectors.shape[1]))

        return {
            "assignments": assignments,
            "confidences": confidences,
            "distances_to_clusters": distances_to_clusters,
            "cluster_centers": km.cluster_centers_,
            "concept_steering_vectors": concept_steering_vectors,
            "pairs_by_concept": pairs_by_concept,
            "n_pairs": n_pairs,
            "n_valid_pairs": int(valid_mask.sum()),
            "n_invalid_pairs": int((~valid_mask).sum()),
        }
    except Exception as e:
        n = min(len(pos_activations), len(neg_activations))
        return {"assignments": [0] * n, "confidences": [0.0] * n, "error": str(e)}


def find_mixed_pairs(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    threshold: float = 0.3,
) -> List[int]:
    """Find pairs that don't fit well into any cluster."""
    try:
        from sklearn.cluster import KMeans

        n_pairs = min(len(pos_activations), len(neg_activations))
        diff_vectors = (pos_activations[:n_pairs] - neg_activations[:n_pairs]).float().cpu().numpy()

        norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
        valid_mask = norms.squeeze() > 1e-8
        diff_normalized = diff_vectors[valid_mask] / norms[valid_mask]

        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        km.fit(diff_normalized)

        distances = km.transform(diff_normalized).min(axis=1)
        median_dist = np.median(distances)

        mixed_mask = distances > median_dist * (1 + threshold)
        valid_indices = np.where(valid_mask)[0]

        return list(valid_indices[mixed_mask])
    except Exception:
        return []


def get_pure_concept_pairs(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    concept_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get pairs belonging to a specific concept cluster."""
    from .concept_analysis import split_by_concepts
    concepts = split_by_concepts(pos_activations, neg_activations, n_concepts=2)

    if concept_idx < len(concepts):
        return concepts[concept_idx]
    return pos_activations, neg_activations


def analyze_concept_structure(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, Any]:
    """Analyze concept structure - returns raw metrics for steering decisions."""
    from .concept_analysis import decompose_into_concepts, detect_multiple_concepts

    decomposition = decompose_into_concepts(pos_activations, neg_activations)
    n_concepts = decomposition["n_concepts"]
    independence = decomposition["independence"]
    mean_coherence = decomposition.get("mean_coherence", 0.0)
    coherences = decomposition.get("coherences", [])

    detection = detect_multiple_concepts(pos_activations, neg_activations)

    return {
        "n_concepts": n_concepts,
        "best_silhouette": detection.get("best_silhouette"),
        "silhouette_scores": detection.get("silhouette_scores", {}),
        "independence": independence,
        "coherences": coherences,
        "mean_coherence": mean_coherence,
        "concept_sizes": [len(c[0]) for c in decomposition.get("concepts", [])],
    }


def recommend_per_concept_steering(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, Any]:
    """Deprecated: Use analyze_concept_structure instead."""
    return analyze_concept_structure(pos_activations, neg_activations)
