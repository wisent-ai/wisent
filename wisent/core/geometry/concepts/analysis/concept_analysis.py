"""
Concept analysis for detecting and decomposing multiple concepts.

These functions analyze whether activations contain a single concept
or multiple interleaved concepts, and how to separate them.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


def detect_multiple_concepts(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    use_hdbscan: bool = False,
) -> Dict[str, Any]:
    """
    Detect if activations contain multiple concepts.

    Uses efficient coarse-to-fine search with K-means screening.
    SpectralClustering only runs on top candidate k values.
    """
    from .concept_detection import detect_with_hdbscan, detect_with_coarse_fine_search

    try:
        n_pairs = min(len(pos_activations), len(neg_activations))
        if n_pairs < 10:
            return {"n_concepts": 1, "silhouette_scores": {}}

        diff_vectors = (pos_activations[:n_pairs] - neg_activations[:n_pairs]).float().cpu().numpy()

        norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
        valid_mask = norms.squeeze() > 1e-8
        diff_normalized = diff_vectors[valid_mask] / norms[valid_mask]

        if len(diff_normalized) < 10:
            return {"n_concepts": 1, "silhouette_scores": {}}

        if use_hdbscan:
            return detect_with_hdbscan(diff_normalized)

        return detect_with_coarse_fine_search(diff_normalized)

    except Exception as e:
        return {"n_concepts": 1, "silhouette_scores": {}, "error": str(e)}


def split_by_concepts(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_concepts: int = 2,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Split activations into separate concepts using SpectralClustering."""
    try:
        from sklearn.cluster import SpectralClustering

        n_pairs = min(len(pos_activations), len(neg_activations))
        diff_vectors = (pos_activations[:n_pairs] - neg_activations[:n_pairs]).float().cpu().numpy()

        norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
        valid_mask = norms.squeeze() > 1e-8
        diff_normalized = diff_vectors[valid_mask] / norms[valid_mask]

        n_neighbors = min(10, len(diff_normalized) - 1)
        spectral = SpectralClustering(
            n_clusters=n_concepts, random_state=42,
            affinity='nearest_neighbors', n_neighbors=n_neighbors
        )
        labels = spectral.fit_predict(diff_normalized)

        valid_indices = np.where(valid_mask)[0]

        result = []
        for c in range(n_concepts):
            concept_mask = labels == c
            original_indices = valid_indices[concept_mask]

            if len(original_indices) > 0:
                pos_c = pos_activations[original_indices]
                neg_c = neg_activations[original_indices]
                result.append((pos_c, neg_c))

        return result
    except Exception:
        return [(pos_activations, neg_activations)]


def analyze_concept_independence(
    concepts: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, Any]:
    """Analyze how independent different concepts are (angle between directions)."""
    try:
        if len(concepts) < 2:
            return {"n_concepts": len(concepts), "independence": 1.0}

        directions = []
        for pos, neg in concepts:
            n = min(len(pos), len(neg))
            if n < 2:
                continue
            diff_mean = (pos[:n] - neg[:n]).float().cpu().numpy().mean(axis=0)
            norm = np.linalg.norm(diff_mean)
            if norm > 1e-8:
                directions.append(diff_mean / norm)

        if len(directions) < 2:
            return {"n_concepts": len(concepts), "independence": 1.0}

        angles = []
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                cos_angle = np.dot(directions[i], directions[j])
                angle = np.degrees(np.arccos(np.clip(np.abs(cos_angle), 0, 1)))
                angles.append(angle)

        mean_angle = float(np.mean(angles))
        independence = mean_angle / 90.0

        return {
            "n_concepts": len(concepts),
            "mean_angle": mean_angle,
            "independence": float(np.clip(independence, 0, 1)),
            "angles": angles,
        }
    except Exception:
        return {"n_concepts": len(concepts), "independence": 0.5}


def compute_concept_coherence(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> float:
    """Compute how coherent the concept is (1 = single direction, 0 = spread)."""
    try:
        n_pairs = min(len(pos_activations), len(neg_activations))
        if n_pairs < 3:
            return 0.0

        diff_vectors = (pos_activations[:n_pairs] - neg_activations[:n_pairs]).float().cpu().numpy()

        norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
        valid_mask = norms.squeeze() > 1e-8
        diff_normalized = diff_vectors[valid_mask] / norms[valid_mask]

        if len(diff_normalized) < 3:
            return 0.0

        U, S, Vh = np.linalg.svd(diff_normalized, full_matrices=False)

        if len(S) == 0 or S.sum() == 0:
            return 0.0

        coherence = float((S[0] ** 2) / (S ** 2).sum())
        return coherence
    except Exception:
        return 0.0


def compute_concept_stability(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_bootstrap: int = 20,
) -> float:
    """Compute stability of concept direction across bootstrap samples."""
    try:
        from ...metrics.direction.direction_metrics import compute_direction_stability

        result = compute_direction_stability(
            pos_activations, neg_activations, n_bootstrap=n_bootstrap
        )
        return result.get("stability_score", 0.0)
    except Exception:
        return 0.0


def decompose_into_concepts(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_concepts: int = None,
) -> Dict[str, Any]:
    """Full concept decomposition: detect, split, and analyze."""
    detection = detect_multiple_concepts(pos_activations, neg_activations)

    if n_concepts is None:
        n_concepts = detection.get("n_concepts", 1)

    if n_concepts == 1:
        return {
            "n_concepts": 1,
            "concepts": [(pos_activations, neg_activations)],
            "independence": 1.0,
            "coherence": compute_concept_coherence(pos_activations, neg_activations),
        }

    concepts = split_by_concepts(pos_activations, neg_activations, n_concepts)
    independence = analyze_concept_independence(concepts)

    coherences = [compute_concept_coherence(p, n) for p, n in concepts]

    return {
        "n_concepts": len(concepts),
        "concepts": concepts,
        "independence": independence.get("independence", 0.5),
        "coherences": coherences,
        "mean_coherence": float(np.mean(coherences)) if coherences else 0.0,
    }


# Re-export functions from concept_pairs for backwards compatibility
from .concept_pairs import (
    compute_concept_linear_separability,
    get_pair_concept_assignments,
    find_mixed_pairs,
    get_pure_concept_pairs,
    analyze_concept_structure,
    recommend_per_concept_steering,
)

# Clustering validation
from ...validation.clustering_validation import validate_clustering_quality
