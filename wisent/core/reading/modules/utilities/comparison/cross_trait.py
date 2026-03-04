"""Cross-trait representation comparison metrics.

Answers: "How does the representation of coding differ from hallucination?"

Given multiple steering objects (one per trait), computes:
- Pairwise cosine similarity matrix (per-layer and aggregated)
- Shared subspace via SVD across all objects
- Per-object uniqueness (residual after projecting onto shared subspace)
- Hierarchical clustering on the distance matrix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from wisent.core.utils.config_tools.constants import LOG_EPS


@dataclass
class ComparisonResult:
    """Result of comparing multiple steering objects across traits."""

    labels: List[str]
    similarity_matrix: List[List[float]]
    per_layer_similarity: Dict[int, List[List[float]]]
    clusters: List[int]
    uniqueness_scores: Dict[str, float]
    shared_variance_explained: float
    most_similar_pair: Tuple[str, str, float]
    most_different_pair: Tuple[str, str, float]


def _extract_vectors_from_object(obj) -> Dict[int, torch.Tensor]:
    """Extract per-layer vectors from any steering object type."""
    if hasattr(obj, 'vectors') and isinstance(obj.vectors, dict):
        return {int(l): v.float().clone() for l, v in obj.vectors.items()}
    if hasattr(obj, 'displacements') and isinstance(obj.displacements, dict):
        return {int(l): d.float().mean(dim=0) for l, d in obj.displacements.items()}
    vectors = {}
    if hasattr(obj, 'metadata') and hasattr(obj.metadata, 'layers'):
        for layer in obj.metadata.layers:
            try:
                vectors[int(layer)] = obj.get_steering_vector(layer).float().clone()
            except (KeyError, Exception):
                pass
    return vectors


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two 1D tensors."""
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def _compute_pairwise_cosine(
    vectors_list: List[Dict[int, torch.Tensor]],
    layers: List[int],
) -> Tuple[List[List[float]], Dict[int, List[List[float]]]]:
    """Compute pairwise cosine similarity, per-layer and aggregated."""
    n = len(vectors_list)
    per_layer: Dict[int, List[List[float]]] = {}

    for layer in layers:
        layer_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                vi = vectors_list[i].get(layer)
                vj = vectors_list[j].get(layer)
                if vi is not None and vj is not None:
                    layer_matrix[i][j] = _cosine_similarity(vi, vj)
                elif i == j:
                    layer_matrix[i][j] = 1.0
        per_layer[layer] = layer_matrix

    # Aggregate: mean across layers
    agg = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            vals = [per_layer[l][i][j] for l in layers]
            agg[i][j] = sum(vals) / max(len(vals), 1)

    return agg, per_layer


def compute_shared_subspace(
    all_vectors: List[Dict[int, torch.Tensor]],
    layers: List[int],
    variance_threshold: float,
) -> Tuple[Dict[int, torch.Tensor], float]:
    """Find shared subspace across all objects per layer via SVD.

    Returns (subspace_basis_per_layer, mean_variance_explained).
    """
    total_explained = 0.0
    count = 0
    subspace: Dict[int, torch.Tensor] = {}

    for layer in layers:
        vecs = [obj[layer] for obj in all_vectors if layer in obj]
        if len(vecs) < 2:
            continue
        stacked = torch.stack(vecs, dim=0)  # (n_objects, hidden_dim)
        U, S, Vh = torch.linalg.svd(stacked, full_matrices=False)
        total_var = (S ** 2).sum()
        if total_var < LOG_EPS:
            continue
        cumvar = torch.cumsum(S ** 2, dim=0) / total_var
        k = int((cumvar >= variance_threshold).float().argmax().item()) + 1
        k = max(1, min(k, len(S)))
        subspace[layer] = Vh[:k]
        explained = cumvar[k - 1].item()
        total_explained += explained
        count += 1

    mean_explained = total_explained / max(count, 1)
    return subspace, mean_explained


def compute_uniqueness_scores(
    all_vectors: List[Dict[int, torch.Tensor]],
    labels: List[str],
    subspace: Dict[int, torch.Tensor],
    layers: List[int],
) -> Dict[str, float]:
    """Compute per-object uniqueness as residual norm after shared projection."""
    scores: Dict[str, float] = {}
    for idx, label in enumerate(labels):
        total_residual = 0.0
        total_norm = 0.0
        for layer in layers:
            vec = all_vectors[idx].get(layer)
            basis = subspace.get(layer)
            if vec is None:
                continue
            vec_norm = vec.norm().item()
            total_norm += vec_norm ** 2
            if basis is not None:
                coeffs = vec @ basis.T
                projection = coeffs @ basis
                residual = (vec - projection).norm().item()
            else:
                residual = vec_norm
            total_residual += residual ** 2
        if total_norm > 0:
            scores[label] = (total_residual ** 0.5) / (total_norm ** 0.5)
        else:
            scores[label] = 0.0
    return scores


def cluster_traits(
    similarity_matrix: List[List[float]],
    min_clusters: int,
    n_clusters: Optional[int] = None,
    *,
    max_silhouette_clusters: int,
) -> List[int]:
    """Agglomerative clustering on distance matrix derived from cosine similarities.

    Auto-detects optimal cluster count via silhouette score if n_clusters is None.
    """
    n = len(similarity_matrix)
    if n <= 2:
        return list(range(n))

    # Convert similarity to distance
    dist = [[max(0.0, 1.0 - abs(similarity_matrix[i][j]))
             for j in range(n)] for i in range(n)]

    try:
        import numpy as np
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        dist_arr = np.array(dist)
        np.fill_diagonal(dist_arr, 0)
        condensed = squareform(dist_arr, checks=False)
        Z = linkage(condensed, method='average')

        if n_clusters is not None:
            return fcluster(Z, t=n_clusters, criterion='maxclust').tolist()

        # Auto-detect k via silhouette
        from sklearn.metrics import silhouette_score
        best_k, best_sil = 2, -1.0
        for k in range(min_clusters, min(n, max_silhouette_clusters)):
            labels = fcluster(Z, t=k, criterion='maxclust')
            if len(set(labels)) < 2:
                continue
            sil = silhouette_score(dist_arr, labels, metric='precomputed')
            if sil > best_sil:
                best_sil = sil
                best_k = k
        return fcluster(Z, t=best_k, criterion='maxclust').tolist()
    except ImportError:
        return list(range(n))


def _find_extreme_pairs(
    labels: List[str],
    matrix: List[List[float]],
) -> Tuple[Tuple[str, str, float], Tuple[str, str, float]]:
    """Find most similar and most different pairs from similarity matrix."""
    n = len(labels)
    most_sim = ("", "", -2.0)
    most_diff = ("", "", 2.0)
    for i in range(n):
        for j in range(i + 1, n):
            val = matrix[i][j]
            if val > most_sim[2]:
                most_sim = (labels[i], labels[j], val)
            if val < most_diff[2]:
                most_diff = (labels[i], labels[j], val)
    return most_sim, most_diff


def compare_steering_objects(
    objects: list,
    labels: List[str],
    min_clusters: int,
    layers: Optional[List[int]] = None,
    variance_threshold: float = None,
) -> ComparisonResult:
    """Compare multiple steering objects across traits.

    Args:
        objects: List of loaded steering objects (any type).
        labels: Trait label for each object.
        layers: Optional layer filter (default: all common layers).
        variance_threshold: For shared subspace SVD.

    Returns:
        ComparisonResult with full comparison metrics.
    """
    if variance_threshold is None:
        raise ValueError("variance_threshold is required")
    # Extract vectors from all objects
    all_vectors = [_extract_vectors_from_object(obj) for obj in objects]

    # Find common layers
    if layers is None:
        common = set()
        for vecs in all_vectors:
            if not common:
                common = set(vecs.keys())
            else:
                common &= set(vecs.keys())
        layers = sorted(common)

    if not layers:
        raise ValueError("No common layers found across steering objects")

    # Pairwise cosine similarity
    agg_matrix, per_layer = _compute_pairwise_cosine(all_vectors, layers)

    # Shared subspace
    subspace, shared_var = compute_shared_subspace(
        all_vectors, layers, variance_threshold,
    )

    # Uniqueness
    uniqueness = compute_uniqueness_scores(all_vectors, labels, subspace, layers)

    # Clustering
    clusters = cluster_traits(agg_matrix, min_clusters=min_clusters)

    # Extreme pairs
    most_sim, most_diff = _find_extreme_pairs(labels, agg_matrix)

    return ComparisonResult(
        labels=labels,
        similarity_matrix=agg_matrix,
        per_layer_similarity=per_layer,
        clusters=clusters,
        uniqueness_scores=uniqueness,
        shared_variance_explained=shared_var,
        most_similar_pair=most_sim,
        most_different_pair=most_diff,
    )
