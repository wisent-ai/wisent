"""Individual geometry structure detectors."""

from __future__ import annotations

from typing import Dict

import torch

from .geometry_types import StructureType, StructureScore, GeometryAnalysisConfig
from wisent.core.utils.config_tools.constants import NORM_EPS, COMPARE_TOL, GEO_MANIFOLD_SCORE_DEFAULT, GEO_MANIFOLD_CONFIDENCE, GEO_ORTHOGONAL_CONFIDENCE, KMEANS_MAX_ITERATIONS_DEFAULT
from wisent.core import constants as _C


def detect_linear_structure(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    diff_vectors: torch.Tensor,
    cfg: GeometryAnalysisConfig,
) -> StructureScore:
    """Detect if a single linear direction captures the behavior."""
    if pos_tensor.shape[0] < 2 or neg_tensor.shape[0] < 2:
        return StructureScore(StructureType.LINEAR, 0.0, 0.0, {"reason": "insufficient_data"})

    try:
        mean_diff = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
        mean_diff_norm = mean_diff.norm()
        if mean_diff_norm < NORM_EPS:
            return StructureScore(StructureType.LINEAR, 0.0, 0.0, {"reason": "no_separation"})

        primary_dir = mean_diff / mean_diff_norm

        pos_proj = pos_tensor @ primary_dir
        neg_proj = neg_tensor @ primary_dir

        pos_mean, pos_std = pos_proj.mean(), pos_proj.std()
        neg_mean, neg_std = neg_proj.mean(), neg_proj.std()
        pooled_std = ((pos_std**2 + neg_std**2) / 2).sqrt()
        cohens_d = abs(pos_mean - neg_mean) / (pooled_std + NORM_EPS)

        pos_residual = pos_tensor - (pos_proj.unsqueeze(1) * primary_dir.unsqueeze(0))
        neg_residual = neg_tensor - (neg_proj.unsqueeze(1) * primary_dir.unsqueeze(0))

        total_var = pos_tensor.var() + neg_tensor.var()
        residual_var = pos_residual.var() + neg_residual.var()
        variance_explained = 1 - (residual_var / (total_var + NORM_EPS))
        variance_explained = max(0, min(1, float(variance_explained)))

        within_class_spread = (pos_std + neg_std) / 2
        between_class_dist = abs(pos_mean - neg_mean)
        spread_ratio = within_class_spread / (between_class_dist + NORM_EPS)
        consistency = max(0, 1 - spread_ratio)

        linear_score = (
            0.35 * min(float(cohens_d) / _C.DETECTOR_COHENS_D_DIVISOR, 1.0) +
            0.35 * variance_explained +
            0.30 * consistency
        )

        confidence = min(1.0, (pos_tensor.shape[0] + neg_tensor.shape[0]) / _C.DETECTOR_LARGE_SAMPLE_N)

        return StructureScore(
            StructureType.LINEAR, score=float(linear_score),
            confidence=float(confidence),
            details={"cohens_d": float(cohens_d), "variance_explained": variance_explained}
        )
    except Exception as e:
        return StructureScore(StructureType.LINEAR, 0.0, 0.0, {"error": str(e)})


def detect_cone_structure_score(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    cfg: GeometryAnalysisConfig,
) -> StructureScore:
    """Detect cone structure using raw cosine similarity of difference vectors."""
    try:
        n_pairs = min(pos_tensor.shape[0], neg_tensor.shape[0])
        if n_pairs < 3:
            return StructureScore(StructureType.CONE, 0.0, 0.0, {"reason": "insufficient_pairs"})

        diff_vectors = pos_tensor[:n_pairs] - neg_tensor[:n_pairs]
        norms = diff_vectors.norm(dim=1, keepdim=True)
        valid_mask = (norms.squeeze() > NORM_EPS)
        if valid_mask.sum() < 3:
            return StructureScore(StructureType.CONE, 0.0, 0.0, {"reason": "zero_differences"})

        diff_normalized = diff_vectors[valid_mask] / norms[valid_mask]
        cos_sim_matrix = diff_normalized @ diff_normalized.T

        n = cos_sim_matrix.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=cos_sim_matrix.device)
        off_diagonal = cos_sim_matrix[mask]

        mean_cos_sim = float(off_diagonal.mean())
        std_cos_sim = float(off_diagonal.std())

        if mean_cos_sim < 0:
            cone_score = 0.0
        elif mean_cos_sim < _C.DETECTOR_CONE_THRESHOLD_LOW:
            cone_score = mean_cos_sim
        elif mean_cos_sim < _C.DETECTOR_CONE_RANGE_MID:
            cone_score = (_C.DETECTOR_CONE_THRESHOLD_LOW
                          + _C.DETECTOR_CONE_SCALE_MID
                          * ((mean_cos_sim - _C.DETECTOR_CONE_THRESHOLD_LOW)
                             / _C.DETECTOR_CONE_SCALE_MID))
        elif mean_cos_sim < _C.DETECTOR_CONE_RANGE_HIGH:
            cone_score = (_C.DETECTOR_CONE_RANGE_MID
                          + (_C.DETECTOR_CONE_OFFSET_TOP
                             - _C.DETECTOR_CONE_RANGE_MID)
                          * ((mean_cos_sim - _C.DETECTOR_CONE_RANGE_MID)
                             / _C.DETECTOR_CONE_SCALE_HIGH))
        else:
            _top_span = 1.0 - _C.DETECTOR_CONE_OFFSET_TOP
            _top_frac = (mean_cos_sim - _C.DETECTOR_CONE_RANGE_TOP) / _C.DETECTOR_CONE_SCALE_TOP
            cone_score = _C.DETECTOR_CONE_OFFSET_TOP + _top_span * _top_frac

        consistency = max(0, 1 - std_cos_sim)
        confidence = consistency * min(1.0, n_pairs / _C.DETECTOR_SMALL_SAMPLE_N)

        return StructureScore(
            StructureType.CONE, score=float(cone_score),
            confidence=float(confidence),
            details={"raw_mean_cosine_similarity": mean_cos_sim}
        )
    except Exception as e:
        return StructureScore(StructureType.CONE, 0.0, 0.0, {"error": str(e)})


def detect_cluster_structure(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    diff_vectors: torch.Tensor,
    cfg: GeometryAnalysisConfig,
) -> StructureScore:
    """Detect if activations form discrete clusters."""
    all_activations = torch.cat([pos_tensor, neg_tensor], dim=0)
    n_samples = all_activations.shape[0]

    if n_samples < 6:
        return StructureScore(StructureType.CLUSTER, 0.0, 0.0, {"reason": "insufficient_data"})

    best_silhouette = -1.0
    best_k = _C.MIN_CLUSTERS

    for k in range(_C.MIN_CLUSTERS, min(cfg.max_clusters + 1, n_samples // 2)):
        try:
            labels, centroids, silhouette = _kmeans_with_silhouette(all_activations, k)
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
        except Exception:
            continue

    if best_silhouette < 0:
        return StructureScore(StructureType.CLUSTER, 0.0, 0.0, {"reason": "clustering_failed"})

    cluster_score = max(0, min(1, (best_silhouette + 1) / 2))
    confidence = min(1.0, n_samples / _C.DETECTOR_CLUSTER_SAMPLE_N)

    return StructureScore(
        StructureType.CLUSTER, score=float(cluster_score),
        confidence=float(confidence),
        details={"best_k": best_k, "silhouette": float(best_silhouette)}
    )


def detect_sparse_structure(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    diff_vectors: torch.Tensor,
    cfg: GeometryAnalysisConfig,
) -> StructureScore:
    """Detect if behavior is encoded in sparse activations."""
    try:
        mean_diff = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
        abs_diff = mean_diff.abs()
        threshold = abs_diff.max() * cfg.sparse_threshold
        active_dims = (abs_diff > threshold).sum().item()
        sparsity = 1 - (active_dims / mean_diff.shape[0])
        sparse_score = float(sparsity)
        confidence = _C.SPARSE_DETECTION_CONFIDENCE
        return StructureScore(
            StructureType.SPARSE, score=sparse_score,
            confidence=confidence,
            details={"active_dims": active_dims, "sparsity": sparsity}
        )
    except Exception as e:
        return StructureScore(StructureType.SPARSE, 0.0, 0.0, {"error": str(e)})


def detect_manifold_structure(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    diff_vectors: torch.Tensor,
    cfg: GeometryAnalysisConfig,
) -> StructureScore:
    """Detect non-linear manifold structure (fallback)."""
    # Manifold is the most general - use moderate score as fallback
    return StructureScore(
        StructureType.MANIFOLD, score=GEO_MANIFOLD_SCORE_DEFAULT, confidence=GEO_MANIFOLD_CONFIDENCE,
        details={"note": "Manifold is fallback structure"}
    )


def detect_bimodal_structure(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    diff_vectors: torch.Tensor,
    cfg: GeometryAnalysisConfig,
) -> StructureScore:
    """Detect bimodal distribution in projections."""
    try:
        mean_diff = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
        mean_diff = mean_diff / (mean_diff.norm() + NORM_EPS)
        all_activations = torch.cat([pos_tensor, neg_tensor], dim=0)
        projections = all_activations @ mean_diff
        # Simple bimodality check: gap between pos and neg means
        pos_proj = pos_tensor @ mean_diff
        neg_proj = neg_tensor @ mean_diff
        gap = abs(pos_proj.mean() - neg_proj.mean())
        std = projections.std()
        bimodal_score = min(1.0, gap / (2 * std + NORM_EPS))
        return StructureScore(
            StructureType.BIMODAL, score=float(bimodal_score),
            confidence=_C.BIMODAL_DETECTION_CONFIDENCE, details={"gap_over_std": float(gap / (std + NORM_EPS))}
        )
    except Exception as e:
        return StructureScore(StructureType.BIMODAL, 0.0, 0.0, {"error": str(e)})


def detect_orthogonal_structure(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    diff_vectors: torch.Tensor,
    cfg: GeometryAnalysisConfig,
) -> StructureScore:
    """Detect orthogonal subspaces."""
    try:
        n_pairs = min(pos_tensor.shape[0], neg_tensor.shape[0])
        if n_pairs < 3:
            return StructureScore(StructureType.ORTHOGONAL, 0.0, 0.0, {})
        diff = pos_tensor[:n_pairs] - neg_tensor[:n_pairs]
        norms = diff.norm(dim=1, keepdim=True)
        valid = norms.squeeze() > NORM_EPS
        if valid.sum() < 3:
            return StructureScore(StructureType.ORTHOGONAL, 0.0, 0.0, {})
        diff_norm = diff[valid] / norms[valid]
        cos_sim = diff_norm @ diff_norm.T
        n = cos_sim.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool)
        mean_abs_cos = cos_sim[mask].abs().mean().item()
        orthogonal_score = max(0, 1 - mean_abs_cos / cfg.orthogonal_threshold)
        return StructureScore(
            StructureType.ORTHOGONAL, score=float(orthogonal_score),
            confidence=GEO_ORTHOGONAL_CONFIDENCE, details={"mean_abs_cosine": mean_abs_cos}
        )
    except Exception as e:
        return StructureScore(StructureType.ORTHOGONAL, 0.0, 0.0, {"error": str(e)})


def _kmeans_with_silhouette(data: torch.Tensor, k: int, max_iters: int = KMEANS_MAX_ITERATIONS_DEFAULT):
    """Simple k-means with silhouette score."""
    n = data.shape[0]
    idx = torch.randperm(n)[:k]
    centroids = data[idx].clone()

    for _ in range(max_iters):
        dists = torch.cdist(data, centroids)
        labels = dists.argmin(dim=1)
        new_centroids = torch.stack([
            data[labels == i].mean(dim=0) if (labels == i).sum() > 0 else centroids[i]
            for i in range(k)
        ])
        if torch.allclose(centroids, new_centroids, atol=COMPARE_TOL):
            break
        centroids = new_centroids

    # Compute silhouette
    dists = torch.cdist(data, centroids)
    labels = dists.argmin(dim=1)
    silhouettes = []
    for i in range(n):
        cluster = labels[i]
        same_cluster = data[labels == cluster]
        if len(same_cluster) > 1:
            a = (data[i] - same_cluster).norm(dim=1).sum() / (len(same_cluster) - 1)
        else:
            a = 0.0
        b = float('inf')
        for c in range(k):
            if c != cluster:
                other = data[labels == c]
                if len(other) > 0:
                    b = min(b, (data[i] - other).norm(dim=1).mean().item())
        if b == float('inf'):
            b = 0.0
        s = (b - a) / (max(a, b) + NORM_EPS)
        silhouettes.append(s)
    return labels, centroids, sum(silhouettes) / len(silhouettes) if silhouettes else -1.0
