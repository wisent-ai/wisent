"""
Single-cloud activation space metrics.

Functions for analyzing individual activation point clouds:
shape, cone fit, sphere fit, manifold dimension, clustering, density, topology.

Extracted from activation_structure.py to keep files under 300 lines.
"""

import torch
import numpy as np
from typing import Dict, Any
from scipy.spatial.distance import pdist, squareform
from wisent.core import constants as _C


def compute_cloud_shape(activations: torch.Tensor) -> Dict[str, Any]:
    """Characterize the shape of a single activation point cloud."""
    X = activations.float().cpu().numpy()
    n, d = X.shape
    if n < 3:
        return {"error": "need at least 3 points"}
    norms = np.linalg.norm(X, axis=1)
    centroid = X.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    X_centered = X - centroid
    cov = np.cov(X_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    total_var = eigenvalues.sum()
    if total_var > 0:
        cumsum = np.cumsum(eigenvalues) / total_var
        dim_50 = int(np.searchsorted(cumsum, _C.CLOUD_VAR_PERCENTILE_50) + 1)
        dim_90 = int(np.searchsorted(cumsum, _C.CLOUD_VAR_PERCENTILE_90) + 1)
        dim_99 = int(np.searchsorted(cumsum, _C.CLOUD_VAR_PERCENTILE_99) + 1)
        participation_ratio = (total_var ** 2) / (np.sum(eigenvalues ** 2) + _C.NORM_EPS)
    else:
        dim_50, dim_90, dim_99 = 1, 1, 1
        participation_ratio = 1
    if eigenvalues[0] > 0:
        sphericity = eigenvalues[-1] / eigenvalues[0]
    else:
        sphericity = 0
    norm_mean = float(norms.mean())
    norm_std = float(norms.std())
    norm_cv = norm_std / (norm_mean + _C.NORM_EPS)
    return {
        "centroid_norm": float(centroid_norm),
        "mean_norm": norm_mean, "norm_std": norm_std, "norm_cv": norm_cv,
        "dims_for_50pct": dim_50, "dims_for_90pct": dim_90, "dims_for_99pct": dim_99,
        "participation_ratio": float(participation_ratio),
        "sphericity": float(sphericity),
        "top_eigenvalue_ratio": float(eigenvalues[0] / total_var) if total_var > 0 else 0,
        "top_eigenvalues": eigenvalues[:_C.INTRINSIC_DIM_TOP_N].tolist(),
    }


def compute_cone_fit(activations: torch.Tensor) -> Dict[str, Any]:
    """Test if activations lie on a cone from the origin."""
    X = activations.float().cpu().numpy()
    n, d = X.shape
    if n < 3:
        return {"error": "need at least 3 points"}
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    valid = norms.squeeze() > _C.NORM_EPS
    X_normalized = X[valid] / norms[valid]
    if len(X_normalized) < 3:
        return {"error": "too few valid points"}
    mean_dir = X_normalized.mean(axis=0)
    mean_dir_norm = np.linalg.norm(mean_dir)
    if mean_dir_norm < _C.NORM_EPS:
        return {"is_cone": False, "cone_concentration": 0}
    cone_axis = mean_dir / mean_dir_norm
    cos_angles = X_normalized @ cone_axis
    angles = np.arccos(np.clip(cos_angles, -1, 1))
    angles_deg = np.degrees(angles)
    cone_half_angle = float(np.percentile(angles_deg, _C.CLOUD_CONE_PERCENTILE))
    concentration = float(cos_angles.mean())
    return {
        "cone_axis_strength": float(mean_dir_norm),
        "cone_concentration": concentration,
        "cone_half_angle_90pct": cone_half_angle,
        "angle_mean": float(angles_deg.mean()),
        "angle_std": float(angles_deg.std()),
        "angle_max": float(angles_deg.max()),
    }


def compute_sphere_fit(activations: torch.Tensor) -> Dict[str, Any]:
    """Test if activations lie on a sphere."""
    X = activations.float().cpu().numpy()
    n, d = X.shape
    if n < 3:
        return {"error": "need at least 3 points"}
    centroid = X.mean(axis=0)
    distances = np.linalg.norm(X - centroid, axis=1)
    mean_radius = float(distances.mean())
    radius_std = float(distances.std())
    radius_cv = radius_std / (mean_radius + _C.NORM_EPS)
    origin_distances = np.linalg.norm(X, axis=1)
    origin_radius_mean = float(origin_distances.mean())
    origin_radius_std = float(origin_distances.std())
    origin_radius_cv = origin_radius_std / (origin_radius_mean + _C.NORM_EPS)
    return {
        "centroid_radius_mean": mean_radius,
        "centroid_radius_std": radius_std, "centroid_radius_cv": radius_cv,
        "origin_radius_mean": origin_radius_mean,
        "origin_radius_std": origin_radius_std, "origin_radius_cv": origin_radius_cv,
    }


def compute_manifold_dimension(activations: torch.Tensor, k_neighbors: int = _C.CLOUD_K_NEIGHBORS_DEFAULT) -> Dict[str, Any]:
    """Estimate intrinsic dimension using local methods."""
    X = activations.float().cpu().numpy()
    n, d = X.shape
    if n < k_neighbors + 1:
        return {"error": "not enough points for local estimation"}
    if n > _C.CLOUD_MAX_SAMPLES_MANIFOLD:
        idx = np.random.choice(n, _C.CLOUD_MAX_SAMPLES_MANIFOLD, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    dists = squareform(pdist(X_sub))
    local_dims = []
    for i, X_sub in enumerate(X_sub):
        d_i = np.sort(dists[i])[1:k_neighbors+1]
        if d_i[-1] > 0 and d_i[0] > 0:
            log_ratios = np.log(d_i[-1] / d_i[:-1])
            if log_ratios.sum() > 0:
                local_dim = (k_neighbors - 1) / log_ratios.sum()
                local_dims.append(local_dim)
    if not local_dims:
        return {"error": "could not estimate local dimensions"}
    local_dims = np.array(local_dims)
    return {
        "local_dim_mean": float(local_dims.mean()),
        "local_dim_std": float(local_dims.std()),
        "local_dim_median": float(np.median(local_dims)),
        "local_dim_min": float(local_dims.min()),
        "local_dim_max": float(local_dims.max()),
    }


def compute_cluster_structure(activations: torch.Tensor, max_clusters: int = _C.MAX_CLUSTERS) -> Dict[str, Any]:
    """Analyze cluster structure of the point cloud."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    X = activations.float().cpu().numpy()
    n, d = X.shape
    if n < _C.MIN_CLOUD_CLUSTER_POINTS:
        return {"error": "need at least 6 points"}
    silhouettes = {}
    inertias = {}
    for k in range(2, min(max_clusters + 1, n // 2)):
        km = KMeans(n_clusters=k, random_state=_C.DEFAULT_RANDOM_SEED, n_init=_C.LINEARITY_N_INIT)
        labels = km.fit_predict(X)
        try:
            sil = silhouette_score(X, labels)
            silhouettes[k] = float(sil)
        except:
            silhouettes[k] = 0
        inertias[k] = float(km.inertia_)
    if silhouettes:
        best_k = max(silhouettes, key=silhouettes.get)
        best_silhouette = silhouettes[best_k]
    else:
        best_k = 1
        best_silhouette = 0
    centroid = X.mean(axis=0)
    distances_to_centroid = np.linalg.norm(X - centroid, axis=1)
    compactness = float(distances_to_centroid.mean())
    return {
        "best_k": best_k, "best_silhouette": best_silhouette,
        "silhouette_scores": silhouettes, "single_cluster_compactness": compactness,
    }


def compute_density_structure(activations: torch.Tensor, k_neighbors: int = _C.CLOUD_K_NEIGHBORS_DEFAULT) -> Dict[str, Any]:
    """Analyze density variations in the point cloud."""
    X = activations.float().cpu().numpy()
    n, d = X.shape
    if n < k_neighbors + 1:
        return {"error": "not enough points"}
    dists = squareform(pdist(X))
    local_densities = []
    for i in range(n):
        kth_dist = np.sort(dists[i])[k_neighbors]
        if kth_dist > 0:
            local_densities.append(1.0 / kth_dist)
        else:
            local_densities.append(float('inf'))
    local_densities = np.array(local_densities)
    finite_densities = local_densities[np.isfinite(local_densities)]
    if len(finite_densities) == 0:
        return {"error": "all densities infinite"}
    density_cv = float(finite_densities.std() / (finite_densities.mean() + _C.NORM_EPS))
    return {
        "density_mean": float(finite_densities.mean()),
        "density_std": float(finite_densities.std()),
        "density_cv": density_cv,
        "density_min": float(finite_densities.min()),
        "density_max": float(finite_densities.max()),
    }


def compute_topology_indicators(activations: torch.Tensor) -> Dict[str, Any]:
    """Basic topological indicators using MST connectivity."""
    X = activations.float().cpu().numpy()
    n, d = X.shape
    if n < 10:
        return {"error": "need at least 10 points"}
    if n > _C.CLOUD_MAX_SAMPLES_TOPOLOGY:
        idx = np.random.choice(n, _C.CLOUD_MAX_SAMPLES_TOPOLOGY, replace=False)
        X = X[idx]
        n = _C.CLOUD_MAX_SAMPLES_TOPOLOGY
    dists = squareform(pdist(X))
    sorted_edges = np.sort(dists[np.triu_indices(n, k=1)])
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.sparse import csr_matrix
    mst = minimum_spanning_tree(csr_matrix(dists))
    mst_edges = mst.data
    connectivity_radius = float(mst_edges.max()) if len(mst_edges) > 0 else 0
    mean_pairwise = float(sorted_edges.mean())
    connectivity_ratio = connectivity_radius / (mean_pairwise + _C.NORM_EPS)
    return {
        "connectivity_radius": connectivity_radius,
        "mean_pairwise_distance": mean_pairwise,
        "connectivity_ratio": connectivity_ratio,
    }
