"""
Activation space structure metrics.

These metrics characterize the geometry of the activation space itself,
not just the difference vectors. They describe where pos and neg
point clouds live and their relationship.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import pdist, squareform


def compute_cloud_shape(
    activations: torch.Tensor,
) -> Dict[str, Any]:
    """
    Characterize the shape of a single activation point cloud.

    Measures:
    - Dimension (intrinsic)
    - Spread (how dispersed)
    - Centering (distance from origin)
    - Sphericity (how sphere-like)
    """
    X = activations.float().cpu().numpy()
    n, d = X.shape

    if n < 3:
        return {"error": "need at least 3 points"}

    # Norms and centering
    norms = np.linalg.norm(X, axis=1)
    centroid = X.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)

    # Center the data
    X_centered = X - centroid

    # Covariance and eigenvalues
    cov = np.cov(X_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)

    total_var = eigenvalues.sum()

    # Intrinsic dimension estimates
    if total_var > 0:
        cumsum = np.cumsum(eigenvalues) / total_var
        dim_50 = int(np.searchsorted(cumsum, 0.5) + 1)
        dim_90 = int(np.searchsorted(cumsum, 0.9) + 1)
        dim_99 = int(np.searchsorted(cumsum, 0.99) + 1)

        # Participation ratio
        participation_ratio = (total_var ** 2) / (np.sum(eigenvalues ** 2) + 1e-8)
    else:
        dim_50, dim_90, dim_99 = 1, 1, 1
        participation_ratio = 1

    # Sphericity: ratio of smallest to largest eigenvalue
    # High sphericity = roughly equal spread in all directions
    if eigenvalues[0] > 0:
        sphericity = eigenvalues[-1] / eigenvalues[0]
    else:
        sphericity = 0

    # Norm distribution
    norm_mean = float(norms.mean())
    norm_std = float(norms.std())
    norm_cv = norm_std / (norm_mean + 1e-8)  # coefficient of variation

    return {
        # Centering
        "centroid_norm": float(centroid_norm),
        "mean_norm": norm_mean,
        "norm_std": norm_std,
        "norm_cv": norm_cv,  # low = similar norms (sphere-like)

        # Intrinsic dimension
        "dims_for_50pct": dim_50,
        "dims_for_90pct": dim_90,
        "dims_for_99pct": dim_99,
        "participation_ratio": float(participation_ratio),

        # Shape
        "sphericity": float(sphericity),
        "top_eigenvalue_ratio": float(eigenvalues[0] / total_var) if total_var > 0 else 0,

        # Raw eigenvalues (top 10)
        "top_eigenvalues": eigenvalues[:10].tolist(),
    }


def compute_cone_fit(
    activations: torch.Tensor,
) -> Dict[str, Any]:
    """
    Test if activations lie on a cone from the origin.

    A cone means: all vectors point in similar directions
    but with varying magnitudes.
    """
    X = activations.float().cpu().numpy()
    n, d = X.shape

    if n < 3:
        return {"error": "need at least 3 points"}

    # Normalize to unit sphere
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    valid = norms.squeeze() > 1e-8
    X_normalized = X[valid] / norms[valid]

    if len(X_normalized) < 3:
        return {"error": "too few valid points"}

    # Cone axis = mean direction
    mean_dir = X_normalized.mean(axis=0)
    mean_dir_norm = np.linalg.norm(mean_dir)

    if mean_dir_norm < 1e-8:
        return {"is_cone": False, "cone_concentration": 0}

    cone_axis = mean_dir / mean_dir_norm

    # Angular distance from cone axis
    cos_angles = X_normalized @ cone_axis
    angles = np.arccos(np.clip(cos_angles, -1, 1))
    angles_deg = np.degrees(angles)

    # Cone half-angle (contains 90% of points)
    cone_half_angle = float(np.percentile(angles_deg, 90))

    # Concentration: how tightly clustered around axis
    # 1 = all on axis, 0 = spread over sphere
    concentration = float(cos_angles.mean())

    return {
        "cone_axis_strength": float(mean_dir_norm),  # how strong the mean direction is
        "cone_concentration": concentration,
        "cone_half_angle_90pct": cone_half_angle,
        "angle_mean": float(angles_deg.mean()),
        "angle_std": float(angles_deg.std()),
        "angle_max": float(angles_deg.max()),
    }


def compute_sphere_fit(
    activations: torch.Tensor,
) -> Dict[str, Any]:
    """
    Test if activations lie on a sphere (constant distance from a center).
    """
    X = activations.float().cpu().numpy()
    n, d = X.shape

    if n < 3:
        return {"error": "need at least 3 points"}

    # Fit sphere: find center that minimizes variance of distances
    # Simple approach: use centroid as center
    centroid = X.mean(axis=0)

    # Distances from centroid
    distances = np.linalg.norm(X - centroid, axis=1)
    mean_radius = float(distances.mean())
    radius_std = float(distances.std())

    # Sphere fit quality: low std/mean = good sphere
    radius_cv = radius_std / (mean_radius + 1e-8)

    # Also check distances from origin
    origin_distances = np.linalg.norm(X, axis=1)
    origin_radius_mean = float(origin_distances.mean())
    origin_radius_std = float(origin_distances.std())
    origin_radius_cv = origin_radius_std / (origin_radius_mean + 1e-8)

    return {
        # Sphere around centroid
        "centroid_radius_mean": mean_radius,
        "centroid_radius_std": radius_std,
        "centroid_radius_cv": radius_cv,  # low = good sphere fit

        # Sphere around origin
        "origin_radius_mean": origin_radius_mean,
        "origin_radius_std": origin_radius_std,
        "origin_radius_cv": origin_radius_cv,
    }


def compute_manifold_dimension(
    activations: torch.Tensor,
    k_neighbors: int = 10,
) -> Dict[str, Any]:
    """
    Estimate intrinsic dimension using local methods.

    Uses correlation dimension and local PCA.
    """
    X = activations.float().cpu().numpy()
    n, d = X.shape

    if n < k_neighbors + 1:
        return {"error": "not enough points for local estimation"}

    # Compute pairwise distances (subsample if large)
    if n > 500:
        idx = np.random.choice(n, 500, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    dists = squareform(pdist(X_sub))

    # Local intrinsic dimension via MLE
    # For each point, estimate dimension from k nearest neighbors
    local_dims = []
    for i in range(len(X_sub)):
        # Get k nearest neighbor distances (excluding self)
        d_i = np.sort(dists[i])[1:k_neighbors+1]

        if d_i[-1] > 0 and d_i[0] > 0:
            # MLE estimate: d = k / sum(log(r_k / r_j))
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


def compute_cluster_structure(
    activations: torch.Tensor,
    max_clusters: int = 5,
) -> Dict[str, Any]:
    """
    Analyze cluster structure of the point cloud.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = activations.float().cpu().numpy()
    n, d = X.shape

    if n < 6:
        return {"error": "need at least 6 points"}

    # Try different numbers of clusters
    silhouettes = {}
    inertias = {}

    for k in range(2, min(max_clusters + 1, n // 2)):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        try:
            sil = silhouette_score(X, labels)
            silhouettes[k] = float(sil)
        except:
            silhouettes[k] = 0

        inertias[k] = float(km.inertia_)

    # Best k by silhouette
    if silhouettes:
        best_k = max(silhouettes, key=silhouettes.get)
        best_silhouette = silhouettes[best_k]
    else:
        best_k = 1
        best_silhouette = 0

    # Single cluster compactness
    centroid = X.mean(axis=0)
    distances_to_centroid = np.linalg.norm(X - centroid, axis=1)
    compactness = float(distances_to_centroid.mean())

    return {
        "best_k": best_k,
        "best_silhouette": best_silhouette,
        "silhouette_scores": silhouettes,
        "single_cluster_compactness": compactness,
    }


def compute_density_structure(
    activations: torch.Tensor,
    k_neighbors: int = 10,
) -> Dict[str, Any]:
    """
    Analyze density variations in the point cloud.
    """
    X = activations.float().cpu().numpy()
    n, d = X.shape

    if n < k_neighbors + 1:
        return {"error": "not enough points"}

    # Compute local density for each point
    # Density ~ 1 / (distance to k-th neighbor)
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

    # Density uniformity: low std/mean = uniform density
    density_cv = float(finite_densities.std() / (finite_densities.mean() + 1e-8))

    return {
        "density_mean": float(finite_densities.mean()),
        "density_std": float(finite_densities.std()),
        "density_cv": density_cv,  # low = uniform, high = variable
        "density_min": float(finite_densities.min()),
        "density_max": float(finite_densities.max()),
    }


def compute_topology_indicators(
    activations: torch.Tensor,
) -> Dict[str, Any]:
    """
    Basic topological indicators.

    Full persistent homology is expensive, so we use proxies.
    """
    X = activations.float().cpu().numpy()
    n, d = X.shape

    if n < 10:
        return {"error": "need at least 10 points"}

    # Subsample if large
    if n > 300:
        idx = np.random.choice(n, 300, replace=False)
        X = X[idx]
        n = 300

    dists = squareform(pdist(X))

    # Connectivity at different scales
    # At what distance threshold is the point cloud connected?
    sorted_edges = np.sort(dists[np.triu_indices(n, k=1)])

    # Approximate: find distance where graph becomes connected
    # Using minimum spanning tree proxy
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.sparse import csr_matrix

    mst = minimum_spanning_tree(csr_matrix(dists))
    mst_edges = mst.data
    connectivity_radius = float(mst_edges.max()) if len(mst_edges) > 0 else 0

    # Ratio of connectivity radius to mean pairwise distance
    mean_pairwise = float(sorted_edges.mean())
    connectivity_ratio = connectivity_radius / (mean_pairwise + 1e-8)

    return {
        "connectivity_radius": connectivity_radius,
        "mean_pairwise_distance": mean_pairwise,
        "connectivity_ratio": connectivity_ratio,  # low = compact, high = spread/disconnected
    }


def compute_two_cloud_relationship(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, Any]:
    """
    Analyze the geometric relationship between pos and neg clouds.
    """
    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()

    n_pos, d = pos.shape
    n_neg = neg.shape[0]

    if n_pos < 3 or n_neg < 3:
        return {"error": "need at least 3 points in each cloud"}

    # Centroids
    pos_centroid = pos.mean(axis=0)
    neg_centroid = neg.mean(axis=0)

    # Distance between centroids
    centroid_distance = float(np.linalg.norm(pos_centroid - neg_centroid))

    # Direction between centroids
    centroid_direction = pos_centroid - neg_centroid
    centroid_dir_norm = np.linalg.norm(centroid_direction)
    if centroid_dir_norm > 1e-8:
        centroid_direction = centroid_direction / centroid_dir_norm

    # Within-class spread
    pos_spread = float(np.linalg.norm(pos - pos_centroid, axis=1).mean())
    neg_spread = float(np.linalg.norm(neg - neg_centroid, axis=1).mean())

    # Separation ratio: distance between centroids / average spread
    avg_spread = (pos_spread + neg_spread) / 2
    separation_ratio = centroid_distance / (avg_spread + 1e-8)

    # Overlap: what fraction of pos is closer to neg_centroid than pos_centroid?
    pos_to_pos_centroid = np.linalg.norm(pos - pos_centroid, axis=1)
    pos_to_neg_centroid = np.linalg.norm(pos - neg_centroid, axis=1)
    pos_overlap = float((pos_to_neg_centroid < pos_to_pos_centroid).mean())

    neg_to_pos_centroid = np.linalg.norm(neg - pos_centroid, axis=1)
    neg_to_neg_centroid = np.linalg.norm(neg - neg_centroid, axis=1)
    neg_overlap = float((neg_to_pos_centroid < neg_to_neg_centroid).mean())

    # Parallelism: do the clouds have similar shape/orientation?
    # Compare principal components
    pos_centered = pos - pos_centroid
    neg_centered = neg - neg_centroid

    try:
        _, _, Vh_pos = np.linalg.svd(pos_centered, full_matrices=False)
        _, _, Vh_neg = np.linalg.svd(neg_centered, full_matrices=False)

        # Alignment of top principal components
        pc1_alignment = float(abs(np.dot(Vh_pos[0], Vh_neg[0])))
        pc2_alignment = float(abs(np.dot(Vh_pos[1], Vh_neg[1]))) if len(Vh_pos) > 1 else 0
    except:
        pc1_alignment = 0
        pc2_alignment = 0

    return {
        # Distance
        "centroid_distance": centroid_distance,
        "separation_ratio": separation_ratio,  # high = well separated

        # Spread
        "pos_spread": pos_spread,
        "neg_spread": neg_spread,

        # Overlap
        "pos_overlap_fraction": pos_overlap,  # low = separated
        "neg_overlap_fraction": neg_overlap,

        # Shape alignment
        "pc1_alignment": pc1_alignment,  # high = parallel manifolds
        "pc2_alignment": pc2_alignment,
    }


def compute_relative_position(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, Any]:
    """
    Analyze relative position: is neg a shifted/rotated version of pos?
    """
    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()

    n = min(len(pos), len(neg))
    pos, neg = pos[:n], neg[:n]

    # Centroids
    pos_centroid = pos.mean(axis=0)
    neg_centroid = neg.mean(axis=0)
    shift = pos_centroid - neg_centroid
    shift_norm = np.linalg.norm(shift)

    # If we shift neg by the centroid difference, how close is it to pos?
    neg_shifted = neg + shift

    # After shifting, measure alignment
    residuals = pos - neg_shifted
    residual_norms = np.linalg.norm(residuals, axis=1)

    # Compare to original differences
    orig_diffs = pos - neg
    orig_diff_norms = np.linalg.norm(orig_diffs, axis=1)

    # Shift explains how much of the difference?
    shift_explains = 1 - (residual_norms.mean() / (orig_diff_norms.mean() + 1e-8))

    # Is the relationship a pure translation?
    # If yes, all (pos - neg) vectors should be similar
    diff_normalized = orig_diffs / (orig_diff_norms[:, np.newaxis] + 1e-8)
    mean_diff_dir = diff_normalized.mean(axis=0)
    mean_diff_dir = mean_diff_dir / (np.linalg.norm(mean_diff_dir) + 1e-8)

    translation_consistency = float((diff_normalized @ mean_diff_dir).mean())

    return {
        "shift_vector_norm": float(shift_norm),
        "shift_explains_fraction": float(shift_explains),  # high = pure translation
        "translation_consistency": translation_consistency,  # high = consistent shift
        "residual_after_shift": float(residual_norms.mean()),
    }


def analyze_activation_structure(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, Any]:
    """
    Comprehensive analysis of activation space structure.

    Analyzes:
    - Individual cloud shapes (pos and neg)
    - Relationship between clouds
    - Specific structure tests (cone, sphere, manifold)
    """
    results = {}

    # Individual cloud analysis
    results["pos_shape"] = compute_cloud_shape(pos_activations)
    results["neg_shape"] = compute_cloud_shape(neg_activations)

    # Cone fit
    results["pos_cone"] = compute_cone_fit(pos_activations)
    results["neg_cone"] = compute_cone_fit(neg_activations)

    # Sphere fit
    results["pos_sphere"] = compute_sphere_fit(pos_activations)
    results["neg_sphere"] = compute_sphere_fit(neg_activations)

    # Manifold dimension
    results["pos_manifold"] = compute_manifold_dimension(pos_activations)
    results["neg_manifold"] = compute_manifold_dimension(neg_activations)

    # Cluster structure
    results["pos_clusters"] = compute_cluster_structure(pos_activations)
    results["neg_clusters"] = compute_cluster_structure(neg_activations)

    # Density
    results["pos_density"] = compute_density_structure(pos_activations)
    results["neg_density"] = compute_density_structure(neg_activations)

    # Relationship between clouds
    results["relationship"] = compute_two_cloud_relationship(pos_activations, neg_activations)
    results["relative_position"] = compute_relative_position(pos_activations, neg_activations)

    return results
