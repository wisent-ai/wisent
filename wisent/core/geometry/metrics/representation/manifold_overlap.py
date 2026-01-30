"""
Manifold and overlap metrics for activation geometry.

Metrics for manifold structure, curvature, and direction overlap.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional


def compute_manifold_metrics(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_neighbors: int = 10,
) -> Dict[str, Any]:
    """
    Compute manifold and curvature metrics.

    Describes:
    - Is the separation surface flat or curved?
    - Local vs global structure
    - Intrinsic dimensionality estimates
    """
    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()

    n = min(len(pos), len(neg))
    pos, neg = pos[:n], neg[:n]
    diffs = pos - neg

    # PCA on diffs
    from sklearn.decomposition import PCA

    n_components = min(50, n - 1, diffs.shape[1])
    pca = PCA(n_components=n_components)
    diffs_pca = pca.fit_transform(diffs)

    explained_variance = pca.explained_variance_ratio_
    cumsum_variance = np.cumsum(explained_variance)

    # Intrinsic dim estimates
    dims_for_50 = int(np.searchsorted(cumsum_variance, 0.5) + 1)
    dims_for_90 = int(np.searchsorted(cumsum_variance, 0.9) + 1)
    dims_for_99 = int(np.searchsorted(cumsum_variance, 0.99) + 1)

    # Participation ratio (effective dimensionality)
    participation_ratio = (explained_variance.sum() ** 2) / (np.sum(explained_variance ** 2) + 1e-8)

    # Local linearity: how well does local PCA match global PCA
    local_linearities = []
    for i in range(min(n, 50)):
        # Find k nearest neighbors
        distances = np.linalg.norm(diffs - diffs[i], axis=1)
        neighbor_idx = np.argsort(distances)[1:n_neighbors+1]

        if len(neighbor_idx) >= 3:
            local_diffs = diffs[neighbor_idx]
            local_mean = local_diffs.mean(axis=0)
            local_norm = np.linalg.norm(local_mean)

            global_mean = diffs.mean(axis=0)
            global_norm = np.linalg.norm(global_mean)

            if local_norm > 1e-8 and global_norm > 1e-8:
                local_dir = local_mean / local_norm
                global_dir = global_mean / global_norm
                local_linearities.append(np.abs(np.dot(local_dir, global_dir)))

    local_linearities = np.array(local_linearities)

    # Curvature proxy: variance of local directions
    curvature_proxy = 1 - local_linearities.mean() if len(local_linearities) > 0 else None

    return {
        # PCA variance
        "variance_pc1": float(explained_variance[0]) if len(explained_variance) > 0 else None,
        "variance_pc2": float(explained_variance[1]) if len(explained_variance) > 1 else None,
        "variance_pc3": float(explained_variance[2]) if len(explained_variance) > 2 else None,
        "variance_top5": float(cumsum_variance[4]) if len(cumsum_variance) > 4 else None,
        "variance_top10": float(cumsum_variance[9]) if len(cumsum_variance) > 9 else None,

        # Dimensionality
        "dims_for_50pct_variance": dims_for_50,
        "dims_for_90pct_variance": dims_for_90,
        "dims_for_99pct_variance": dims_for_99,
        "participation_ratio": float(participation_ratio),

        # Curvature/linearity
        "local_linearity_mean": float(local_linearities.mean()) if len(local_linearities) > 0 else None,
        "local_linearity_std": float(local_linearities.std()) if len(local_linearities) > 0 else None,
        "local_linearity_min": float(local_linearities.min()) if len(local_linearities) > 0 else None,
        "curvature_proxy": float(curvature_proxy) if curvature_proxy is not None else None,

        # Full explained variance (for plotting)
        "explained_variance_ratio": explained_variance.tolist(),
    }


def compute_direction_overlap_metrics(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    other_directions: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Compute overlap with other known directions.

    Describes:
    - Does this concept's direction overlap with other concepts?
    - How unique is this direction?

    Args:
        other_directions: Dict mapping concept name -> unit direction vector
    """
    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()

    n = min(len(pos), len(neg))
    diffs = pos[:n] - neg[:n]
    mean_diff = diffs.mean(axis=0)
    norm = np.linalg.norm(mean_diff)

    if norm < 1e-8:
        return {"error": "steering direction norm too small"}

    direction = mean_diff / norm

    if not other_directions:
        return {
            "steering_direction_norm": float(norm),
            "other_directions_provided": False,
        }

    # Compute overlap with each other direction
    overlaps = {}
    for name, other_dir in other_directions.items():
        if len(other_dir) == len(direction):
            other_norm = np.linalg.norm(other_dir)
            if other_norm > 1e-8:
                other_normalized = other_dir / other_norm
                overlap = float(np.dot(direction, other_normalized))
                overlaps[name] = overlap

    # Find most overlapping
    if overlaps:
        most_overlapping = max(overlaps.keys(), key=lambda k: abs(overlaps[k]))
        least_overlapping = min(overlaps.keys(), key=lambda k: abs(overlaps[k]))
    else:
        most_overlapping = None
        least_overlapping = None

    return {
        "steering_direction_norm": float(norm),
        "other_directions_provided": True,
        "n_other_directions": len(other_directions),

        # Overlaps
        "overlaps": overlaps,
        "most_overlapping": most_overlapping,
        "most_overlapping_value": overlaps.get(most_overlapping) if most_overlapping else None,
        "least_overlapping": least_overlapping,
        "least_overlapping_value": overlaps.get(least_overlapping) if least_overlapping else None,

        # Summary stats
        "mean_absolute_overlap": float(np.mean([abs(v) for v in overlaps.values()])) if overlaps else None,
        "max_absolute_overlap": float(np.max([abs(v) for v in overlaps.values()])) if overlaps else None,
    }
