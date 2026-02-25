"""
Visualization functions for activation geometry.

All functions return matplotlib figures or data for plotting.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import warnings
from wisent.core.constants import NORM_EPS, VIZ_N_COMPONENTS_2D, VIZ_PCA_COMPONENTS, VIZ_MAX_SAMPLES


def plot_pca_projection(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_components: int = VIZ_N_COMPONENTS_2D,
    title: str = "PCA Projection",
) -> Dict[str, Any]:
    """
    Project activations to 2D or 3D using PCA and return plot data.
    """
    from sklearn.decomposition import PCA

    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()

    # Combine for fitting PCA
    X = np.vstack([pos, neg])
    labels = np.array([1] * len(pos) + [0] * len(neg))

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    pos_pca = X_pca[:len(pos)]
    neg_pca = X_pca[len(pos):]

    return {
        "pos_projected": pos_pca,
        "neg_projected": neg_pca,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "components": pca.components_,
        "n_components": n_components,
        "title": title,
    }


def plot_diff_vectors(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    title: str = "Difference Vectors",
) -> Dict[str, Any]:
    """
    Visualize the difference vectors projected to 2D.
    """
    from sklearn.decomposition import PCA

    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()

    n = min(len(pos), len(neg))
    diffs = pos[:n] - neg[:n]

    # Project diffs to 2D
    pca = PCA(n_components=VIZ_N_COMPONENTS_2D)
    diffs_2d = pca.fit_transform(diffs)

    # Mean direction
    mean_diff = diffs.mean(axis=0)
    mean_diff_2d = pca.transform(mean_diff.reshape(1, -1))[0]

    return {
        "diffs_projected": diffs_2d,
        "mean_diff_projected": mean_diff_2d,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "title": title,
    }


def plot_norm_distribution(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    title: str = "Norm Distribution",
) -> Dict[str, Any]:
    """
    Plot distribution of activation norms.
    """
    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()

    pos_norms = np.linalg.norm(pos, axis=1)
    neg_norms = np.linalg.norm(neg, axis=1)

    return {
        "pos_norms": pos_norms,
        "neg_norms": neg_norms,
        "pos_mean": float(pos_norms.mean()),
        "neg_mean": float(neg_norms.mean()),
        "pos_std": float(pos_norms.std()),
        "neg_std": float(neg_norms.std()),
        "title": title,
    }


def plot_alignment_distribution(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    title: str = "Alignment Distribution",
) -> Dict[str, Any]:
    """
    Plot distribution of how well each diff aligns with mean diff.
    """
    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()

    n = min(len(pos), len(neg))
    diffs = pos[:n] - neg[:n]

    # Mean direction
    mean_diff = diffs.mean(axis=0)
    mean_diff_norm = np.linalg.norm(mean_diff)

    if mean_diff_norm < NORM_EPS:
        return {"error": "mean diff too small"}

    mean_diff_normalized = mean_diff / mean_diff_norm

    # Per-diff alignment
    diff_norms = np.linalg.norm(diffs, axis=1)
    valid = diff_norms > NORM_EPS
    alignments = np.zeros(n)
    alignments[valid] = (diffs[valid] / diff_norms[valid, np.newaxis]) @ mean_diff_normalized

    return {
        "alignments": alignments,
        "mean": float(alignments.mean()),
        "std": float(alignments.std()),
        "min": float(alignments.min()),
        "max": float(alignments.max()),
        "title": title,
    }


def plot_eigenvalue_spectrum(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_components: int = VIZ_PCA_COMPONENTS,
    title: str = "Eigenvalue Spectrum",
) -> Dict[str, Any]:
    """
    Plot eigenvalue spectrum of the difference vectors.
    """
    from sklearn.decomposition import PCA

    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()

    n = min(len(pos), len(neg))
    diffs = pos[:n] - neg[:n]

    n_comp = min(n_components, n - 1, diffs.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(diffs)

    return {
        "eigenvalues": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "title": title,
    }


def plot_pairwise_distances(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    max_samples: int = VIZ_MAX_SAMPLES,
    title: str = "Pairwise Distances",
) -> Dict[str, Any]:
    """
    Plot distribution of pairwise distances within and between classes.
    """
    from scipy.spatial.distance import pdist

    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()

    # Subsample if needed
    if len(pos) > max_samples:
        idx = np.random.choice(len(pos), max_samples, replace=False)
        pos = pos[idx]
    if len(neg) > max_samples:
        idx = np.random.choice(len(neg), max_samples, replace=False)
        neg = neg[idx]

    # Within-class distances
    pos_dists = pdist(pos)
    neg_dists = pdist(neg)

    # Between-class distances
    from scipy.spatial.distance import cdist
    between_dists = cdist(pos, neg).flatten()

    # Full pairwise distance matrix for visualization
    all_data = np.vstack([pos, neg])
    distance_matrix = cdist(all_data, all_data)

    return {
        "pos_within": pos_dists,
        "neg_within": neg_dists,
        "between": between_dists,
        "pos_within_mean": float(pos_dists.mean()),
        "neg_within_mean": float(neg_dists.mean()),
        "between_mean": float(between_dists.mean()),
        "distance_matrix": distance_matrix,
        "n_pos": len(pos),
        "title": title,
    }


