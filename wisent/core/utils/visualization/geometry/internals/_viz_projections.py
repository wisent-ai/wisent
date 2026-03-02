"""Advanced projection visualizations."""
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import warnings
from wisent.core.utils.config_tools.constants import (
    NORM_EPS, VIZ_PERPLEXITY, VIZ_N_NEIGHBORS, VIZ_MIN_DIST,
    VIZ_N_NEIGHBORS_TRIMAP, VIZ_NUM_ITERS, VIZ_PCA_DIMS_TRIMAP,
    DEFAULT_RANDOM_SEED, VIZ_N_COMPONENTS_2D,
)

def plot_tsne_projection(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    perplexity: int = VIZ_PERPLEXITY,
    title: str = "t-SNE Projection",
) -> Dict[str, Any]:
    """
    Project activations using t-SNE.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return {"error": "sklearn not available"}

    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()

    X = np.vstack([pos, neg])

    # Adjust perplexity if needed
    n_samples = len(X)
    perplexity = min(perplexity, n_samples - 1)

    if n_samples < 5:
        return {"error": "not enough samples for t-SNE"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne = TSNE(n_components=VIZ_N_COMPONENTS_2D, perplexity=perplexity, random_state=DEFAULT_RANDOM_SEED)
        X_tsne = tsne.fit_transform(X)

    pos_tsne = X_tsne[:len(pos)]
    neg_tsne = X_tsne[len(pos):]

    return {
        "pos_projected": pos_tsne,
        "neg_projected": neg_tsne,
        "title": title,
    }


def plot_umap_projection(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_neighbors: int = VIZ_N_NEIGHBORS,
    min_dist: float = VIZ_MIN_DIST,
    title: str = "UMAP Projection",
) -> Dict[str, Any]:
    """
    Project activations using UMAP.
    """
    try:
        import umap
    except ImportError:
        return {"error": "umap not available"}

    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()

    X = np.vstack([pos, neg])

    n_samples = len(X)
    n_neighbors = min(n_neighbors, n_samples - 1)

    if n_samples < 5:
        return {"error": "not enough samples for UMAP"}

    reducer = umap.UMAP(n_components=VIZ_N_COMPONENTS_2D, n_neighbors=n_neighbors, min_dist=min_dist, random_state=DEFAULT_RANDOM_SEED)
    X_umap = reducer.fit_transform(X)

    pos_umap = X_umap[:len(pos)]
    neg_umap = X_umap[len(pos):]

    return {
        "pos_projected": pos_umap,
        "neg_projected": neg_umap,
        "title": title,
    }


def plot_pacmap_projection(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_neighbors: int = VIZ_N_NEIGHBORS_TRIMAP,
    num_iters: int = VIZ_NUM_ITERS,
    pca_dims: int = VIZ_PCA_DIMS_TRIMAP,
    title: str = "PaCMAP Projection",
) -> Dict[str, Any]:
    """
    Project activations using PaCMAP (alternative implementation).

    Uses pacmap_alt which avoids numba threading issues on macOS.
    """
    from .pacmap_alt import plot_pacmap_alt

    return plot_pacmap_alt(
        pos_activations,
        neg_activations,
        n_neighbors=n_neighbors,
        num_iters=num_iters,
        title=title,
    )


def plot_cone_visualization(
    activations: torch.Tensor,
    title: str = "Cone Structure",
) -> Dict[str, Any]:
    """
    Visualize how cone-like the activations are.

    Projects to 2D and shows angular distribution from mean direction.
    """
    X = activations.float().cpu().numpy()

    # Normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    valid = norms.squeeze() > NORM_EPS
    X_norm = X[valid] / norms[valid]

    # Mean direction
    mean_dir = X_norm.mean(axis=0)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + NORM_EPS)

    # Angles from mean
    cos_angles = X_norm @ mean_dir
    angles = np.degrees(np.arccos(np.clip(cos_angles, -1, 1)))

    # Project to 2D: mean direction and orthogonal
    from sklearn.decomposition import PCA
    pca = PCA(n_components=VIZ_N_COMPONENTS_2D)
    X_2d = pca.fit_transform(X_norm)

    return {
        "angles_from_mean": angles,
        "projected_2d": X_2d,
        "mean_angle": float(angles.mean()),
        "max_angle": float(angles.max()),
        "title": title,
    }


def plot_layer_comparison(
    activations_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    metric_fn,
    metric_name: str = "metric",
    title: str = "Metric Across Layers",
) -> Dict[str, Any]:
    """
    Plot a metric across different layers.
    """
    layers = sorted(activations_by_layer.keys())
    values = []

    for layer in layers:
        pos, neg = activations_by_layer[layer]
        try:
            result = metric_fn(pos, neg)
            if isinstance(result, dict):
                # Try to extract a scalar
                val = result.get(metric_name, result.get("mean", 0))
            else:
                val = result
            values.append(float(val))
        except:
            values.append(0)

    return {
        "layers": layers,
        "values": values,
        "metric_name": metric_name,
        "title": title,
    }


