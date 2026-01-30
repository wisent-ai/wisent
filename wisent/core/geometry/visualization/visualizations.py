"""
Visualization functions for activation geometry.

All functions return matplotlib figures or data for plotting.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import warnings


def plot_pca_projection(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_components: int = 2,
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
    pca = PCA(n_components=2)
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

    if mean_diff_norm < 1e-8:
        return {"error": "mean diff too small"}

    mean_diff_normalized = mean_diff / mean_diff_norm

    # Per-diff alignment
    diff_norms = np.linalg.norm(diffs, axis=1)
    valid = diff_norms > 1e-8
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
    n_components: int = 50,
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
    max_samples: int = 200,
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


def plot_tsne_projection(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    perplexity: int = 30,
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
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
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
    n_neighbors: int = 15,
    min_dist: float = 0.1,
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

    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
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
    n_neighbors: int = 10,
    num_iters: int = 50,
    pca_dims: int = 30,
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
    valid = norms.squeeze() > 1e-8
    X_norm = X[valid] / norms[valid]

    # Mean direction
    mean_dir = X_norm.mean(axis=0)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-8)

    # Angles from mean
    cos_angles = X_norm @ mean_dir
    angles = np.degrees(np.arccos(np.clip(cos_angles, -1, 1)))

    # Project to 2D: mean direction and orthogonal
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
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


def create_summary_figure(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    metrics: Dict[str, Any] = None,
    include_pacmap: bool = True,
) -> str:
    """
    Create a multi-panel summary figure and return as base64.

    Creates a 3x3 grid with all 9 visualizations:
    Row 1: PCA, t-SNE, UMAP
    Row 2: PaCMAP, Diff Vectors, Alignment Distribution
    Row 3: Eigenvalue Spectrum, Norm Distribution, Pairwise Distances

    Returns:
        Base64-encoded PNG string
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io
    import base64

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    def plot_projection(ax, data, title):
        """Helper to plot a 2D projection."""
        if "error" in data:
            ax.text(0.5, 0.5, f"Error: {data['error']}", ha='center', va='center', transform=ax.transAxes)
        else:
            pos = data["pos_projected"]
            neg = data["neg_projected"]
            ax.scatter(pos[:, 0], pos[:, 1], c='blue', alpha=0.6, label='Pos', s=20)
            ax.scatter(neg[:, 0], neg[:, 1], c='red', alpha=0.6, label='Neg', s=20)
            ax.legend(loc='upper right', fontsize=8)
        ax.set_title(title, fontsize=11)

    # Row 1: Dimensionality reduction projections
    # 1. PCA projection
    pca_data = plot_pca_projection(pos_activations, neg_activations)
    plot_projection(axes[0, 0], pca_data, "PCA Projection")

    # 2. t-SNE projection
    tsne_data = plot_tsne_projection(pos_activations, neg_activations)
    plot_projection(axes[0, 1], tsne_data, "t-SNE Projection")

    # 3. UMAP projection
    umap_data = plot_umap_projection(pos_activations, neg_activations)
    plot_projection(axes[0, 2], umap_data, "UMAP Projection")

    # Row 2: More projections and distributions
    # 4. PaCMAP projection
    if include_pacmap:
        try:
            from .pacmap_alt import plot_pacmap_alt
            pacmap_data = plot_pacmap_alt(pos_activations, neg_activations)
        except Exception as e:
            pacmap_data = {"error": str(e)}
    else:
        pacmap_data = {"error": "PaCMAP disabled"}
    plot_projection(axes[1, 0], pacmap_data, "PaCMAP Projection")

    # 5. Diff vectors
    diff_data = plot_diff_vectors(pos_activations, neg_activations)
    ax = axes[1, 1]
    if "error" not in diff_data:
        diffs = diff_data["diffs_projected"]
        mean = diff_data["mean_diff_projected"]
        ax.scatter(diffs[:, 0], diffs[:, 1], c='green', alpha=0.6, s=20)
        scale = max(abs(mean[0]), abs(mean[1]), 0.1) * 0.8
        ax.arrow(0, 0, mean[0]*0.8, mean[1]*0.8, head_width=scale*0.1, head_length=scale*0.05, fc='black', ec='black')
    ax.set_title("Diff Vectors (Mean Direction)", fontsize=11)

    # 6. Alignment distribution
    align_data = plot_alignment_distribution(pos_activations, neg_activations)
    ax = axes[1, 2]
    if "error" not in align_data:
        ax.hist(align_data["alignments"], bins=20, color='green', alpha=0.7)
        ax.axvline(align_data["mean"], color='black', linestyle='--', linewidth=2)
        ax.set_title(f"Alignment (mean={align_data['mean']:.2f})", fontsize=11)
    else:
        ax.text(0.5, 0.5, "Error", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Alignment Distribution", fontsize=11)
    ax.set_xlabel("Cosine Alignment")
    ax.set_ylabel("Count")

    # Row 3: Spectral and distance analysis
    # 7. Eigenvalue spectrum
    eigen_data = plot_eigenvalue_spectrum(pos_activations, neg_activations)
    ax = axes[2, 0]
    n_show = min(20, len(eigen_data["explained_variance_ratio"]))
    ax.bar(range(n_show), eigen_data["explained_variance_ratio"][:n_show], color='purple', alpha=0.7)
    ax.set_title("Eigenvalue Spectrum", fontsize=11)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Ratio")

    # 8. Norm distribution
    norm_data = plot_norm_distribution(pos_activations, neg_activations)
    ax = axes[2, 1]
    ax.hist(norm_data["pos_norms"], bins=20, alpha=0.5, label='Pos', color='blue')
    ax.hist(norm_data["neg_norms"], bins=20, alpha=0.5, label='Neg', color='red')
    ax.legend(fontsize=8)
    ax.set_title("Norm Distribution", fontsize=11)
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Count")

    # 9. Pairwise distances
    dist_data = plot_pairwise_distances(pos_activations, neg_activations)
    ax = axes[2, 2]
    if "error" not in dist_data:
        im = ax.imshow(dist_data["distance_matrix"], cmap='viridis', aspect='auto')
        n_pos = dist_data.get("n_pos", len(pos_activations))
        ax.axhline(n_pos - 0.5, color='white', linestyle='--', linewidth=1)
        ax.axvline(n_pos - 0.5, color='white', linestyle='--', linewidth=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Pairwise Distances", fontsize=11)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Sample Index")

    # Add overall title with key metrics if available
    if metrics:
        # Include concept title if available
        title_prefix = metrics.get('concept_title', 'Layer Analysis')
        suptitle = f"{title_prefix} | Linear Probe: {metrics.get('linear_probe_accuracy', 0):.2f} | ICD: {metrics.get('icd_icd', 0):.1f} | Rec: {metrics.get('recommended_method', 'N/A')}"
        fig.suptitle(suptitle, fontsize=14, y=1.01)

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return img_base64


def render_matplotlib_figure(
    plot_data: Dict[str, Any],
    plot_type: str = None,
    figsize: Tuple[int, int] = (8, 6),
    return_base64: bool = True,
) -> str:
    """
    Render plot data to a matplotlib figure and return as base64.

    Args:
        plot_data: Output from one of the plot_* functions
        plot_type: One of 'pca', 'diff_vectors', 'norms', 'alignments',
                   'eigenvalues', 'distances', 'tsne', 'umap', 'cone'.
                   If None, auto-detect from plot_data.
        figsize: Figure size
        return_base64: If True, return base64 string; if False, return figure

    Returns:
        Base64-encoded PNG string (or figure if return_base64=False)
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import io
    import base64

    # Auto-detect plot type from data
    if plot_type is None:
        if "pos_projected" in plot_data and "neg_projected" in plot_data:
            plot_type = "pca"
        elif "diffs_projected" in plot_data:
            plot_type = "diff_vectors"
        elif "pos_norms" in plot_data:
            plot_type = "norms"
        elif "alignments" in plot_data:
            plot_type = "alignments"
        elif "explained_variance_ratio" in plot_data:
            plot_type = "eigenvalues"
        elif "pos_within" in plot_data:
            plot_type = "distances"
        elif "angles_from_mean" in plot_data:
            plot_type = "cone"
        else:
            plot_type = "unknown"

    fig, ax = plt.subplots(figsize=figsize)

    if "error" in plot_data:
        ax.text(0.5, 0.5, f"Error: {plot_data['error']}", ha='center', va='center')
    else:
        title = plot_data.get("title", plot_type)

        if plot_type in ['pca', 'tsne', 'umap']:
            pos = plot_data["pos_projected"]
            neg = plot_data["neg_projected"]
            ax.scatter(pos[:, 0], pos[:, 1], c='blue', alpha=0.6, label='Positive')
            ax.scatter(neg[:, 0], neg[:, 1], c='red', alpha=0.6, label='Negative')
            ax.legend()
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

        elif plot_type == 'diff_vectors':
            diffs = plot_data["diffs_projected"]
            mean = plot_data["mean_diff_projected"]
            ax.scatter(diffs[:, 0], diffs[:, 1], c='green', alpha=0.6)
            ax.arrow(0, 0, mean[0], mean[1], head_width=0.1, head_length=0.05, fc='black', ec='black')
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

        elif plot_type == 'norms':
            ax.hist(plot_data["pos_norms"], bins=20, alpha=0.5, label='Positive', color='blue')
            ax.hist(plot_data["neg_norms"], bins=20, alpha=0.5, label='Negative', color='red')
            ax.legend()
            ax.set_xlabel("Norm")
            ax.set_ylabel("Count")

        elif plot_type == 'alignments':
            ax.hist(plot_data["alignments"], bins=20, color='green', alpha=0.7)
            ax.axvline(plot_data["mean"], color='black', linestyle='--', label=f'Mean: {plot_data["mean"]:.2f}')
            ax.legend()
            ax.set_xlabel("Alignment with mean direction")
            ax.set_ylabel("Count")

        elif plot_type == 'eigenvalues':
            ax.bar(range(len(plot_data["explained_variance_ratio"])), plot_data["explained_variance_ratio"])
            ax.set_xlabel("Principal Component")
            ax.set_ylabel("Explained Variance Ratio")

        elif plot_type == 'distances':
            ax.hist(plot_data["pos_within"], bins=30, alpha=0.5, label='Within Pos', color='blue')
            ax.hist(plot_data["neg_within"], bins=30, alpha=0.5, label='Within Neg', color='red')
            ax.hist(plot_data["between"], bins=30, alpha=0.5, label='Between', color='green')
            ax.legend()
            ax.set_xlabel("Distance")
            ax.set_ylabel("Count")

        elif plot_type == 'cone':
            ax.hist(plot_data["angles_from_mean"], bins=20, color='purple', alpha=0.7)
            ax.axvline(plot_data["mean_angle"], color='black', linestyle='--',
                       label=f'Mean: {plot_data["mean_angle"]:.1f}°')
            ax.legend()
            ax.set_xlabel("Angle from mean direction (degrees)")
            ax.set_ylabel("Count")

        ax.set_title(title)

    plt.tight_layout()

    if not return_base64:
        return fig

    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return img_base64
