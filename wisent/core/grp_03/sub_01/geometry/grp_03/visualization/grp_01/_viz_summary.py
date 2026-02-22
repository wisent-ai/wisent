"""Summary figure creation and rendering."""
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import warnings
from wisent.core.geometry.visualization._viz_basic import (
    plot_pca_projection, plot_diff_vectors, plot_norm_distribution,
    plot_alignment_distribution, plot_eigenvalue_spectrum, plot_pairwise_distances,
)
from wisent.core.geometry.visualization._viz_projections import (
    plot_tsne_projection, plot_umap_projection, plot_pacmap_projection,
    plot_cone_visualization, plot_layer_comparison,
)

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
