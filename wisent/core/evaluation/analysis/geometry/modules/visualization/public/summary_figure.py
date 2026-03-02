"""Create combined summary figures with all visualizations."""
import numpy as np
import torch
from typing import Dict, Any
from wisent.core import constants as _C

from .visualizations import (
    plot_pca_projection,
    plot_tsne_projection,
    plot_umap_projection,
    plot_diff_vectors,
    plot_alignment_distribution,
    plot_eigenvalue_spectrum,
    plot_norm_distribution,
    plot_pairwise_distances,
)
from .pacmap_alt import plot_pacmap_alt


def create_full_summary_figure(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    metrics: Dict[str, Any] = None,
    layer_num: int = None,
) -> str:
    """
    Create a 3x3 grid summary figure with all 9 visualizations.

    Layout:
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

    fig, axes = plt.subplots(_C.VIZ_GRID_SUMMARY_ROWS, _C.VIZ_GRID_SUMMARY_COLS, figsize=_C.VIZ_FIGSIZE_SUMMARY)

    def plot_projection(ax, data, title):
        """Helper to plot a 2D projection."""
        if "error" in data:
            ax.text(0.5, 0.5, f"Error: {data['error']}", ha='center', va='center', transform=ax.transAxes)
        else:
            pos = data["pos_projected"]
            neg = data["neg_projected"]
            ax.scatter(pos[:, 0], pos[:, 1], c='blue', alpha=_C.VIZ_ALPHA_MEDIUM, label='Pos', s=_C.VIZ_MARKER_SIZE_XS)
            ax.scatter(neg[:, 0], neg[:, 1], c='red', alpha=_C.VIZ_ALPHA_MEDIUM, label='Neg', s=_C.VIZ_MARKER_SIZE_XS)
            ax.legend(loc='upper right', fontsize=_C.VIZ_FONTSIZE_SMALL)
        ax.set_title(title, fontsize=_C.VIZ_FONTSIZE_TITLE)

    # Row 1: Dimensionality reduction projections
    pca_data = plot_pca_projection(pos_activations, neg_activations)
    plot_projection(axes[0, 0], pca_data, "PCA Projection")

    tsne_data = plot_tsne_projection(pos_activations, neg_activations)
    plot_projection(axes[0, 1], tsne_data, "t-SNE Projection")

    umap_data = plot_umap_projection(pos_activations, neg_activations)
    plot_projection(axes[0, 2], umap_data, "UMAP Projection")

    # Row 2: PaCMAP, diff vectors, alignment
    try:
        pacmap_data = plot_pacmap_alt(pos_activations, neg_activations)
    except Exception as e:
        pacmap_data = {"error": str(e)}
    plot_projection(axes[1, 0], pacmap_data, "PaCMAP Projection")

    diff_data = plot_diff_vectors(pos_activations, neg_activations)
    ax = axes[1, 1]
    if "error" not in diff_data:
        diffs = diff_data["diffs_projected"]
        mean = diff_data["mean_diff_projected"]
        ax.scatter(diffs[:, 0], diffs[:, 1], c='green', alpha=_C.VIZ_ALPHA_MEDIUM, s=_C.VIZ_MARKER_SIZE_XS)
        scale = max(abs(mean[0]), abs(mean[1]), _C.VIZ_MIN_SCALE) * _C.VIZ_SCALE_PADDING
        ax.arrow(0, 0, mean[0]*_C.VIZ_MOVEMENT_SCALE, mean[1]*_C.VIZ_MOVEMENT_SCALE, head_width=scale*_C.VIZ_ARROW_HEAD_LENGTH, head_length=scale*_C.VIZ_ARROW_HEAD_WIDTH, fc='black', ec='black')
    ax.set_title("Diff Vectors (Mean Direction)", fontsize=_C.VIZ_FONTSIZE_TITLE)

    align_data = plot_alignment_distribution(pos_activations, neg_activations)
    ax = axes[1, 2]
    if "error" not in align_data:
        ax.hist(align_data["alignments"], bins=_C.VIZ_HISTOGRAM_BINS_20, color='green', alpha=_C.VIZ_ALPHA_HIGH)
        ax.axvline(align_data["mean"], color='black', linestyle='--', linewidth=_C.VIZ_LINEWIDTH_NORMAL)
        ax.set_title(f"Alignment (mean={align_data['mean']:.2f})", fontsize=_C.VIZ_FONTSIZE_TITLE)
    else:
        ax.text(0.5, 0.5, "Error", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Alignment Distribution", fontsize=_C.VIZ_FONTSIZE_TITLE)
    ax.set_xlabel("Cosine Alignment")
    ax.set_ylabel("Count")

    # Row 3: Spectral and distance analysis
    eigen_data = plot_eigenvalue_spectrum(pos_activations, neg_activations)
    ax = axes[2, 0]
    n_show = min(_C.EIGENVALUE_DISPLAY_LIMIT, len(eigen_data["explained_variance_ratio"]))
    ax.bar(range(n_show), eigen_data["explained_variance_ratio"][:n_show], color='purple', alpha=_C.VIZ_ALPHA_HIGH)
    ax.set_title("Eigenvalue Spectrum", fontsize=_C.VIZ_FONTSIZE_TITLE)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Ratio")

    norm_data = plot_norm_distribution(pos_activations, neg_activations)
    ax = axes[2, 1]
    ax.hist(norm_data["pos_norms"], bins=_C.VIZ_HISTOGRAM_BINS_20, alpha=_C.VIZ_HISTOGRAM_ALPHA, label='Pos', color='blue')
    ax.hist(norm_data["neg_norms"], bins=_C.VIZ_HISTOGRAM_BINS_20, alpha=_C.VIZ_HISTOGRAM_ALPHA, label='Neg', color='red')
    ax.legend(fontsize=_C.VIZ_FONTSIZE_SMALL)
    ax.set_title("Norm Distribution", fontsize=_C.VIZ_FONTSIZE_TITLE)
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Count")

    dist_data = plot_pairwise_distances(pos_activations, neg_activations)
    ax = axes[2, 2]
    if "error" not in dist_data:
        im = ax.imshow(dist_data["distance_matrix"], cmap='viridis', aspect='auto')
        n_pos = dist_data.get("n_pos", len(pos_activations))
        ax.axhline(n_pos - 0.5, color='white', linestyle='--', linewidth=_C.VIZ_LINEWIDTH_HAIRLINE)
        ax.axvline(n_pos - 0.5, color='white', linestyle='--', linewidth=_C.VIZ_LINEWIDTH_HAIRLINE)
        plt.colorbar(im, ax=ax, fraction=_C.VIZ_COLORBAR_FRACTION, pad=_C.VIZ_COLORBAR_PAD)
    ax.set_title("Pairwise Distances", fontsize=_C.VIZ_FONTSIZE_TITLE)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Sample Index")

    # Overall title
    title_parts = []
    if layer_num is not None:
        title_parts.append(f"Layer {layer_num}")
    if metrics:
        title_parts.append(f"Linear: {metrics.get('linear_probe_accuracy', 0):.2f}")
        title_parts.append(f"ICD: {metrics.get('icd_icd', 0):.1f}")
        title_parts.append(f"Rec: {metrics.get('recommended_method', 'N/A')}")
    if title_parts:
        fig.suptitle(" | ".join(title_parts), fontsize=_C.VIZ_FONTSIZE_SUPTITLE, y=_C.VIZ_SUPTITLE_Y_OFFSET)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=_C.VIZ_DPI_STANDARD, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return img_base64
