"""
Visualization generation for geometry metrics.

This module contains the generate_metrics_visualizations function that
orchestrates all visualization generation for compute_geometry_metrics.
"""

from typing import Dict, Any
import torch

from ...visualization.visualizations import (
    plot_pca_projection,
    plot_diff_vectors,
    plot_alignment_distribution,
    plot_eigenvalue_spectrum,
    plot_tsne_projection,
    plot_umap_projection,
    plot_pacmap_projection,
    plot_norm_distribution,
    plot_pairwise_distances,
    create_summary_figure,
    render_matplotlib_figure,
)


def generate_metrics_visualizations(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    metrics: Dict[str, Any],
) -> Dict[str, str]:
    """
    Generate all visualization figures for geometry metrics.

    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        metrics: Current metrics dict (for summary figure)

    Returns:
        Dict mapping visualization name to base64 encoded PNG
    """
    visualizations = {}

    # PCA projection
    try:
        pca_data = plot_pca_projection(pos_activations, neg_activations)
        visualizations["pca_projection"] = render_matplotlib_figure(pca_data)
    except Exception:
        pass

    # t-SNE projection
    try:
        tsne_data = plot_tsne_projection(pos_activations, neg_activations)
        if "error" not in tsne_data:
            visualizations["tsne_projection"] = render_matplotlib_figure(tsne_data)
    except Exception:
        pass

    # UMAP projection
    try:
        umap_data = plot_umap_projection(pos_activations, neg_activations)
        if "error" not in umap_data:
            visualizations["umap_projection"] = render_matplotlib_figure(umap_data)
    except Exception:
        pass

    # PaCMAP projection
    try:
        pacmap_data = plot_pacmap_projection(pos_activations, neg_activations)
        if "error" not in pacmap_data:
            visualizations["pacmap_projection"] = render_matplotlib_figure(pacmap_data)
    except Exception:
        pass

    # Diff vectors
    try:
        diff_data = plot_diff_vectors(pos_activations, neg_activations)
        visualizations["diff_vectors"] = render_matplotlib_figure(diff_data)
    except Exception:
        pass

    # Alignment distribution
    try:
        align_data = plot_alignment_distribution(pos_activations, neg_activations)
        visualizations["alignment_distribution"] = render_matplotlib_figure(align_data)
    except Exception:
        pass

    # Eigenvalue spectrum
    try:
        eigen_data = plot_eigenvalue_spectrum(pos_activations, neg_activations)
        visualizations["eigenvalue_spectrum"] = render_matplotlib_figure(eigen_data)
    except Exception:
        pass

    # Norm distribution
    try:
        norm_data = plot_norm_distribution(pos_activations, neg_activations)
        visualizations["norm_distribution"] = render_matplotlib_figure(norm_data)
    except Exception:
        pass

    # Pairwise distances
    try:
        dist_data = plot_pairwise_distances(pos_activations, neg_activations)
        visualizations["pairwise_distances"] = render_matplotlib_figure(dist_data)
    except Exception:
        pass

    # Summary figure (returns base64 directly)
    try:
        visualizations["summary"] = create_summary_figure(pos_activations, neg_activations, metrics)
    except Exception:
        pass

    return visualizations
