"""
Visualization generation for geometry metrics.

This module contains the generate_metrics_visualizations function that
orchestrates all visualization generation for compute_geometry_metrics.
"""

import logging
from typing import Dict, Any
import torch

from wisent.core.utils.visualization.geometry.public.visualizations import (
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

logger = logging.getLogger(__name__)


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

    viz_funcs = [
        ("pca_projection", lambda: render_matplotlib_figure(plot_pca_projection(pos_activations, neg_activations))),
        ("tsne_projection", lambda: render_matplotlib_figure(plot_tsne_projection(pos_activations, neg_activations, title="t-SNE Projection"))),
        ("umap_projection", lambda: render_matplotlib_figure(plot_umap_projection(pos_activations, neg_activations))),
        ("pacmap_projection", lambda: render_matplotlib_figure(plot_pacmap_projection(pos_activations, neg_activations))),
        ("diff_vectors", lambda: render_matplotlib_figure(plot_diff_vectors(pos_activations, neg_activations))),
        ("alignment_distribution", lambda: render_matplotlib_figure(plot_alignment_distribution(pos_activations, neg_activations))),
        ("eigenvalue_spectrum", lambda: render_matplotlib_figure(plot_eigenvalue_spectrum(pos_activations, neg_activations))),
        ("norm_distribution", lambda: render_matplotlib_figure(plot_norm_distribution(pos_activations, neg_activations))),
        ("pairwise_distances", lambda: render_matplotlib_figure(plot_pairwise_distances(pos_activations, neg_activations))),
        ("summary", lambda: create_summary_figure(pos_activations, neg_activations, metrics)),
    ]

    for name, func in viz_funcs:
        try:
            result = func()
            if result and "error" not in (result if isinstance(result, dict) else {}):
                visualizations[name] = result
        except Exception as e:
            logger.warning("Visualization '%s' failed: %s", name, e)

    return visualizations
