"""
Visualization functions for activation geometry.

Re-exports from split modules for backward compatibility.
"""
from wisent.core.geometry.visualization._viz_basic import (
    plot_pca_projection,
    plot_diff_vectors,
    plot_norm_distribution,
    plot_alignment_distribution,
    plot_eigenvalue_spectrum,
    plot_pairwise_distances,
)
from wisent.core.geometry.visualization._viz_projections import (
    plot_tsne_projection,
    plot_umap_projection,
    plot_pacmap_projection,
    plot_cone_visualization,
    plot_layer_comparison,
)
from wisent.core.geometry.visualization._viz_summary import (
    create_summary_figure,
    render_matplotlib_figure,
)

__all__ = [
    "plot_pca_projection",
    "plot_diff_vectors",
    "plot_norm_distribution",
    "plot_alignment_distribution",
    "plot_eigenvalue_spectrum",
    "plot_pairwise_distances",
    "plot_tsne_projection",
    "plot_umap_projection",
    "plot_pacmap_projection",
    "plot_cone_visualization",
    "plot_layer_comparison",
    "create_summary_figure",
    "render_matplotlib_figure",
]
