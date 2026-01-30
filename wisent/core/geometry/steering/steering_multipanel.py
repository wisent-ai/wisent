"""Steering multi-panel visualizations - 9-panel grid like repscan."""

import numpy as np
import torch
from typing import List, Optional
import io
import base64


def create_steering_multipanel_figure(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    base_activations: torch.Tensor,
    steered_activations: torch.Tensor,
    title: str = "Steering Effect Multi-Panel Visualization",
    base_evaluations: List[str] = None,
    steered_evaluations: List[str] = None,
    base_space_probs: List[float] = None,
    steered_space_probs: List[float] = None,
    extraction_strategy: str = None,
) -> str:
    """
    Create 9-panel steering visualization showing multiple projection methods.

    Layout:
    Row 1: PCA, LDA, t-SNE
    Row 2: UMAP, PCA + Decision Boundary, Movement Vectors
    Row 3: Norm Distribution, Alignment Histogram, Distance to Centroids

    Args:
        extraction_strategy: Where activations were extracted from (e.g., "chat_last")

    Returns:
        Base64-encoded PNG string
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from .steering_panels import (
        plot_pca_panel, plot_lda_panel, plot_tsne_panel, plot_umap_panel,
        plot_pacmap_panel, plot_pca_with_boundary, plot_movement_vectors,
        plot_norm_distribution, plot_alignment_histogram, plot_centroid_distances
    )

    def to_numpy(t):
        if hasattr(t, 'numpy'):
            return t.float().cpu().numpy()
        return np.asarray(t, dtype=np.float32)

    pos = to_numpy(pos_activations)
    neg = to_numpy(neg_activations)
    base = to_numpy(base_activations)
    steered = to_numpy(steered_activations)

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    # Row 1: Dimensionality reduction methods
    plot_pca_panel(axes[0, 0], pos, neg, base, steered, base_evaluations, steered_evaluations)
    plot_tsne_panel(axes[0, 1], pos, neg, base, steered, base_evaluations, steered_evaluations)
    plot_umap_panel(axes[0, 2], pos, neg, base, steered, base_evaluations, steered_evaluations)

    # Row 2: More projections and decision boundary
    plot_pacmap_panel(axes[1, 0], pos, neg, base, steered, base_evaluations, steered_evaluations)
    plot_lda_panel(axes[1, 1], pos, neg, base, steered, base_evaluations, steered_evaluations)
    plot_pca_with_boundary(axes[1, 2], pos, neg, base, steered, base_evaluations, steered_evaluations)

    # Row 3: Statistical analysis
    plot_movement_vectors(axes[2, 0], pos, neg, base, steered)
    plot_norm_distribution(axes[2, 1], pos, neg, base, steered)
    plot_alignment_histogram(axes[2, 2], pos, neg, base, steered)

    # Add summary metrics
    metrics_text = _compute_summary_metrics(
        pos, neg, base, steered,
        base_evaluations, steered_evaluations,
        base_space_probs, steered_space_probs
    )
    # Show extraction source in title
    extraction_info = f"Activations from: GENERATED RESPONSE ({extraction_strategy})" if extraction_strategy else ""
    fig.suptitle(f"{title}\n{metrics_text}\n{extraction_info}", fontsize=12, y=1.02)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return base64.b64encode(buf.read()).decode('utf-8')


def _compute_summary_metrics(pos, neg, base, steered, base_evals, steered_evals,
                              base_probs, steered_probs) -> str:
    """Compute summary metrics string."""
    pos_centroid = pos.mean(axis=0)
    neg_centroid = neg.mean(axis=0)

    base_to_pos = np.linalg.norm(base - pos_centroid, axis=1)
    steered_to_pos = np.linalg.norm(steered - pos_centroid, axis=1)

    moved_toward_pos = np.sum(steered_to_pos < base_to_pos)
    total = len(base)

    parts = [f"Moved toward pos: {moved_toward_pos}/{total} ({100*moved_toward_pos/total:.0f}%)"]

    if base_evals and steered_evals:
        base_truthful = sum(1 for e in base_evals if e == "TRUTHFUL")
        steered_truthful = sum(1 for e in steered_evals if e == "TRUTHFUL")
        parts.append(f"Text TRUTHFUL: {base_truthful}->{steered_truthful}")

    if base_probs and steered_probs:
        base_in_truthful = sum(1 for p in base_probs if p >= 0.5)
        steered_in_truthful = sum(1 for p in steered_probs if p >= 0.5)
        parts.append(f"In truthful region: {base_in_truthful}->{steered_in_truthful}")

        # Diagnostic
        if base_evals and steered_evals:
            activations_shifted = steered_in_truthful > base_in_truthful
            text_improved = sum(1 for e in steered_evals if e == "TRUTHFUL") > sum(1 for e in base_evals if e == "TRUTHFUL")
            if activations_shifted and text_improved:
                diagnostic = "STEERING EFFECTIVE"
            elif activations_shifted and not text_improved:
                diagnostic = "ACTIVATION STEERING IMPROPERLY IDENTIFIED"
            else:
                diagnostic = "STEERING INEFFECTIVE"
            parts.append(f"Diagnostic: {diagnostic}")

    return " | ".join(parts)
