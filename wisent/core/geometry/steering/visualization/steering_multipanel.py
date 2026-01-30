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


def create_interactive_steering_figure(
    pos_activations: torch.Tensor, neg_activations: torch.Tensor,
    base_activations: torch.Tensor, steered_activations: torch.Tensor,
    title: str = "Interactive Steering Effect Visualization",
    base_evaluations: List[str] = None, steered_evaluations: List[str] = None,
    base_space_probs: List[float] = None, steered_space_probs: List[float] = None,
    prompts: List[str] = None, base_responses: List[str] = None, steered_responses: List[str] = None,
) -> str:
    """
    Create interactive HTML visualization with hover information using Plotly.
    Hovering over any point shows: prompt text, response text, evaluation, probability.
    Returns HTML string that can be saved to a file.
    """
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA

    def to_numpy(t):
        if hasattr(t, 'numpy'):
            return t.float().cpu().numpy()
        return np.asarray(t, dtype=np.float32)

    pos, neg = to_numpy(pos_activations), to_numpy(neg_activations)
    base, steered = to_numpy(base_activations), to_numpy(steered_activations)
    reference = np.vstack([pos, neg])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(reference)
    pos_2d, neg_2d = pca.transform(pos), pca.transform(neg)
    base_2d, steered_2d = pca.transform(base), pca.transform(steered)
    pos_centroid, neg_centroid = pos_2d.mean(axis=0), neg_2d.mean(axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pos_2d[:, 0], y=pos_2d[:, 1], mode='markers',
        marker=dict(color='blue', size=8, opacity=0.3), name='Positive (truthful)',
        hovertext=[f"Positive reference #{i}" for i in range(len(pos_2d))], hoverinfo='text'))
    fig.add_trace(go.Scatter(x=neg_2d[:, 0], y=neg_2d[:, 1], mode='markers',
        marker=dict(color='red', size=8, opacity=0.3), name='Negative (untruthful)',
        hovertext=[f"Negative reference #{i}" for i in range(len(neg_2d))], hoverinfo='text'))
    fig.add_trace(go.Scatter(x=[pos_centroid[0]], y=[pos_centroid[1]], mode='markers',
        marker=dict(color='blue', size=20, symbol='star', line=dict(color='black', width=1)),
        name='Positive centroid', hovertext=['Positive centroid'], hoverinfo='text'))
    fig.add_trace(go.Scatter(x=[neg_centroid[0]], y=[neg_centroid[1]], mode='markers',
        marker=dict(color='red', size=20, symbol='star', line=dict(color='black', width=1)),
        name='Negative centroid', hovertext=['Negative centroid'], hoverinfo='text'))

    base_hover = _build_hover_texts(len(base_2d), "Base", prompts, base_responses, base_evaluations, base_space_probs)
    steered_hover = _build_hover_texts(len(steered_2d), "Steered", prompts, steered_responses, steered_evaluations, steered_space_probs)
    base_colors = _get_marker_colors(base_evaluations, len(base_2d))
    steered_colors = _get_marker_colors(steered_evaluations, len(steered_2d))

    fig.add_trace(go.Scatter(x=base_2d[:, 0], y=base_2d[:, 1], mode='markers',
        marker=dict(color='gray', size=12, symbol='circle', line=dict(color=base_colors, width=3)),
        name='Base (no steering)', hovertext=base_hover, hoverinfo='text'))
    fig.add_trace(go.Scatter(x=steered_2d[:, 0], y=steered_2d[:, 1], mode='markers',
        marker=dict(color='lime', size=12, symbol='square', line=dict(color=steered_colors, width=3)),
        name='Steered', hovertext=steered_hover, hoverinfo='text'))

    for i in range(min(len(base_2d), len(steered_2d))):
        fig.add_annotation(x=steered_2d[i, 0], y=steered_2d[i, 1], ax=base_2d[i, 0], ay=base_2d[i, 1],
            xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2,
            arrowsize=1, arrowwidth=1.5, arrowcolor='green', opacity=0.6)

    metrics = _compute_interactive_metrics(base_2d, steered_2d, pos_centroid, neg_centroid,
                                           base_evaluations, steered_evaluations, base_space_probs, steered_space_probs)
    fig.update_layout(title=dict(text=f"{title}<br><sub>{metrics}</sub>", x=0.5),
        xaxis_title="PCA Component 1", yaxis_title="PCA Component 2", hovermode='closest',
        showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99), width=1000, height=800)
    return fig.to_html(full_html=True, include_plotlyjs=True)


def _build_hover_texts(n, prefix, prompts, responses, evals, probs):
    """Build hover text list for Plotly."""
    hover = []
    for i in range(n):
        parts = [f"<b>{prefix} Sample #{i+1}</b>"]
        if prompts and i < len(prompts):
            parts.append(f"<b>Prompt:</b> {prompts[i][:100]}..." if len(prompts[i]) > 100 else f"<b>Prompt:</b> {prompts[i]}")
        if responses and i < len(responses):
            parts.append(f"<b>Response:</b> {responses[i][:150]}..." if len(responses[i]) > 150 else f"<b>Response:</b> {responses[i]}")
        if evals and i < len(evals):
            parts.append(f"<b>Evaluation:</b> {evals[i]}")
        if probs and i < len(probs):
            parts.append(f"<b>P(truthful):</b> {probs[i]:.3f}")
        hover.append("<br>".join(parts))
    return hover


def _get_marker_colors(evals, n):
    """Get marker edge colors based on evaluations."""
    if evals is None:
        return ['black'] * n
    return ['green' if i < len(evals) and evals[i] == "TRUTHFUL" else 'red' for i in range(n)]


def _compute_interactive_metrics(base_2d, steered_2d, pos_c, neg_c, base_evals, steered_evals, base_probs, steered_probs):
    """Compute summary metrics string for interactive figure title."""
    base_to_pos = np.linalg.norm(base_2d - pos_c, axis=1)
    steered_to_pos = np.linalg.norm(steered_2d - pos_c, axis=1)
    moved = np.sum(steered_to_pos < base_to_pos)
    total = len(base_2d)
    parts = [f"Moved toward pos: {moved}/{total}"]
    if base_evals and steered_evals:
        bt = sum(1 for e in base_evals if e == "TRUTHFUL")
        st = sum(1 for e in steered_evals if e == "TRUTHFUL")
        parts.append(f"Text: {bt}->{st} truthful")
    if base_probs and steered_probs:
        bit = sum(1 for p in base_probs if p >= 0.5)
        sit = sum(1 for p in steered_probs if p >= 0.5)
        parts.append(f"Activation: {bit}->{sit} in truthful")
    return " | ".join(parts)
