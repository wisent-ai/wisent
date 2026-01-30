"""Steering effect visualizations - shows base vs steered activations."""

import numpy as np
import torch
from typing import List, Optional
import io
import base64
from sklearn.decomposition import PCA


def create_steering_effect_figure(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    base_activations: torch.Tensor,
    steered_activations: torch.Tensor,
    title: str = "Steering Effect Visualization",
    base_evaluations: List[str] = None,
    steered_evaluations: List[str] = None,
    base_space_probs: List[float] = None,
    steered_space_probs: List[float] = None,
    classifier=None,
) -> str:
    """
    Create visualization showing where base and steered samples land relative to pos/neg.
    PCA is fitted on pos+neg reference data, then base and steered points transformed.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pos, neg, base, steered = _prepare_arrays(pos_activations, neg_activations,
                                               base_activations, steered_activations)
    reference = np.vstack([pos, neg])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(reference)

    pos_2d, neg_2d = pca.transform(pos), pca.transform(neg)
    base_2d, steered_2d = pca.transform(base), pca.transform(steered)
    pos_centroid, neg_centroid = pos_2d.mean(axis=0), neg_2d.mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 10))

    if base_space_probs is not None and steered_space_probs is not None:
        _draw_decision_boundary(ax, pos_2d, neg_2d)

    _plot_scatter_points(ax, pos_2d, neg_2d, base_2d, steered_2d, pos_centroid, neg_centroid,
                         base_evaluations, steered_evaluations)

    for i in range(len(base_2d)):
        ax.annotate('', xy=(steered_2d[i, 0], steered_2d[i, 1]),
                    xytext=(base_2d[i, 0], base_2d[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.6, lw=1.5))

    metrics_text = _compute_metrics_text(base_2d, steered_2d, pos_centroid, neg_centroid,
                                         base_evaluations, steered_evaluations,
                                         base_space_probs, steered_space_probs)
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_per_concept_steering_figure(
    concept_name: str, concept_id: int,
    pos_activations: torch.Tensor, neg_activations: torch.Tensor,
    base_activations: torch.Tensor, steered_activations: torch.Tensor,
    base_evaluations: List[str], steered_evaluations: List[str],
    layer: int, strength: float,
    base_space_probs: List[float] = None, steered_space_probs: List[float] = None,
) -> str:
    """Create steering visualization for a single concept."""
    title = f"Concept {concept_id}: {concept_name} (layer {layer}, strength {strength})"
    return create_steering_effect_figure(
        pos_activations=pos_activations, neg_activations=neg_activations,
        base_activations=base_activations, steered_activations=steered_activations,
        title=title, base_evaluations=base_evaluations, steered_evaluations=steered_evaluations,
        base_space_probs=base_space_probs, steered_space_probs=steered_space_probs,
    )


def _prepare_arrays(pos_act, neg_act, base_act, steered_act):
    """Convert tensors to numpy arrays."""
    def to_numpy(t):
        if hasattr(t, 'numpy'):
            return t.float().cpu().numpy()
        return np.asarray(t, dtype=np.float32)
    return to_numpy(pos_act), to_numpy(neg_act), to_numpy(base_act), to_numpy(steered_act)


def _draw_decision_boundary(ax, pos_2d, neg_2d):
    """Draw classifier decision boundary on matplotlib axis."""
    from sklearn.linear_model import LogisticRegression
    X_2d = np.vstack([pos_2d, neg_2d])
    y_2d = np.concatenate([np.ones(len(pos_2d)), np.zeros(len(neg_2d))])
    clf_2d = LogisticRegression(random_state=42, max_iter=1000)
    clf_2d.fit(X_2d, y_2d)
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = clf_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#FFCCCC', '#CCCCFF'], alpha=0.4)
    ax.contour(xx, yy, Z, levels=[0.5], colors=['black'], linewidths=2, linestyles=['--'])
    ax.plot([], [], 'k--', linewidth=2, label='Decision boundary (P=0.5)')


def _plot_scatter_points(ax, pos_2d, neg_2d, base_2d, steered_2d, pos_c, neg_c, base_evals, steered_evals):
    """Plot all scatter points on matplotlib axis."""
    ax.scatter(pos_2d[:, 0], pos_2d[:, 1], c='blue', alpha=0.3, s=30, label='Positive (truthful)')
    ax.scatter(neg_2d[:, 0], neg_2d[:, 1], c='red', alpha=0.3, s=30, label='Negative (untruthful)')
    ax.scatter([pos_c[0]], [pos_c[1]], c='blue', s=200, marker='*', edgecolors='black', linewidths=1,
               label='Positive centroid', zorder=5)
    ax.scatter([neg_c[0]], [neg_c[1]], c='red', s=200, marker='*', edgecolors='black', linewidths=1,
               label='Negative centroid', zorder=5)
    if base_evals is not None:
        for i, (x, y) in enumerate(base_2d):
            edge = 'green' if i < len(base_evals) and base_evals[i] == "TRUTHFUL" else 'red'
            ax.scatter([x], [y], c='gray', s=80, marker='o', edgecolors=edge, linewidths=2, zorder=4)
        ax.scatter([], [], c='gray', s=80, marker='o', edgecolors='green', linewidths=2, label='Base (TRUTHFUL)')
        ax.scatter([], [], c='gray', s=80, marker='o', edgecolors='red', linewidths=2, label='Base (UNTRUTHFUL)')
    else:
        ax.scatter(base_2d[:, 0], base_2d[:, 1], c='gray', s=80, marker='o', edgecolors='black',
                   linewidths=1, label='Base (no steering)', zorder=4)
    if steered_evals is not None:
        for i, (x, y) in enumerate(steered_2d):
            edge = 'green' if i < len(steered_evals) and steered_evals[i] == "TRUTHFUL" else 'red'
            ax.scatter([x], [y], c='lime', s=80, marker='s', edgecolors=edge, linewidths=2, zorder=4)
        ax.scatter([], [], c='lime', s=80, marker='s', edgecolors='green', linewidths=2, label='Steered (TRUTHFUL)')
        ax.scatter([], [], c='lime', s=80, marker='s', edgecolors='red', linewidths=2, label='Steered (UNTRUTHFUL)')
    else:
        ax.scatter(steered_2d[:, 0], steered_2d[:, 1], c='green', s=80, marker='s', edgecolors='black',
                   linewidths=1, label='Steered', zorder=4)


def _compute_metrics_text(base_2d, steered_2d, pos_c, neg_c, base_evals, steered_evals, base_probs, steered_probs):
    """Compute metrics text for matplotlib figure."""
    base_to_pos = np.linalg.norm(base_2d - pos_c, axis=1)
    steered_to_pos = np.linalg.norm(steered_2d - pos_c, axis=1)
    base_to_neg = np.linalg.norm(base_2d - neg_c, axis=1)
    steered_to_neg = np.linalg.norm(steered_2d - neg_c, axis=1)
    moved_to_pos = np.sum(steered_to_pos < base_to_pos)
    moved_from_neg = np.sum(steered_to_neg > base_to_neg)
    total = len(base_2d)
    avg_imp = np.mean(base_to_pos - steered_to_pos)
    text = (f"Samples moved toward positive: {moved_to_pos}/{total} ({100*moved_to_pos/total:.1f}%)\n"
            f"Samples moved away from negative: {moved_from_neg}/{total} ({100*moved_from_neg/total:.1f}%)\n"
            f"Avg distance improvement to positive: {avg_imp:.2f}")
    if base_evals and steered_evals:
        bt = sum(1 for e in base_evals if e == "TRUTHFUL")
        st = sum(1 for e in steered_evals if e == "TRUTHFUL")
        text += f"\n\nText: Base TRUTHFUL {bt}/{len(base_evals)}, Steered {st}/{len(steered_evals)}"
    if base_probs and steered_probs:
        bit = sum(1 for p in base_probs if p >= 0.5)
        sit = sum(1 for p in steered_probs if p >= 0.5)
        text += f"\nActivation space: Base in truthful {bit}/{len(base_probs)}, Steered {sit}/{len(steered_probs)}"
    return text
