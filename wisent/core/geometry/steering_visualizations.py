"""Steering effect visualizations - shows base vs steered activations."""

import numpy as np
import torch
from typing import List, Tuple
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
) -> str:
    """
    Create visualization showing where base and steered samples land relative to pos/neg.

    PCA is fitted on pos+neg reference data, then base and steered points
    are transformed into the same space using .transform().
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def to_numpy(t):
        if hasattr(t, 'numpy'):
            return t.float().cpu().numpy()
        return np.asarray(t, dtype=np.float32)

    pos = to_numpy(pos_activations)
    neg = to_numpy(neg_activations)
    base = to_numpy(base_activations)
    steered = to_numpy(steered_activations)

    # Fit PCA on reference data (pos + neg)
    reference = np.vstack([pos, neg])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(reference)

    # Transform all data into the same space
    pos_2d = pca.transform(pos)
    neg_2d = pca.transform(neg)
    base_2d = pca.transform(base)
    steered_2d = pca.transform(steered)

    pos_centroid = pos_2d.mean(axis=0)
    neg_centroid = neg_2d.mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(pos_2d[:, 0], pos_2d[:, 1], c='blue', alpha=0.3, s=30, label='Positive (reference)')
    ax.scatter(neg_2d[:, 0], neg_2d[:, 1], c='red', alpha=0.3, s=30, label='Negative (reference)')

    ax.scatter([pos_centroid[0]], [pos_centroid[1]], c='blue', s=200, marker='*',
               edgecolors='black', linewidths=1, label='Positive centroid', zorder=5)
    ax.scatter([neg_centroid[0]], [neg_centroid[1]], c='red', s=200, marker='*',
               edgecolors='black', linewidths=1, label='Negative centroid', zorder=5)

    # Plot base points with evaluation coloring if available
    if base_evaluations is not None:
        for i, (x, y) in enumerate(base_2d):
            eval_label = base_evaluations[i] if i < len(base_evaluations) else "UNKNOWN"
            edge_color = 'green' if eval_label == "TRUTHFUL" else 'red'
            ax.scatter([x], [y], c='gray', s=80, marker='o',
                       edgecolors=edge_color, linewidths=2, zorder=4)
        # Add legend entries for evaluation
        ax.scatter([], [], c='gray', s=80, marker='o', edgecolors='green', linewidths=2, label='Base (TRUTHFUL)')
        ax.scatter([], [], c='gray', s=80, marker='o', edgecolors='red', linewidths=2, label='Base (UNTRUTHFUL)')
    else:
        ax.scatter(base_2d[:, 0], base_2d[:, 1], c='gray', s=80, marker='o',
                   edgecolors='black', linewidths=1, label='Base (no steering)', zorder=4)

    # Plot steered points with evaluation coloring if available
    if steered_evaluations is not None:
        for i, (x, y) in enumerate(steered_2d):
            eval_label = steered_evaluations[i] if i < len(steered_evaluations) else "UNKNOWN"
            edge_color = 'green' if eval_label == "TRUTHFUL" else 'red'
            ax.scatter([x], [y], c='lime', s=80, marker='s',
                       edgecolors=edge_color, linewidths=2, zorder=4)
        ax.scatter([], [], c='lime', s=80, marker='s', edgecolors='green', linewidths=2, label='Steered (TRUTHFUL)')
        ax.scatter([], [], c='lime', s=80, marker='s', edgecolors='red', linewidths=2, label='Steered (UNTRUTHFUL)')
    else:
        ax.scatter(steered_2d[:, 0], steered_2d[:, 1], c='green', s=80, marker='s',
                   edgecolors='black', linewidths=1, label='Steered', zorder=4)

    for i in range(len(base_2d)):
        ax.annotate('', xy=(steered_2d[i, 0], steered_2d[i, 1]),
                    xytext=(base_2d[i, 0], base_2d[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.6, lw=1.5))

    base_to_pos = np.linalg.norm(base_2d - pos_centroid, axis=1)
    steered_to_pos = np.linalg.norm(steered_2d - pos_centroid, axis=1)
    base_to_neg = np.linalg.norm(base_2d - neg_centroid, axis=1)
    steered_to_neg = np.linalg.norm(steered_2d - neg_centroid, axis=1)

    moved_to_pos = np.sum(steered_to_pos < base_to_pos)
    moved_away_from_neg = np.sum(steered_to_neg > base_to_neg)
    total_samples = len(base_2d)
    avg_improvement = np.mean(base_to_pos - steered_to_pos)

    metrics_text = (
        f"Samples moved toward positive: {moved_to_pos}/{total_samples} ({100*moved_to_pos/total_samples:.1f}%)\n"
        f"Samples moved away from negative: {moved_away_from_neg}/{total_samples} ({100*moved_away_from_neg/total_samples:.1f}%)\n"
        f"Avg distance improvement to positive: {avg_improvement:.2f}"
    )

    # Add evaluation stats if available
    if base_evaluations is not None and steered_evaluations is not None:
        base_truthful = sum(1 for e in base_evaluations if e == "TRUTHFUL")
        steered_truthful = sum(1 for e in steered_evaluations if e == "TRUTHFUL")
        metrics_text += (
            f"\n\nBase TRUTHFUL: {base_truthful}/{len(base_evaluations)} ({100*base_truthful/len(base_evaluations):.1f}%)\n"
            f"Steered TRUTHFUL: {steered_truthful}/{len(steered_evaluations)} ({100*steered_truthful/len(steered_evaluations):.1f}%)"
        )
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


def extract_base_and_steered_activations(
    wisent,
    prompts: List[str],
    steering_vectors,
    layer: int,
    steering_strength: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract activations before and after steering for a set of prompts."""
    from wisent.core.adapters.base import SteeringConfig

    adapter = wisent.adapter
    layer_name = f"layer.{layer}"

    base_acts = []
    steered_acts = []

    for prompt in prompts:
        base_layer_acts = adapter.extract_activations(prompt, layers=[layer_name])
        base_act = base_layer_acts.get(layer_name)
        if base_act is not None:
            base_acts.append(base_act[0, -1, :])

        steered_act = _extract_with_steering(
            adapter, prompt, layer_name, steering_vectors,
            SteeringConfig(strength=steering_strength)
        )
        if steered_act is not None:
            steered_acts.append(steered_act)

    if not base_acts or not steered_acts:
        raise ValueError("No activations extracted")

    return torch.stack(base_acts), torch.stack(steered_acts)


def _extract_with_steering(adapter, prompt, layer_name, steering_vectors, config):
    """Extract activations from a single forward pass with steering applied."""
    from wisent.core.modalities import TextContent

    content = TextContent(text=prompt) if isinstance(prompt, str) else prompt

    inputs = adapter.tokenizer(content.text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(adapter.model.device) for k, v in inputs.items()}

    activation_storage = {}

    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activation_storage['activation'] = output.detach().cpu()

    all_points = {ip.name: ip for ip in adapter.get_intervention_points()}
    if layer_name not in all_points:
        return None

    ip = all_points[layer_name]
    module = adapter._get_module_by_path(ip.module_path)
    if module is None:
        return None

    try:
        capture_handle = module.register_forward_hook(capture_hook)

        with adapter._steering_hooks(steering_vectors, config):
            with torch.no_grad():
                adapter.model(**inputs)

        capture_handle.remove()

        if 'activation' in activation_storage:
            return activation_storage['activation'][0, -1, :]
        return None

    except Exception as e:
        print(f"Error extracting steered activation: {e}")
        return None


def create_per_concept_steering_figure(
    concept_name: str,
    concept_id: int,
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    base_activations: torch.Tensor,
    steered_activations: torch.Tensor,
    base_evaluations: List[str],
    steered_evaluations: List[str],
    layer: int,
    strength: float,
) -> str:
    """Create steering visualization for a single concept."""
    title = f"Concept {concept_id}: {concept_name} (layer {layer}, strength {strength})"
    return create_steering_effect_figure(
        pos_activations=pos_activations,
        neg_activations=neg_activations,
        base_activations=base_activations,
        steered_activations=steered_activations,
        title=title,
        base_evaluations=base_evaluations,
        steered_evaluations=steered_evaluations,
    )
