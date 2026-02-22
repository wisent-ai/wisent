"""Heatmap visualizations for concept analysis."""

import numpy as np
from typing import Dict, Any, List, Optional
import io
import base64


def create_layer_accuracy_heatmap(
    concepts: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Create heatmap showing linear separability per concept per layer.

    Returns:
        Base64-encoded PNG string, or None if no layer data
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Check if concepts have layer_accuracies
    if not concepts or "layer_accuracies" not in concepts[0]:
        return None

    # Build matrix
    all_layers = set()
    for c in concepts:
        all_layers.update(c.get("layer_accuracies", {}).keys())

    if not all_layers:
        return None

    layers = sorted(all_layers)
    n_concepts = len(concepts)

    matrix = np.zeros((n_concepts, len(layers)))
    for i, concept in enumerate(concepts):
        layer_accs = concept.get("layer_accuracies", {})
        for j, layer in enumerate(layers):
            matrix[i, j] = layer_accs.get(layer, 0.5)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.5), max(4, n_concepts * 0.8)))

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("Layer")

    concept_names = [c.get("name", f"Concept {c['id']}") for c in concepts]
    ax.set_yticks(range(n_concepts))
    ax.set_yticklabels(concept_names)
    ax.set_ylabel("Concept")

    # Mark optimal layer per concept
    for i, concept in enumerate(concepts):
        opt_layer = concept.get("optimal_layer")
        if opt_layer in layers:
            j = layers.index(opt_layer)
            ax.scatter(j, i, marker='*', s=200, c='black', zorder=10)

    plt.colorbar(im, ax=ax, label="Linear Accuracy")
    ax.set_title("Linear Separability by Concept and Layer\n(* = optimal layer)")

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return base64.b64encode(buf.read()).decode('utf-8')


def create_inter_concept_similarity_heatmap(
    inter_concept_similarity: Dict[str, Any],
    concept_names: List[str],
) -> Optional[str]:
    """
    Create heatmap of inter-concept centroid similarities.

    Returns:
        Base64-encoded PNG string
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    matrix = inter_concept_similarity.get("matrix")
    if matrix is None:
        return None

    matrix = np.array(matrix)
    n_concepts = len(matrix)

    fig, ax = plt.subplots(figsize=(max(6, n_concepts * 1.2), max(5, n_concepts)))

    im = ax.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)

    ax.set_xticks(range(n_concepts))
    ax.set_xticklabels(concept_names, rotation=45, ha='right')
    ax.set_yticks(range(n_concepts))
    ax.set_yticklabels(concept_names)

    # Add text annotations
    for i in range(n_concepts):
        for j in range(n_concepts):
            color = 'white' if abs(matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title("Inter-Concept Similarity\n(centroid cosine similarity)")

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return base64.b64encode(buf.read()).decode('utf-8')
