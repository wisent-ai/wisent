"""Heatmap visualizations for concept analysis."""

import numpy as np
from typing import Dict, Any, List, Optional
import io
import base64

from wisent.core.constants import (
    VIZ_DPI, HEATMAP_TEXT_CONTRAST_THRESHOLD, VIZ_HEATMAP_ASTERISK_SIZE,
    CLASSIFIER_DECISION_THRESHOLD, VIZ_FONTSIZE_ANNOTATION,
    VIZ_FIGSIZE_MIN_WIDTH, VIZ_FIGSIZE_MIN_HEIGHT,
    VIZ_FIGSIZE_HEATMAP_SCALE_W, VIZ_FIGSIZE_HEATMAP_SCALE_H,
    VIZ_FIGSIZE_CONCEPT_MIN_W, VIZ_FIGSIZE_CONCEPT_SCALE,
    VIZ_HEATMAP_ACCURACY_VMIN, VIZ_HEATMAP_ACCURACY_VMAX,
    VIZ_CORRELATION_VMIN, VIZ_CORRELATION_VMAX,
)


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
            matrix[i, j] = layer_accs.get(layer, CLASSIFIER_DECISION_THRESHOLD)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(VIZ_FIGSIZE_MIN_WIDTH, len(layers) * VIZ_FIGSIZE_HEATMAP_SCALE_W), max(VIZ_FIGSIZE_MIN_HEIGHT, n_concepts * VIZ_FIGSIZE_HEATMAP_SCALE_H)))

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=VIZ_HEATMAP_ACCURACY_VMIN, vmax=VIZ_HEATMAP_ACCURACY_VMAX)

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
            ax.scatter(j, i, marker='*', s=VIZ_HEATMAP_ASTERISK_SIZE, c='black', zorder=10)

    plt.colorbar(im, ax=ax, label="Linear Accuracy")
    ax.set_title("Linear Separability by Concept and Layer\n(* = optimal layer)")

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=VIZ_DPI, bbox_inches='tight')
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

    fig, ax = plt.subplots(figsize=(max(VIZ_FIGSIZE_CONCEPT_MIN_W, n_concepts * VIZ_FIGSIZE_CONCEPT_SCALE), max(VIZ_FIGSIZE_MIN_HEIGHT + 1, n_concepts)))

    im = ax.imshow(matrix, cmap='coolwarm', vmin=VIZ_CORRELATION_VMIN, vmax=VIZ_CORRELATION_VMAX)

    ax.set_xticks(range(n_concepts))
    ax.set_xticklabels(concept_names, rotation=45, ha='right')
    ax.set_yticks(range(n_concepts))
    ax.set_yticklabels(concept_names)

    # Add text annotations
    for i in range(n_concepts):
        for j in range(n_concepts):
            color = 'white' if abs(matrix[i, j]) > HEATMAP_TEXT_CONTRAST_THRESHOLD else 'black'
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center', color=color, fontsize=VIZ_FONTSIZE_ANNOTATION)

    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title("Inter-Concept Similarity\n(centroid cosine similarity)")

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=VIZ_DPI, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return base64.b64encode(buf.read()).decode('utf-8')
