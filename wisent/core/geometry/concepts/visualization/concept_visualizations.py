"""Per-concept visualizations for RepScan - reuses existing visualization infrastructure."""

import numpy as np
import torch
from typing import Dict, Any, List, Optional
import io
import base64

from ...visualization.visualizations import (
    plot_pca_projection,
    plot_pacmap_projection,
    create_summary_figure,
)


def create_concept_overview_figure(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    cluster_labels: np.ndarray,
    concept_names: Optional[List[str]] = None,
) -> str:
    """Create overview figure showing all concepts with different colors."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_pairs = min(len(pos_activations), len(neg_activations))
    pos = pos_activations[:n_pairs]
    neg = neg_activations[:n_pairs]
    diff = (pos - neg).float().cpu().numpy()

    labels = cluster_labels[:n_pairs]
    n_concepts = len(np.unique(labels))

    if concept_names is None:
        concept_names = [f"Concept {i+1}" for i in range(n_concepts)]

    diff_tensor = torch.tensor(diff, dtype=torch.float32)
    zeros = torch.zeros_like(diff_tensor)

    proj_data = plot_pacmap_projection(diff_tensor, zeros, title="Concept Overview (PaCMAP)")
    if "error" in proj_data:
        proj_data = plot_pca_projection(diff_tensor, zeros, title="Concept Overview (PCA)")

    if "error" in proj_data:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f"Error: {proj_data['error']}", ha='center', va='center')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')

    diff_2d = proj_data["pos_projected"]
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_concepts))

    for i in range(n_concepts):
        mask = labels == i
        ax.scatter(diff_2d[mask, 0], diff_2d[mask, 1], c=[colors[i]], label=concept_names[i], alpha=0.7, s=50)

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(proj_data.get("title", "Concept Overview"))
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_per_concept_figure(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    concept_id: int,
    concept_name: str = None,
    coherence: float = None,
    silhouette: float = None,
    intra_similarity: Dict[str, float] = None,
) -> str:
    """Create full 3x3 summary figure for a single concept."""
    n_pairs = min(len(pos_activations), len(neg_activations))

    if n_pairs < 3:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Not enough data", ha='center', va='center')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')

    from .probe_metrics import compute_linear_probe_accuracy
    from .icd import compute_icd

    try:
        lp_acc = compute_linear_probe_accuracy(pos_activations[:n_pairs], neg_activations[:n_pairs])
    except Exception:
        lp_acc = 0.0

    try:
        icd_result = compute_icd(pos_activations[:n_pairs], neg_activations[:n_pairs])
        icd_val = icd_result.get("icd", 0.0) if isinstance(icd_result, dict) else float(icd_result)
    except Exception:
        icd_val = 0.0

    metrics = {
        "linear_probe_accuracy": lp_acc,
        "icd_icd": icd_val,
        "recommended_method": "N/A",
    }

    if coherence is not None:
        metrics["concept_coherence"] = coherence
    if silhouette is not None:
        metrics["silhouette"] = silhouette
    if intra_similarity is not None and "mean" in intra_similarity:
        metrics["intra_similarity"] = intra_similarity["mean"]

    title_parts = [f"Concept {concept_id}"]
    if concept_name:
        title_parts.append(concept_name)
    metrics["concept_title"] = " - ".join(title_parts)
    metrics["n_pairs"] = n_pairs

    return create_summary_figure(pos_activations[:n_pairs], neg_activations[:n_pairs], metrics=metrics, include_pacmap=True)


def create_all_concept_figures(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    cluster_labels: np.ndarray,
    concepts: List[Dict[str, Any]],
    inter_concept_similarity: Dict[str, Any] = None,
) -> Dict[str, str]:
    """Create all concept visualizations."""
    from .concept_heatmaps import create_layer_accuracy_heatmap, create_inter_concept_similarity_heatmap

    n_pairs = min(len(pos_activations), len(neg_activations))
    labels = cluster_labels[:n_pairs]
    concept_names = [c.get("name", f"Concept {c['id']}") for c in concepts]

    result = {}
    result["overview"] = create_concept_overview_figure(pos_activations, neg_activations, labels, concept_names)

    for concept in concepts:
        concept_id = concept["id"]
        idx = concept_id - 1
        mask = labels == idx

        if mask.sum() < 3:
            continue

        pos_concept = pos_activations[:n_pairs][mask]
        neg_concept = neg_activations[:n_pairs][mask]

        result[f"concept_{concept_id}"] = create_per_concept_figure(
            pos_concept, neg_concept, concept_id=concept_id, concept_name=concept.get("name"),
            coherence=concept.get("coherence"), silhouette=concept.get("silhouette"),
            intra_similarity=concept.get("intra_similarity"),
        )

    layer_heatmap = create_layer_accuracy_heatmap(concepts)
    if layer_heatmap:
        result["layer_heatmap"] = layer_heatmap

    if inter_concept_similarity:
        sim_heatmap = create_inter_concept_similarity_heatmap(inter_concept_similarity, concept_names)
        if sim_heatmap:
            result["similarity_heatmap"] = sim_heatmap

    return result
