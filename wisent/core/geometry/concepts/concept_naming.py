"""Automatic concept naming based on contrastive pair content."""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch


def name_concepts(
    concepts: List[Dict[str, Any]],
    pair_texts: Optional[Dict[int, Dict[str, str]]] = None,
    cluster_labels: Optional[np.ndarray] = None,
    llm_model: str = "Qwen/Qwen3-8B",
) -> List[Dict[str, Any]]:
    """
    Automatically name concepts using LLM.

    Args:
        concepts: List of concept dicts with metrics
        pair_texts: Optional dict mapping pair_id -> {prompt, positive, negative}
        cluster_labels: Optional array mapping pair index -> cluster id
        llm_model: HuggingFace model to use for naming

    Returns:
        Updated concepts with names and descriptions
    """
    if pair_texts is None or cluster_labels is None:
        # No text available, use generic names
        for i, concept in enumerate(concepts):
            concept["name"] = f"concept_{i+1}"
            concept["description"] = "Concept cluster (no text available for naming)"
        return concepts

    # Use LLM naming
    from .llm_concept_naming import name_all_concepts_with_llm
    return name_all_concepts_with_llm(concepts, pair_texts, cluster_labels, model=llm_model)


def find_optimal_layer_per_concept(
    activations_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    cluster_labels: np.ndarray,
    n_concepts: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Find the optimal layer for each concept based on linear separability.

    Different concepts may be best represented at different layers.

    Args:
        activations_by_layer: Dict mapping layer -> (pos_activations, neg_activations)
        cluster_labels: Cluster assignment for each pair
        n_concepts: Number of concepts

    Returns:
        Dict mapping concept_id -> {best_layer, best_accuracy, layer_accuracies}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    results = {}

    for concept_idx in range(n_concepts):
        pair_mask = cluster_labels == concept_idx
        pair_indices = np.where(pair_mask)[0]
        n_pairs = len(pair_indices)

        layer_accuracies = {}
        best_layer = None
        best_acc = 0.0

        for layer, (pos_all, neg_all) in sorted(activations_by_layer.items()):
            # Get activations for this concept's pairs
            if n_pairs < 10:
                continue

            pos_concept = pos_all[pair_indices].float().cpu().numpy()
            neg_concept = neg_all[pair_indices].float().cpu().numpy()

            X = np.vstack([pos_concept, neg_concept])
            y = np.array([1] * len(pos_concept) + [0] * len(neg_concept))

            clf = LogisticRegression(max_iter=500, solver='lbfgs', C=1.0)
            try:
                n_cv = min(5, min(np.sum(y == 0), np.sum(y == 1)))
                if n_cv >= 2:
                    scores = cross_val_score(clf, X, y, cv=n_cv, scoring='accuracy')
                    acc = float(np.mean(scores))
                else:
                    clf.fit(X, y)
                    acc = float(clf.score(X, y))
            except Exception:
                acc = 0.5

            layer_accuracies[layer] = acc

            if acc > best_acc:
                best_acc = acc
                best_layer = layer

        results[concept_idx] = {
            "best_layer": best_layer,
            "best_accuracy": best_acc,
            "layer_accuracies": layer_accuracies,
        }

    return results


def _build_result(diff_normalized, cluster_labels, n_concepts, pair_texts,
                  pos_activations, neg_activations, generate_visualizations, llm_model):
    """Build decomposition result dict from cluster labels."""
    from sklearn.metrics import silhouette_samples
    from sklearn.metrics.pairwise import cosine_similarity

    n_pairs = len(diff_normalized)
    try:
        silhouette_per_sample = silhouette_samples(diff_normalized, cluster_labels)
    except:
        silhouette_per_sample = np.zeros(n_pairs)

    concepts = []
    for i in range(n_concepts):
        mask = cluster_labels == i
        cluster_vectors = diff_normalized[mask]
        n_cluster = int(np.sum(mask))

        if n_cluster >= 2:
            sim_matrix = cosine_similarity(cluster_vectors)
            upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            intra_sim = {"mean": float(np.mean(upper_tri)), "std": float(np.std(upper_tri)),
                         "min": float(np.min(upper_tri)), "max": float(np.max(upper_tri))}
        else:
            intra_sim = {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}

        cluster_silhouette = float(np.mean(silhouette_per_sample[mask])) if n_cluster > 0 else 0.0

        if n_cluster > 0:
            centroid = cluster_vectors.mean(axis=0, keepdims=True)
            dists = cosine_similarity(cluster_vectors, centroid).flatten()
            closest = np.argsort(dists)[-min(3, n_cluster):][::-1]
            representative_pairs = [int(np.where(mask)[0][j]) for j in closest]
        else:
            representative_pairs = []

        concepts.append({"id": i + 1, "n_pairs": n_cluster, "coherence": 0.0,
                         "silhouette": cluster_silhouette, "intra_similarity": intra_sim,
                         "representative_pairs": representative_pairs})

    concepts = name_concepts(concepts, pair_texts, cluster_labels, llm_model)

    pair_assignments = {}
    if pair_texts:
        for idx, pair_id in enumerate(sorted(pair_texts.keys())):
            if idx < len(cluster_labels):
                pair_assignments[pair_id] = int(cluster_labels[idx])

    centroids = [diff_normalized[cluster_labels == i].mean(axis=0) if np.sum(cluster_labels == i) > 0
                 else np.zeros(diff_normalized.shape[1]) for i in range(n_concepts)]
    inter_sim_matrix = cosine_similarity(np.array(centroids))
    off_diag = inter_sim_matrix[np.triu_indices_from(inter_sim_matrix, k=1)]

    result = {
        "n_concepts": n_concepts,
        "independence": 0.0, "mean_coherence": 0.0,
        "inter_concept_similarity": {"mean": float(np.mean(off_diag)) if len(off_diag) > 0 else 0.0,
                                     "max": float(np.max(off_diag)) if len(off_diag) > 0 else 0.0,
                                     "min": float(np.min(off_diag)) if len(off_diag) > 0 else 0.0,
                                     "matrix": inter_sim_matrix.tolist()},
        "concepts": concepts, "pair_assignments": pair_assignments,
        "cluster_labels": cluster_labels.tolist() if hasattr(cluster_labels, 'tolist') else list(cluster_labels),
        "pair_texts": pair_texts,
    }

    if generate_visualizations:
        from .concept_visualizations import create_all_concept_figures
        result["visualizations"] = create_all_concept_figures(
            pos_activations, neg_activations, cluster_labels, concepts,
            inter_concept_similarity=result["inter_concept_similarity"])

    return result


def decompose_and_name_concepts_with_labels(
    pos_activations,
    neg_activations,
    pair_texts: Optional[Dict[int, Dict[str, str]]] = None,
    cluster_labels: Optional[List[int]] = None,
    n_concepts: Optional[int] = None,
    generate_visualizations: bool = False,
    llm_model: str = "Qwen/Qwen3-8B",
) -> Dict[str, Any]:
    """
    Concept decomposition with pre-computed cluster labels (from multi-layer detection).

    Args:
        pos_activations: Positive activation tensors (for metrics, not clustering)
        neg_activations: Negative activation tensors (for metrics, not clustering)
        pair_texts: Optional dict mapping pair_id -> {prompt, positive, negative}
        cluster_labels: Pre-computed cluster labels from multi-layer detection
        n_concepts: Number of concepts (derived from cluster_labels if not provided)
        generate_visualizations: Whether to generate per-concept visualizations
        llm_model: HuggingFace model to use for naming

    Returns:
        Complete decomposition results with named concepts
    """
    cluster_labels = np.array(cluster_labels) if cluster_labels is not None else np.array([])

    if n_concepts is None:
        n_concepts = len(set(cluster_labels)) if len(cluster_labels) > 0 else 1

    n_pairs = min(len(pos_activations), len(neg_activations))
    if hasattr(pos_activations, 'numpy'):
        pos_np = pos_activations[:n_pairs].float().cpu().numpy()
        neg_np = neg_activations[:n_pairs].float().cpu().numpy()
    else:
        pos_np = pos_activations[:n_pairs]
        neg_np = neg_activations[:n_pairs]

    diff_vectors = pos_np - neg_np
    norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    diff_normalized = diff_vectors / norms

    return _build_result(
        diff_normalized, cluster_labels, n_concepts, pair_texts,
        pos_activations, neg_activations, generate_visualizations, llm_model
    )


def decompose_and_name_concepts(
    pos_activations,
    neg_activations,
    pair_texts: Optional[Dict[int, Dict[str, str]]] = None,
    generate_visualizations: bool = False,
    llm_model: str = "Qwen/Qwen3-8B",
) -> Dict[str, Any]:
    """Full concept decomposition with single-layer clustering and LLM naming."""
    from .concept_analysis import decompose_into_concepts
    from sklearn.cluster import SpectralClustering

    decomposition = decompose_into_concepts(pos_activations, neg_activations)
    n_concepts = decomposition["n_concepts"]

    n_pairs = min(len(pos_activations), len(neg_activations))
    if hasattr(pos_activations, 'numpy'):
        pos_np = pos_activations[:n_pairs].float().cpu().numpy()
        neg_np = neg_activations[:n_pairs].float().cpu().numpy()
    else:
        pos_np = pos_activations[:n_pairs]
        neg_np = neg_activations[:n_pairs]

    diff_vectors = pos_np - neg_np
    norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    diff_normalized = diff_vectors / norms

    spectral = SpectralClustering(
        n_clusters=n_concepts, random_state=42,
        affinity='nearest_neighbors', n_neighbors=min(10, n_pairs - 1)
    )
    cluster_labels = spectral.fit_predict(diff_normalized)

    # Delegate to shared builder
    return _build_result(
        diff_normalized, cluster_labels, n_concepts, pair_texts,
        pos_activations, neg_activations, generate_visualizations, llm_model
    )
