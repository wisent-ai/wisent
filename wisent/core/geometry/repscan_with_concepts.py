"""RepScan with automatic concept decomposition and naming."""
from typing import Dict, Any, Optional, List, Tuple
import torch

from .runner import compute_geometry_metrics
from .concept_naming import decompose_and_name_concepts, find_optimal_layer_per_concept
import numpy as np
from .database_loaders import load_activations_from_database, load_pair_texts_from_database

# Re-export database loaders for backwards compatibility
__all__ = [
    "run_repscan_with_concept_naming",
    "extract_pair_texts_from_enriched_pairs",
    "load_activations_from_database",
    "load_pair_texts_from_database",
    "find_optimal_layer_per_concept",
]


def run_repscan_with_concept_naming(
    activations_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    pair_texts: Optional[Dict[int, Dict[str, str]]] = None,
    include_expensive: bool = True,
    generate_visualizations: bool = False,
    llm_model: str = "meta-llama/Llama-3.2-1B-Instruct",
) -> Dict[str, Any]:
    """
    Run RepScan with automatic concept decomposition and LLM naming.

    Args:
        activations_by_layer: Dict mapping layer -> (pos_activations, neg_activations)
        pair_texts: Optional dict mapping pair_id -> {prompt, positive, negative}
        include_expensive: Whether to compute expensive metrics
        generate_visualizations: Whether to generate visualization figures
        llm_model: HuggingFace model for naming

    Returns:
        Dict with:
            - layers: Per-layer metrics
            - best_layer: Layer with highest ICD
            - concept_decomposition: Named concepts with pair assignments
    """
    results = {
        "layers": [],
        "concept_decomposition": None,
    }

    best_layer = None
    best_icd = -1
    best_activations = None

    # Run metrics for each layer
    for layer, (pos, neg) in sorted(activations_by_layer.items()):
        metrics = compute_geometry_metrics(
            pos_activations=pos,
            neg_activations=neg,
            include_expensive=include_expensive,
            generate_visualizations=generate_visualizations,
        )

        layer_result = {
            "layer": layer,
            "n_pos": len(pos),
            "n_neg": len(neg),
            "metrics": metrics,
        }
        results["layers"].append(layer_result)

        # Track best layer
        icd = metrics.get("icd_icd", 0)
        if icd > best_icd:
            best_icd = icd
            best_layer = layer
            best_activations = (pos, neg)

    results["best_layer"] = best_layer
    results["best_layer_icd"] = best_icd

    # Get recommendation from best layer
    for layer_result in results["layers"]:
        if layer_result["layer"] == best_layer:
            results["recommended_method"] = layer_result["metrics"].get("recommended_method")
            results["recommendation_confidence"] = layer_result["metrics"].get("recommendation_confidence")
            break

    # Run concept decomposition on best layer
    if best_activations is not None:
        pos, neg = best_activations
        decomposition = decompose_and_name_concepts(
            pos, neg, pair_texts,
            generate_visualizations=generate_visualizations,
            llm_model=llm_model,
        )

        # Find optimal layer per concept (different concepts may peak at different layers)
        cluster_labels = np.array(decomposition.get("cluster_labels", []))
        n_concepts = decomposition.get("n_concepts", 1)

        if len(cluster_labels) > 0 and n_concepts > 1:
            per_concept_layers = find_optimal_layer_per_concept(
                activations_by_layer, cluster_labels, n_concepts
            )

            # Add optimal layer info to each concept
            for concept in decomposition.get("concepts", []):
                concept_idx = concept["id"] - 1  # 0-indexed
                if concept_idx in per_concept_layers:
                    layer_info = per_concept_layers[concept_idx]
                    concept["optimal_layer"] = layer_info["best_layer"]
                    concept["optimal_layer_accuracy"] = layer_info["best_accuracy"]
                    concept["layer_accuracies"] = layer_info["layer_accuracies"]

            decomposition["per_concept_layers"] = per_concept_layers

        results["concept_decomposition"] = decomposition

    return results


def extract_pair_texts_from_enriched_pairs(enriched_pairs: List[Dict]) -> Dict[int, Dict[str, str]]:
    """
    Extract pair texts from enriched pairs format (wisent CLI output).

    Args:
        enriched_pairs: List of pairs in wisent CLI format

    Returns:
        Dict mapping pair index -> {prompt, positive, negative}
    """
    pair_texts = {}
    for i, pair in enumerate(enriched_pairs):
        pair_texts[i] = {
            "prompt": pair.get("prompt", ""),
            "positive": pair.get("positive_response", {}).get("model_response", ""),
            "negative": pair.get("negative_response", {}).get("model_response", ""),
        }
    return pair_texts
