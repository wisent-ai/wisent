"""RepScan with automatic concept decomposition and naming."""
from typing import Dict, Any, Optional, List, Tuple, Set
import torch
import numpy as np

from ..metrics.core.metrics_core import compute_geometry_metrics
from ..concepts import decompose_and_name_concepts_with_labels, find_optimal_layer_per_concept
from ..data.database_loaders import load_activations_from_database, load_pair_texts_from_database, load_available_layers_from_database

__all__ = [
    "run_repscan_with_concept_naming",
    "extract_pair_texts_from_enriched_pairs",
    "load_activations_from_database",
    "load_pair_texts_from_database",
    "load_available_layers_from_database",
    "find_optimal_layer_per_concept",
]


def run_repscan_with_concept_naming(
    activations_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    pair_texts: Optional[Dict[int, Dict[str, str]]] = None,
    generate_visualizations: bool = False,
    llm_model: str = "meta-llama/Llama-3.2-1B-Instruct",
    steps: str = "all",
) -> Dict[str, Any]:
    """
    Run RepScan with geometry metrics, 5-step protocol, and concept naming.

    Combines:
    - Full geometry metrics (signal_strength, linear_probe, ICD, etc.)
    - 5-step protocol (signal, geometry, decomposition, intervention, editability)
    - Concept decomposition and LLM naming

    Args:
        steps: 'all' runs everything, or comma-separated subset (e.g., 'signal,editability')
    """
    from .repscan_protocol import test_signal, test_geometry, test_decomposition, select_intervention

    # Parse steps
    if steps == "all":
        steps_to_run = {"signal", "geometry", "decomposition", "intervention", "editability"}
    else:
        steps_to_run = {s.strip().lower() for s in steps.split(",")}

    # Concatenate all layers
    sorted_layers = sorted(activations_by_layer.keys())
    n_layers = len(sorted_layers)
    pos_list = [activations_by_layer[l][0] for l in sorted_layers]
    neg_list = [activations_by_layer[l][1] for l in sorted_layers]
    pos_concat = torch.cat(pos_list, dim=1)
    neg_concat = torch.cat(neg_list, dim=1)
    n_pairs = len(pos_concat)

    # Always compute full geometry metrics
    metrics = compute_geometry_metrics(pos_concat, neg_concat, generate_visualizations=generate_visualizations)

    results = {
        "n_pairs": n_pairs, "n_layers": n_layers, "layers_used": sorted_layers,
        "total_dims": pos_concat.shape[1], "steps_run": list(steps_to_run),
        "metrics": metrics,
        "recommended_method": metrics.get("recommended_method"),
        "recommendation_confidence": metrics.get("recommendation_confidence"),
    }

    signal_result, geometry_result, decomposition_result = None, None, None

    # Step 1: Signal Test (with null distribution testing)
    if "signal" in steps_to_run:
        metric_keys = ["knn_accuracy", "knn_pca_accuracy", "mlp_probe_accuracy"]
        signal_result = test_signal(pos_concat, neg_concat, metric_keys)
        results["signal_test"] = {
            "max_z_score": signal_result.max_z_score,
            "min_p_value": signal_result.min_p_value,
            "passed": signal_result.passed,
            "permutation_metrics": signal_result.permutation_metrics,
            "nonsense_metrics": signal_result.nonsense_metrics,
        }

    # Step 2: Geometry Test (linear vs nonlinear)
    if "geometry" in steps_to_run:
        geometry_result = test_geometry(pos_concat, neg_concat)
        results["geometry_test"] = {
            "linear_accuracy": geometry_result.linear_accuracy,
            "nonlinear_accuracy": geometry_result.nonlinear_accuracy,
            "gap": geometry_result.gap,
            "diagnosis": geometry_result.diagnosis,
        }

    # Step 3: Decomposition Test + Concept Naming
    if "decomposition" in steps_to_run:
        decomposition_result = test_decomposition(pos_concat, neg_concat)
        results["decomposition_test"] = {
            "n_concepts": decomposition_result.n_concepts,
            "silhouette_score": decomposition_result.silhouette_score,
            "is_fragmented": decomposition_result.is_fragmented,
            "per_concept_sizes": decomposition_result.per_concept_sizes,
        }
        # Concept naming with LLM
        decomposition = decompose_and_name_concepts_with_labels(
            pos_concat, neg_concat, pair_texts,
            cluster_labels=decomposition_result.cluster_labels,
            n_concepts=decomposition_result.n_concepts,
            generate_visualizations=generate_visualizations, llm_model=llm_model,
        )
        decomposition["n_layers_used"] = n_layers
        # Optimal layer per concept
        if decomposition_result.n_concepts > 1:
            labels_arr = np.array(decomposition_result.cluster_labels)
            per_concept_layers = find_optimal_layer_per_concept(activations_by_layer, labels_arr, decomposition_result.n_concepts)
            for concept in decomposition.get("concepts", []):
                idx = concept["id"] - 1
                if idx in per_concept_layers:
                    concept["optimal_layer"] = per_concept_layers[idx]["best_layer"]
                    concept["optimal_layer_accuracy"] = per_concept_layers[idx]["best_accuracy"]
            decomposition["per_concept_layers"] = per_concept_layers
        results["concept_decomposition"] = decomposition

    # Step 4: Intervention Selection
    if "intervention" in steps_to_run and signal_result and geometry_result and decomposition_result:
        intervention = select_intervention(signal_result, geometry_result, decomposition_result)
        results["intervention"] = {
            "recommended_method": intervention.recommended_method,
            "confidence": intervention.confidence,
            "reasoning": intervention.reasoning,
            "method_scores": intervention.method_scores,
        }
        results["recommended_method"] = intervention.recommended_method
        results["recommendation_confidence"] = intervention.confidence

    # Step 5: Editability Analysis
    if "editability" in steps_to_run:
        from .repscan_editability import test_editability
        cluster_labels = decomposition_result.cluster_labels if decomposition_result else None
        n_concepts = decomposition_result.n_concepts if decomposition_result else 1
        editability_result = test_editability(
            pos_concat, neg_concat, cluster_labels=cluster_labels, n_concepts=n_concepts,
        )
        results["editability_analysis"] = {
            "editing_capacity": editability_result.editing_capacity,
            "effective_preserved_rank": editability_result.effective_preserved_rank,
            "singular_values": editability_result.singular_values,
            "spectral_decay_rate": editability_result.spectral_decay_rate,
            "steering_survival_ratio": editability_result.steering_survival_ratio,
            "verdict": editability_result.verdict,
            "concept_interference": editability_result.concept_interference,
            "editability_score": editability_result.editability_score,
            "participation_ratio": editability_result.participation_ratio,
            "warnings": editability_result.warnings,
        }

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
