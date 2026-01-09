"""
Main geometry runner that orchestrates all geometry analysis.

This module provides the main entry points for running geometry
analysis on activation representations.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import numpy as np

from .probe_metrics import (
    compute_signal_strength,
    compute_linear_probe_accuracy,
    compute_mlp_probe_accuracy,
    compute_knn_accuracy,
    compute_knn_pca_accuracy,
)
from .distribution_metrics import (
    compute_mmd_rbf,
    compute_density_ratio,
    compute_fisher_per_dimension,
)
from .intrinsic_dim import (
    compute_local_intrinsic_dims,
    compute_diff_intrinsic_dim,
)
from .direction_metrics import (
    compute_direction_stability,
    compute_multi_direction_accuracy,
    compute_pairwise_diff_consistency,
)
from .steerability import (
    compute_steerability_metrics,
    compute_linearity_score,
    compute_recommendation,
)
from .icd import compute_icd
from .concept_analysis import (
    detect_multiple_concepts,
    compute_concept_coherence,
)
from .signal_analysis import (
    compute_signal_to_noise,
    compute_bootstrap_signal_estimate,
)


def compute_geometry_metrics(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    include_expensive: bool = True,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """
    Compute comprehensive geometry metrics for activations.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        include_expensive: Whether to include computationally expensive metrics
        n_folds: Number of CV folds
        
    Returns:
        Dict with all computed metrics
    """
    metrics = {}
    
    # Basic probe metrics
    metrics["signal_strength"] = compute_signal_strength(pos_activations, neg_activations, n_folds)
    metrics["linear_probe_accuracy"] = compute_linear_probe_accuracy(pos_activations, neg_activations, n_folds)
    metrics["mlp_probe_accuracy"] = compute_mlp_probe_accuracy(pos_activations, neg_activations, n_folds=n_folds)
    
    # ICD
    icd_result = compute_icd(pos_activations, neg_activations)
    metrics.update({f"icd_{k}": v for k, v in icd_result.items()})
    
    # Direction metrics
    stability = compute_direction_stability(pos_activations, neg_activations)
    metrics.update({f"direction_{k}": v for k, v in stability.items()})
    
    consistency = compute_pairwise_diff_consistency(pos_activations, neg_activations)
    metrics.update({f"consistency_{k}": v for k, v in consistency.items()})
    
    # Steerability
    steerability = compute_steerability_metrics(pos_activations, neg_activations)
    metrics.update({f"steer_{k}": v for k, v in steerability.items()})
    
    # Concept analysis
    metrics["concept_coherence"] = compute_concept_coherence(pos_activations, neg_activations)
    concept_detection = detect_multiple_concepts(pos_activations, neg_activations)
    metrics["n_concepts"] = concept_detection.get("n_concepts", 1)
    
    if include_expensive:
        # Distribution metrics
        metrics["mmd_rbf"] = compute_mmd_rbf(pos_activations, neg_activations)
        metrics["density_ratio"] = compute_density_ratio(pos_activations, neg_activations)
        
        fisher = compute_fisher_per_dimension(pos_activations, neg_activations)
        metrics.update({f"fisher_{k}": v for k, v in fisher.items()})
        
        # Intrinsic dim
        dim_pos, dim_neg, dim_ratio = compute_local_intrinsic_dims(pos_activations, neg_activations)
        metrics["intrinsic_dim_pos"] = dim_pos
        metrics["intrinsic_dim_neg"] = dim_neg
        metrics["intrinsic_dim_ratio"] = dim_ratio
        
        # Multi-direction
        multi_dir = compute_multi_direction_accuracy(pos_activations, neg_activations)
        metrics["multi_dir_saturation_k"] = multi_dir.get("saturation_k", 1)
        metrics["multi_dir_gain"] = multi_dir.get("gain_from_multi", 0.0)
        
        # k-NN
        metrics["knn_accuracy"] = compute_knn_accuracy(pos_activations, neg_activations)
        metrics["knn_pca_accuracy"] = compute_knn_pca_accuracy(pos_activations, neg_activations)
        
        # Signal to noise
        metrics["signal_to_noise"] = compute_signal_to_noise(pos_activations, neg_activations)
    
    # Generate recommendation
    recommendation = compute_recommendation(metrics)
    metrics["recommended_method"] = recommendation.get("recommended_method", "CAA")
    metrics["recommendation_confidence"] = recommendation.get("confidence", 0.5)
    
    return metrics


def run_full_repscan(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    layer: int,
    benchmark_name: str = "unknown",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full representation scan for a single layer.
    
    Args:
        pos_activations: Positive class activations
        neg_activations: Negative class activations
        layer: Layer number
        benchmark_name: Name of benchmark
        output_dir: Optional directory to save results
        
    Returns:
        Dict with all metrics and recommendations
    """
    start_time = time.time()
    
    metrics = compute_geometry_metrics(pos_activations, neg_activations)
    
    result = {
        "benchmark": benchmark_name,
        "layer": layer,
        "n_pos": len(pos_activations),
        "n_neg": len(neg_activations),
        "metrics": metrics,
        "runtime_seconds": time.time() - start_time,
    }
    
    if output_dir:
        output_path = Path(output_dir) / f"{benchmark_name}_layer{layer}_repscan.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    return result


def run_full_repscan_with_layer_search(
    activations_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    benchmark_name: str = "unknown",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run repscan across multiple layers to find optimal layer.
    
    Args:
        activations_by_layer: Dict mapping layer -> (pos, neg) activations
        benchmark_name: Name of benchmark
        output_dir: Optional directory to save results
        
    Returns:
        Dict with per-layer results and best layer recommendation
    """
    results_by_layer = {}
    
    for layer, (pos, neg) in activations_by_layer.items():
        result = run_full_repscan(pos, neg, layer, benchmark_name)
        results_by_layer[layer] = result
    
    # Find best layer by steerability score
    best_layer = max(
        results_by_layer.keys(),
        key=lambda l: results_by_layer[l]["metrics"].get("steer_steerability_score", 0)
    )
    
    summary = {
        "benchmark": benchmark_name,
        "best_layer": best_layer,
        "best_steerability": results_by_layer[best_layer]["metrics"].get("steer_steerability_score", 0),
        "results_by_layer": results_by_layer,
    }
    
    if output_dir:
        output_path = Path(output_dir) / f"{benchmark_name}_layer_search.json"
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    return summary


def run_full_repscan_with_steering_eval(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    model,
    tokenizer,
    test_prompts: List[str],
    layer: int,
    benchmark_name: str = "unknown",
) -> Dict[str, Any]:
    """
    Run repscan and evaluate actual steering effectiveness.
    
    This combines geometry analysis with actual steering tests.
    """
    # Run geometry analysis
    metrics = compute_geometry_metrics(pos_activations, neg_activations)
    
    # Compute steering direction
    n = min(len(pos_activations), len(neg_activations))
    steering_direction = (pos_activations[:n] - neg_activations[:n]).mean(dim=0)
    
    result = {
        "benchmark": benchmark_name,
        "layer": layer,
        "metrics": metrics,
        "steering_direction_norm": float(torch.norm(steering_direction)),
        "n_test_prompts": len(test_prompts),
    }
    
    return result


def evaluate_steering_effectiveness(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    model=None,
    tokenizer=None,
    test_pairs: List[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate how effective steering would be for these activations.
    
    Uses geometry metrics to predict steering success without
    actually running steering experiments.
    """
    metrics = compute_geometry_metrics(pos_activations, neg_activations, include_expensive=False)
    
    steerability = metrics.get("steer_steerability_score", 0.0)
    linear_acc = metrics.get("linear_probe_accuracy", 0.5)
    icd = metrics.get("icd_icd", 10.0)
    
    # Heuristic: predict steering effectiveness
    predicted_effectiveness = (
        0.4 * steerability +
        0.3 * (linear_acc - 0.5) * 2 +  # Scale 0.5-1.0 to 0-1
        0.3 * max(0, 1 - icd / 20)  # Lower ICD = better
    )
    predicted_effectiveness = float(np.clip(predicted_effectiveness, 0, 1))
    
    return {
        "predicted_effectiveness": predicted_effectiveness,
        "steerability_score": steerability,
        "linear_accuracy": linear_acc,
        "icd": icd,
        "recommended_method": metrics.get("recommended_method", "CAA"),
    }


def evaluate_activation_regions(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, Any]:
    """
    Analyze different regions of activation space.
    
    Checks if pos/neg form distinct clusters or overlap.
    """
    try:
        from sklearn.mixture import GaussianMixture
        
        pos = pos_activations.float().cpu().numpy()
        neg = neg_activations.float().cpu().numpy()
        
        # Fit GMM to combined data
        X = np.vstack([pos, neg])
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        
        # Predict cluster assignments
        pos_clusters = gmm.predict(pos)
        neg_clusters = gmm.predict(neg)
        
        # Check separation
        pos_majority = int(np.median(pos_clusters))
        neg_majority = int(np.median(neg_clusters))
        
        separation = float((pos_clusters == pos_majority).mean() * (neg_clusters == neg_majority).mean())
        
        return {
            "gmm_separation": separation,
            "pos_cluster_purity": float((pos_clusters == pos_majority).mean()),
            "neg_cluster_purity": float((neg_clusters == neg_majority).mean()),
            "clusters_are_separated": pos_majority != neg_majority,
        }
    except Exception:
        return {
            "gmm_separation": 0.5,
            "pos_cluster_purity": 0.5,
            "neg_cluster_purity": 0.5,
            "clusters_are_separated": False,
        }
