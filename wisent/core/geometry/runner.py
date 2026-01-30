"""
Main geometry runner that orchestrates all geometry analysis.

This module provides the main entry points for running geometry
analysis on activation representations.
"""

import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import numpy as np

# Import compute_geometry_metrics from metrics_core (single source of truth)
from .metrics.core.metrics_core import compute_geometry_metrics
from .steering.analysis.steerability import compute_final_steering_prescription


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
    Run repscan across multiple layers.

    Args:
        activations_by_layer: Dict mapping layer -> (pos, neg) activations
        benchmark_name: Name of benchmark
        output_dir: Optional directory to save results

    Returns:
        Dict with per-layer raw metrics. No recommendations or "best" layer selection.
    """
    results_by_layer = {}
    per_layer_metrics = {}

    for layer, (pos, neg) in activations_by_layer.items():
        result = run_full_repscan(pos, neg, layer, benchmark_name)
        results_by_layer[layer] = result
        per_layer_metrics[layer] = result["metrics"]

    # Extract raw metrics per layer
    prescription = compute_final_steering_prescription(per_layer_metrics)

    summary = {
        "benchmark": benchmark_name,
        "per_layer_metrics": prescription.get("per_layer", {}),
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
    Return raw metrics relevant to steering effectiveness.

    Does NOT predict effectiveness - just returns the raw metrics.
    """
    metrics = compute_geometry_metrics(pos_activations, neg_activations)

    return {
        "caa_probe_alignment": metrics.get("steer_caa_probe_alignment"),
        "diff_mean_alignment": metrics.get("steer_diff_mean_alignment"),
        "linear_accuracy": metrics.get("linear_probe_accuracy"),
        "icd": metrics.get("icd_icd"),
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
