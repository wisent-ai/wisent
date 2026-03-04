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

from wisent.core.utils.config_tools.constants import DEFAULT_RANDOM_SEED, JSON_INDENT, CHANCE_LEVEL_ACCURACY

# Import compute_geometry_metrics from metrics_core (single source of truth)
from wisent.core.reading.modules.utilities.metrics.core.metrics_core import compute_geometry_metrics
from wisent.core.reading.modules.modules.steering.analysis.steerability import compute_final_steering_prescription


def run_full_zwiad(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    layer: int,
    benchmark_name: str,
    min_clusters: int,
    output_dir: Optional[str] = None,
    *,
    spectral_n_neighbors: int,
    cv_folds: int,
    probe_min_per_class: int,
    probe_small_hidden: int,
    probe_mlp_hidden: int,
    probe_mlp_alpha: float,
    probe_validation_fraction: float,
    probe_knn_k: int,
    knn_min_class_offset: int,
    feature_dim_index: int,
    direction_n_bootstrap: int,
    direction_subset_fraction: float,
    direction_std_penalty: float,
    consistency_w_cosine: float,
    consistency_w_positive: float,
    consistency_w_high_sim: float,
    blend_default: float,
    default_score: float,
    detection_threshold: float,
    subsample_threshold: int,
    pca_dims_limit: int,
) -> Dict[str, Any]:
    """Run full representation scan for a single layer."""
    start_time = time.time()

    metrics = compute_geometry_metrics(
        pos_activations, neg_activations, min_clusters=min_clusters, n_folds=cv_folds,
        spectral_n_neighbors=spectral_n_neighbors,
        probe_min_per_class=probe_min_per_class, probe_small_hidden=probe_small_hidden,
        probe_mlp_hidden=probe_mlp_hidden, probe_mlp_alpha=probe_mlp_alpha,
        probe_validation_fraction=probe_validation_fraction, probe_knn_k=probe_knn_k,
        knn_min_class_offset=knn_min_class_offset, feature_dim_index=feature_dim_index,
        cv_folds=cv_folds, direction_n_bootstrap=direction_n_bootstrap,
        direction_subset_fraction=direction_subset_fraction, direction_std_penalty=direction_std_penalty,
        consistency_w_cosine=consistency_w_cosine, consistency_w_positive=consistency_w_positive,
        consistency_w_high_sim=consistency_w_high_sim,
        blend_default=blend_default, default_score=default_score,
        detection_threshold=detection_threshold,
        subsample_threshold=subsample_threshold, pca_dims_limit=pca_dims_limit,
    )

    result = {
        "benchmark": benchmark_name,
        "layer": layer,
        "n_pos": len(pos_activations),
        "n_neg": len(neg_activations),
        "metrics": metrics,
        "runtime_seconds": time.time() - start_time,
    }

    if output_dir:
        output_path = Path(output_dir) / f"{benchmark_name}_layer{layer}_zwiad.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=JSON_INDENT, default=str)

    return result


def run_full_zwiad_with_layer_search(
    activations_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    benchmark_name: str,
    min_clusters: int,
    output_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run zwiad across multiple layers."""
    results_by_layer = {}
    per_layer_metrics = {}

    for layer, (pos, neg) in activations_by_layer.items():
        result = run_full_zwiad(pos, neg, layer, benchmark_name, min_clusters=min_clusters, **kwargs)
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
            json.dump(summary, f, indent=JSON_INDENT, default=str)

    return summary


def run_full_zwiad_with_steering_eval(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    model,
    tokenizer,
    test_prompts: List[str],
    layer: int,
    benchmark_name: str,
    min_clusters: int,
    **kwargs,
) -> Dict[str, Any]:
    """Run zwiad and evaluate actual steering effectiveness."""
    cv_folds_val = kwargs["cv_folds"]
    metrics = compute_geometry_metrics(
        pos_activations, neg_activations, min_clusters=min_clusters,
        n_folds=cv_folds_val, **kwargs,
    )

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
    min_clusters: int,
    model=None,
    tokenizer=None,
    test_pairs: List[Tuple[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Return raw metrics relevant to steering effectiveness."""
    cv_folds_val = kwargs["cv_folds"]
    metrics = compute_geometry_metrics(
        pos_activations, neg_activations, min_clusters=min_clusters,
        n_folds=cv_folds_val, **kwargs,
    )

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
        gmm = GaussianMixture(n_components=2, random_state=DEFAULT_RANDOM_SEED)
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
            "gmm_separation": CHANCE_LEVEL_ACCURACY,
            "pos_cluster_purity": CHANCE_LEVEL_ACCURACY,
            "neg_cluster_purity": CHANCE_LEVEL_ACCURACY,
            "clusters_are_separated": False,
        }
