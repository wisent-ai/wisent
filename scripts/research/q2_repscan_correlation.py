#!/usr/bin/env python3
"""
Research Question 2: Is RepScan effective at predicting steering?

This module investigates whether geometry-based metrics can predict steering
effectiveness without actually training steering vectors.

Uses wisent.core.geometry.compute_geometry_metrics for comprehensive analysis.

Geometry Metrics Computed (from wisent.core.geometry):
- signal_strength, linear_probe_accuracy, mlp_probe_accuracy
- icd_* metrics (intrinsic concept dimensionality)
- direction_* metrics (stability, consistency)
- steer_* metrics (diff_mean_alignment, caa_probe_alignment, pct_positive_alignment,
  steering_vector_norm_ratio, cluster_direction_angle, per_cluster_alignment_k2,
  spherical_silhouette_k2, effective_steering_dims, steerability_score)
- concept_coherence, n_concepts
- recommended_method, recommendation_confidence

Methodology:
1. For each benchmark, compute geometry metrics using wisent's geometry module
2. Compute actual steering accuracy (ground truth)
3. Correlate geometry metrics with steering accuracy across benchmarks
4. Report Pearson and Spearman correlations
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .common import (
    ActivationData,
    load_activations_from_db,
    compute_geometry_metrics,
    compute_steering_accuracy,
)


def analyze_repscan_correlation(
    activations_by_benchmark: Dict[str, List[ActivationData]]
) -> Dict[str, Any]:
    """
    Compute geometry metrics and correlate with steering accuracy.

    For each benchmark:
    1. Extract positive and negative activation arrays
    2. Compute 5 geometry metrics
    3. Compute actual steering accuracy (CAA method)
    4. Collect all benchmark data points
    5. Compute correlations between each metric and steering accuracy

    Args:
        activations_by_benchmark: Dict mapping benchmark names to activation data

    Returns:
        Dict containing:
        - correlations: Pearson and Spearman correlations per metric
        - num_benchmarks: Number of benchmarks analyzed
        - benchmark_details: Per-benchmark geometry and accuracy data
    """
    geometry_data = []
    steering_data = []
    benchmark_names = []

    for benchmark, activations in activations_by_benchmark.items():
        # Use first available strategy with enough data
        by_strategy = defaultdict(list)
        for act in activations:
            by_strategy[act.strategy].append(act)

        for strategy, acts in by_strategy.items():
            if len(acts) < 20:
                continue

            pos = np.array([a.positive_activation for a in acts])
            neg = np.array([a.negative_activation for a in acts])

            # Compute geometry metrics
            geo_metrics = compute_geometry_metrics(pos, neg)

            # Compute steering accuracy (train/test)
            n = len(pos)
            train_idx = int(n * 0.8)
            train_pos, test_pos = pos[:train_idx], pos[train_idx:]
            train_neg, test_neg = neg[:train_idx], neg[train_idx:]

            if len(test_pos) < 2:
                continue

            steer_acc = compute_steering_accuracy(train_pos, train_neg, test_pos, test_neg, "caa")

            geometry_data.append(geo_metrics)
            steering_data.append(steer_acc)
            benchmark_names.append(f"{benchmark}:{strategy}")

            break  # Only use first strategy per benchmark for correlation

    if len(geometry_data) < 5:
        return {"error": "Not enough data for correlation analysis"}

    # Compute correlations between geometry metrics and steering accuracy
    correlations = {}
    for metric_name in geometry_data[0].keys():
        metric_values = [g[metric_name] for g in geometry_data]
        try:
            pearson_r, pearson_p = pearsonr(metric_values, steering_data)
            spearman_r, spearman_p = spearmanr(metric_values, steering_data)
            correlations[metric_name] = {
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "is_significant_pearson": pearson_p < 0.05,
                "is_significant_spearman": spearman_p < 0.05,
            }
        except Exception as e:
            correlations[metric_name] = {"error": str(e)}

    return {
        "correlations": correlations,
        "num_benchmarks": len(geometry_data),
        "benchmark_details": [
            {"name": name, "geometry": geo, "steering_accuracy": acc}
            for name, geo, acc in zip(benchmark_names, geometry_data, steering_data)
        ],
    }


def interpret_correlations(correlations: Dict[str, Dict]) -> Dict[str, str]:
    """
    Provide interpretations for correlation results.

    Returns:
        Dict of metric -> interpretation string
    """
    interpretations = {}

    for metric, corr in correlations.items():
        if "error" in corr:
            interpretations[metric] = f"Error: {corr['error']}"
            continue

        r = corr["pearson_r"]
        p = corr["pearson_p"]

        if p >= 0.05:
            strength = "not significant"
        elif abs(r) >= 0.7:
            strength = "strong"
        elif abs(r) >= 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        direction = "positive" if r > 0 else "negative"

        if p < 0.05:
            interpretations[metric] = f"{strength.capitalize()} {direction} correlation (r={r:.3f}, p={p:.3f})"
        else:
            interpretations[metric] = f"No significant correlation (r={r:.3f}, p={p:.3f})"

    return interpretations


def print_results(results: Dict[str, Any]):
    """Print formatted results to stdout."""
    print("\n" + "=" * 60)
    print("Q2: REPSCAN CORRELATION WITH STEERING")
    print("=" * 60)

    if "error" in results:
        print(f"\nError: {results['error']}")
        return

    print(f"\nAnalyzed {results['num_benchmarks']} benchmarks")

    print("\nCorrelations (geometry metric -> steering accuracy):")
    print("-" * 60)
    print(f"{'Metric':<25} {'Pearson r':<12} {'p-value':<12} {'Significant?'}")
    print("-" * 60)

    correlations = results["correlations"]
    interpretations = interpret_correlations(correlations)

    for metric in sorted(correlations.keys()):
        corr = correlations[metric]
        if "error" in corr:
            print(f"{metric:<25} ERROR: {corr['error']}")
        else:
            sig = "Yes" if corr["is_significant_pearson"] else "No"
            print(f"{metric:<25} {corr['pearson_r']:<12.3f} {corr['pearson_p']:<12.4f} {sig}")

    print("\nInterpretations:")
    print("-" * 60)
    for metric, interp in interpretations.items():
        print(f"  {metric}: {interp}")

    # Find best predictor
    best_metric = None
    best_r = 0
    for metric, corr in correlations.items():
        if "pearson_r" in corr and corr["is_significant_pearson"]:
            if abs(corr["pearson_r"]) > abs(best_r):
                best_r = corr["pearson_r"]
                best_metric = metric

    if best_metric:
        print(f"\nBest Predictor: {best_metric} (r={best_r:.3f})")
    else:
        print("\nNo significant predictors found")


def run_q2_analysis(model_name: str, layer: int = None, output_path: str = None) -> Dict[str, Any]:
    """
    Run complete Q2 analysis.

    Args:
        model_name: Model to analyze
        layer: Layer to use (default: middle)
        output_path: Optional path to save results JSON

    Returns:
        Complete results dict
    """
    print(f"Loading activations for {model_name}...")
    activations = load_activations_from_db(model_name, layer)
    print(f"Loaded {sum(len(v) for v in activations.values())} activations across {len(activations)} benchmarks")

    print("\nAnalyzing RepScan correlation...")
    results = analyze_repscan_correlation(activations)

    print_results(results)

    output = {
        "model": model_name,
        "layer": layer,
        "question": "Q2: Is RepScan effective at predicting steering?",
        "results": results,
        "interpretations": interpret_correlations(results.get("correlations", {})) if "correlations" in results else {},
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Q2: Is RepScan effective at predicting steering?"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--layer", type=int, default=None, help="Layer (default: middle)")
    parser.add_argument("--output", type=str, default="q2_results.json", help="Output file")
    args = parser.parse_args()

    run_q2_analysis(args.model, args.layer, args.output)


if __name__ == "__main__":
    main()
