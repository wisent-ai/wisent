#!/usr/bin/env python3
"""
Research Question 4: Is there a unified direction that improves performance over all benchmarks?

This module investigates whether a single steering vector trained on all benchmarks
combined can achieve competitive performance across individual benchmarks.

Key Question: Can we find a "universal goodness" direction that generalizes?

Methodology:
1. Collect activations from all benchmarks into a single dataset
2. Train a unified steering vector on the combined data
3. Evaluate the unified vector on each individual benchmark
4. Compare unified performance to per-benchmark optimal vectors
5. Analyze when unified works vs when it doesn't
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np

from .common import ActivationData, load_activations_from_db


def analyze_unified_direction(
    activations_by_benchmark: Dict[str, List[ActivationData]]
) -> Dict[str, Any]:
    """
    Investigate unified steering direction across all benchmarks.

    Steps:
    1. Collect all activations (using strategy with most data per benchmark)
    2. Train unified steering vector (CAA: mean positive - mean negative)
    3. Evaluate unified vector on each benchmark (pairwise accuracy)
    4. Compare to per-benchmark vectors
    5. Analyze correlation between benchmarks

    Args:
        activations_by_benchmark: Dict mapping benchmark names to activation data

    Returns:
        Dict containing unified vs per-benchmark comparison and analysis
    """
    # Collect all activations (using first strategy per benchmark)
    all_pos = []
    all_neg = []
    benchmark_slices = {}  # benchmark -> (start_idx, end_idx)

    for benchmark, activations in activations_by_benchmark.items():
        by_strategy = defaultdict(list)
        for act in activations:
            by_strategy[act.strategy].append(act)

        # Use strategy with most data
        best_strategy = max(by_strategy.keys(), key=lambda s: len(by_strategy[s]), default=None)
        if best_strategy is None or len(by_strategy[best_strategy]) < 10:
            continue

        acts = by_strategy[best_strategy]
        start_idx = len(all_pos)

        for a in acts:
            all_pos.append(a.positive_activation)
            all_neg.append(a.negative_activation)

        benchmark_slices[benchmark] = (start_idx, len(all_pos))

    if len(all_pos) < 50:
        return {"error": "Not enough data for unified direction analysis"}

    all_pos = np.array(all_pos)
    all_neg = np.array(all_neg)

    # Train unified steering vector
    unified_vector = all_pos.mean(axis=0) - all_neg.mean(axis=0)
    unified_norm = np.linalg.norm(unified_vector)
    if unified_norm > 1e-10:
        unified_vector = unified_vector / unified_norm

    # Evaluate on each benchmark
    unified_results = {}
    per_benchmark_results = {}
    per_benchmark_vectors = {}

    for benchmark, (start, end) in benchmark_slices.items():
        pos = all_pos[start:end]
        neg = all_neg[start:end]

        if len(pos) < 5:
            continue

        # Unified direction accuracy
        pos_proj = pos @ unified_vector
        neg_proj = neg @ unified_vector

        unified_correct = 0
        unified_total = 0
        for p in pos_proj:
            for n in neg_proj:
                if p > n:
                    unified_correct += 1
                unified_total += 1
        unified_acc = unified_correct / unified_total if unified_total > 0 else 0.5

        # Per-benchmark direction accuracy
        per_bench_vector = pos.mean(axis=0) - neg.mean(axis=0)
        per_bench_norm = np.linalg.norm(per_bench_vector)
        if per_bench_norm > 1e-10:
            per_bench_vector = per_bench_vector / per_bench_norm
            per_benchmark_vectors[benchmark] = per_bench_vector

        pos_proj_pb = pos @ per_bench_vector
        neg_proj_pb = neg @ per_bench_vector

        pb_correct = 0
        pb_total = 0
        for p in pos_proj_pb:
            for n in neg_proj_pb:
                if p > n:
                    pb_correct += 1
                pb_total += 1
        per_bench_acc = pb_correct / pb_total if pb_total > 0 else 0.5

        unified_results[benchmark] = unified_acc
        per_benchmark_results[benchmark] = per_bench_acc

    # Analyze results
    unified_accs = list(unified_results.values())
    per_bench_accs = list(per_benchmark_results.values())

    # How many benchmarks does unified beat per-benchmark?
    unified_wins = sum(1 for b in unified_results if unified_results[b] >= per_benchmark_results.get(b, 0))
    unified_competitive = sum(1 for b in unified_results
                              if abs(unified_results[b] - per_benchmark_results.get(b, 0)) < 0.05)

    # Analyze benchmark alignment with unified direction
    benchmark_alignments = {}
    for benchmark, vec in per_benchmark_vectors.items():
        alignment = np.dot(vec, unified_vector)
        benchmark_alignments[benchmark] = float(alignment)

    # Categorize benchmarks
    aligned_benchmarks = [b for b, a in benchmark_alignments.items() if a >= 0.5]
    orthogonal_benchmarks = [b for b, a in benchmark_alignments.items() if -0.5 < a < 0.5]
    anti_aligned_benchmarks = [b for b, a in benchmark_alignments.items() if a <= -0.5]

    return {
        "unified_vs_per_benchmark": {
            benchmark: {
                "unified_accuracy": unified_results.get(benchmark, 0),
                "per_benchmark_accuracy": per_benchmark_results.get(benchmark, 0),
                "unified_better": unified_results.get(benchmark, 0) >= per_benchmark_results.get(benchmark, 0),
                "difference": unified_results.get(benchmark, 0) - per_benchmark_results.get(benchmark, 0),
                "alignment_with_unified": benchmark_alignments.get(benchmark, 0),
            }
            for benchmark in unified_results
        },
        "summary": {
            "unified_mean_accuracy": float(np.mean(unified_accs)) if unified_accs else 0.0,
            "per_benchmark_mean_accuracy": float(np.mean(per_bench_accs)) if per_bench_accs else 0.0,
            "unified_std_accuracy": float(np.std(unified_accs)) if unified_accs else 0.0,
            "per_benchmark_std_accuracy": float(np.std(per_bench_accs)) if per_bench_accs else 0.0,
            "unified_wins_count": unified_wins,
            "unified_competitive_count": unified_competitive,
            "total_benchmarks": len(unified_results),
            "unified_win_rate": unified_wins / len(unified_results) if unified_results else 0.0,
        },
        "alignment_analysis": {
            "aligned_benchmarks": aligned_benchmarks,
            "aligned_count": len(aligned_benchmarks),
            "orthogonal_benchmarks": orthogonal_benchmarks,
            "orthogonal_count": len(orthogonal_benchmarks),
            "anti_aligned_benchmarks": anti_aligned_benchmarks,
            "anti_aligned_count": len(anti_aligned_benchmarks),
            "mean_alignment": float(np.mean(list(benchmark_alignments.values()))) if benchmark_alignments else 0.0,
        },
        "total_pairs_used": len(all_pos),
    }


def print_results(results: Dict[str, Any]):
    """Print formatted results to stdout."""
    print("\n" + "=" * 60)
    print("Q4: UNIFIED DIRECTION ANALYSIS")
    print("=" * 60)

    if "error" in results:
        print(f"\nError: {results['error']}")
        return

    summary = results["summary"]
    print(f"\nAnalyzed {summary['total_benchmarks']} benchmarks ({results['total_pairs_used']} total pairs)")

    print("\nAccuracy Comparison:")
    print("-" * 40)
    print(f"  Unified mean accuracy:       {summary['unified_mean_accuracy']:.3f} (std: {summary['unified_std_accuracy']:.3f})")
    print(f"  Per-benchmark mean accuracy: {summary['per_benchmark_mean_accuracy']:.3f} (std: {summary['per_benchmark_std_accuracy']:.3f})")

    print("\nUnified Direction Wins:")
    print("-" * 40)
    print(f"  Unified beats per-benchmark: {summary['unified_wins_count']}/{summary['total_benchmarks']} ({summary['unified_win_rate']:.1%})")
    print(f"  Unified competitive (within 5%): {summary['unified_competitive_count']}/{summary['total_benchmarks']}")

    alignment = results["alignment_analysis"]
    print("\nBenchmark Alignment with Unified Direction:")
    print("-" * 40)
    print(f"  Aligned (>0.5):      {alignment['aligned_count']} benchmarks")
    print(f"  Orthogonal (-0.5 to 0.5): {alignment['orthogonal_count']} benchmarks")
    print(f"  Anti-aligned (<-0.5): {alignment['anti_aligned_count']} benchmarks")
    print(f"  Mean alignment:       {alignment['mean_alignment']:.3f}")

    # Top benchmarks where unified works well
    print("\nBenchmarks where Unified Direction Works Best:")
    print("-" * 60)
    sorted_by_unified = sorted(
        results["unified_vs_per_benchmark"].items(),
        key=lambda x: x[1]["unified_accuracy"],
        reverse=True
    )[:10]

    print(f"{'Benchmark':<35} {'Unified':<10} {'Per-bench':<10} {'Diff'}")
    for bench, data in sorted_by_unified:
        print(f"{bench[:35]:<35} {data['unified_accuracy']:.3f}      {data['per_benchmark_accuracy']:.3f}      {data['difference']:+.3f}")

    # Benchmarks where unified fails
    print("\nBenchmarks where Unified Direction Fails:")
    print("-" * 60)
    sorted_by_diff = sorted(
        results["unified_vs_per_benchmark"].items(),
        key=lambda x: x[1]["difference"]
    )[:10]

    for bench, data in sorted_by_diff:
        print(f"{bench[:35]:<35} {data['unified_accuracy']:.3f}      {data['per_benchmark_accuracy']:.3f}      {data['difference']:+.3f}")


def run_q4_analysis(model_name: str, layer: int = None, output_path: str = None) -> Dict[str, Any]:
    """
    Run complete Q4 analysis.

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

    print("\nAnalyzing unified direction...")
    results = analyze_unified_direction(activations)

    print_results(results)

    output = {
        "model": model_name,
        "layer": layer,
        "question": "Q4: Is there a unified direction that improves performance over all benchmarks?",
        "results": results,
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Q4: Is there a unified direction that improves performance over all benchmarks?"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--layer", type=int, default=None, help="Layer (default: middle)")
    parser.add_argument("--output", type=str, default="q4_results.json", help="Output file")
    args = parser.parse_args()

    run_q4_analysis(args.model, args.layer, args.output)


if __name__ == "__main__":
    main()
