#!/usr/bin/env python3
"""
Research Question 3: What is the maximum we can achieve per benchmark?

This module finds the optimal combination of extraction strategy and steering
method (CAA vs Hyperplane) for each benchmark, establishing upper bounds on
per-benchmark steering performance.

Methodology:
1. For each benchmark, evaluate all available extraction strategies
2. For each strategy, try both CAA and Hyperplane steering methods
3. Select the best (strategy, method) combination
4. Report per-benchmark maximums and aggregate statistics
"""

import argparse
import json
from typing import Dict, Any

import numpy as np

from .common import BenchmarkResults
from .q1_strategy_comparison import analyze_strategy_performance, load_activations_from_db


def analyze_per_benchmark_maximum(
    strategy_results: Dict[str, BenchmarkResults]
) -> Dict[str, Any]:
    """
    Find the best strategy and method combination for each benchmark.

    Args:
        strategy_results: Results from analyze_strategy_performance()

    Returns:
        Dict containing:
        - per_benchmark: Best configuration per benchmark
        - summary: Aggregate statistics
    """
    maximums = {}

    for benchmark, result in strategy_results.items():
        if not result.strategies:
            continue

        best_strategy = None
        best_method = None
        best_acc = 0.0

        for strategy, metrics in result.strategies.items():
            if metrics["caa_accuracy"] > best_acc:
                best_acc = metrics["caa_accuracy"]
                best_strategy = strategy
                best_method = "caa"
            if metrics["hyperplane_accuracy"] > best_acc:
                best_acc = metrics["hyperplane_accuracy"]
                best_strategy = strategy
                best_method = "hyperplane"

        maximums[benchmark] = {
            "best_accuracy": best_acc,
            "best_strategy": best_strategy,
            "best_method": best_method,
            "num_pairs": result.num_pairs,
        }

    # Summary statistics
    accuracies = [m["best_accuracy"] for m in maximums.values()]

    # Categorize by performance tier
    excellent = [b for b, m in maximums.items() if m["best_accuracy"] >= 0.9]
    good = [b for b, m in maximums.items() if 0.7 <= m["best_accuracy"] < 0.9]
    moderate = [b for b, m in maximums.items() if 0.5 < m["best_accuracy"] < 0.7]
    poor = [b for b, m in maximums.items() if m["best_accuracy"] <= 0.5]

    return {
        "per_benchmark": maximums,
        "summary": {
            "mean_best_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "median_best_accuracy": float(np.median(accuracies)) if accuracies else 0.0,
            "std_best_accuracy": float(np.std(accuracies)) if accuracies else 0.0,
            "min_best_accuracy": float(np.min(accuracies)) if accuracies else 0.0,
            "max_best_accuracy": float(np.max(accuracies)) if accuracies else 0.0,
            "num_benchmarks": len(maximums),
        },
        "performance_tiers": {
            "excellent_90_plus": {
                "count": len(excellent),
                "benchmarks": excellent,
            },
            "good_70_90": {
                "count": len(good),
                "benchmarks": good,
            },
            "moderate_50_70": {
                "count": len(moderate),
                "benchmarks": moderate,
            },
            "poor_50_below": {
                "count": len(poor),
                "benchmarks": poor,
            },
        },
    }


def print_results(results: Dict[str, Any]):
    """Print formatted results to stdout."""
    print("\n" + "=" * 60)
    print("Q3: PER-BENCHMARK MAXIMUM ACCURACY")
    print("=" * 60)

    summary = results["summary"]
    print(f"\nAnalyzed {summary['num_benchmarks']} benchmarks")

    print("\nOverall Statistics:")
    print("-" * 40)
    print(f"  Mean best accuracy:   {summary['mean_best_accuracy']:.3f}")
    print(f"  Median best accuracy: {summary['median_best_accuracy']:.3f}")
    print(f"  Std deviation:        {summary['std_best_accuracy']:.3f}")
    print(f"  Range: [{summary['min_best_accuracy']:.3f}, {summary['max_best_accuracy']:.3f}]")

    print("\nPerformance Tiers:")
    print("-" * 40)
    tiers = results["performance_tiers"]
    print(f"  Excellent (>=90%): {tiers['excellent_90_plus']['count']} benchmarks")
    print(f"  Good (70-90%):     {tiers['good_70_90']['count']} benchmarks")
    print(f"  Moderate (50-70%): {tiers['moderate_50_70']['count']} benchmarks")
    print(f"  Poor (<=50%):      {tiers['poor_50_below']['count']} benchmarks")

    # Top 10 benchmarks
    print("\nTop 10 Benchmarks by Steering Accuracy:")
    print("-" * 60)
    sorted_benchmarks = sorted(
        results["per_benchmark"].items(),
        key=lambda x: x[1]["best_accuracy"],
        reverse=True
    )[:10]

    print(f"{'Benchmark':<35} {'Accuracy':<10} {'Strategy':<15} {'Method'}")
    print("-" * 60)
    for bench, data in sorted_benchmarks:
        print(f"{bench[:35]:<35} {data['best_accuracy']:.3f}      {data['best_strategy']:<15} {data['best_method']}")

    # Bottom 10 benchmarks
    if len(results["per_benchmark"]) > 10:
        print("\nBottom 10 Benchmarks by Steering Accuracy:")
        print("-" * 60)
        sorted_benchmarks = sorted(
            results["per_benchmark"].items(),
            key=lambda x: x[1]["best_accuracy"]
        )[:10]

        print(f"{'Benchmark':<35} {'Accuracy':<10} {'Strategy':<15} {'Method'}")
        print("-" * 60)
        for bench, data in sorted_benchmarks:
            print(f"{bench[:35]:<35} {data['best_accuracy']:.3f}      {data['best_strategy']:<15} {data['best_method']}")


def run_q3_analysis(model_name: str, layer: int = None, output_path: str = None) -> Dict[str, Any]:
    """
    Run complete Q3 analysis.

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

    print("\nAnalyzing strategy performance...")
    strategy_results = analyze_strategy_performance(activations)

    print("\nFinding per-benchmark maximums...")
    results = analyze_per_benchmark_maximum(strategy_results)

    print_results(results)

    output = {
        "model": model_name,
        "layer": layer,
        "question": "Q3: What is the maximum we can achieve per benchmark?",
        "results": results,
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Q3: What is the maximum we can achieve per benchmark?"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--layer", type=int, default=None, help="Layer (default: middle)")
    parser.add_argument("--output", type=str, default="q3_results.json", help="Output file")
    args = parser.parse_args()

    run_q3_analysis(args.model, args.layer, args.output)


if __name__ == "__main__":
    main()
