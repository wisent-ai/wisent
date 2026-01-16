#!/usr/bin/env python3
"""
Research Question 1: Which extraction strategy outperforms all the others?

This module evaluates each extraction strategy's steering effectiveness across
all benchmarks and determines which strategy consistently performs best.

Extraction strategies compared:
- CHAT_LAST: Last token of chat-formatted input
- CHAT_MEAN: Mean of all tokens in chat-formatted input
- MC_BALANCED: Multiple-choice with balanced position
- And others depending on benchmark type

Methodology:
1. For each benchmark, group activations by extraction strategy
2. Train steering vectors using 80% of pairs
3. Evaluate on 20% held-out pairs using pairwise accuracy
4. Compare CAA (mean difference) and Hyperplane (logistic regression) methods
5. Aggregate results to find overall best strategy
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np

from .common import (
    ActivationData,
    BenchmarkResults,
    load_activations_from_db,
    compute_steering_accuracy,
)


def analyze_strategy_performance(
    activations_by_benchmark: Dict[str, List[ActivationData]]
) -> Dict[str, BenchmarkResults]:
    """
    Evaluate each extraction strategy's steering effectiveness.

    For each benchmark:
    1. Group activations by extraction strategy
    2. For each strategy with >= 10 pairs:
       - Split into 80/20 train/test
       - Compute CAA steering accuracy
       - Compute Hyperplane steering accuracy
    3. Track best strategy per benchmark

    Args:
        activations_by_benchmark: Dict mapping benchmark names to activation data

    Returns:
        Dict mapping benchmark names to BenchmarkResults with strategy metrics
    """
    results = {}

    for benchmark, activations in activations_by_benchmark.items():
        # Group by strategy
        by_strategy = defaultdict(list)
        for act in activations:
            by_strategy[act.strategy].append(act)

        if not by_strategy:
            continue

        bench_result = BenchmarkResults(name=benchmark, num_pairs=len(activations))

        for strategy, acts in by_strategy.items():
            if len(acts) < 10:
                continue

            pos = np.array([a.positive_activation for a in acts])
            neg = np.array([a.negative_activation for a in acts])

            # Train/test split
            n = len(pos)
            train_idx = int(n * 0.8)

            train_pos, test_pos = pos[:train_idx], pos[train_idx:]
            train_neg, test_neg = neg[:train_idx], neg[train_idx:]

            if len(test_pos) < 2:
                continue

            # Evaluate CAA and Hyperplane methods
            caa_acc = compute_steering_accuracy(train_pos, train_neg, test_pos, test_neg, "caa")
            hp_acc = compute_steering_accuracy(train_pos, train_neg, test_pos, test_neg, "hyperplane")

            bench_result.strategies[strategy] = {
                "caa_accuracy": caa_acc,
                "hyperplane_accuracy": hp_acc,
                "best_accuracy": max(caa_acc, hp_acc),
                "num_pairs": len(acts),
            }

            if max(caa_acc, hp_acc) > bench_result.best_accuracy:
                bench_result.best_accuracy = max(caa_acc, hp_acc)
                bench_result.best_strategy = strategy

        results[benchmark] = bench_result

    return results


def summarize_strategy_results(results: Dict[str, BenchmarkResults]) -> Dict[str, Any]:
    """
    Generate summary statistics for strategy comparison.

    Returns:
        Summary with:
        - strategy_win_counts: How many benchmarks each strategy wins
        - strategy_avg_accuracy: Average accuracy per strategy
        - overall_best_strategy: Strategy with most wins
        - accuracy_by_method: CAA vs Hyperplane comparison
    """
    strategy_wins = defaultdict(int)
    strategy_accuracies = defaultdict(list)
    caa_accuracies = []
    hp_accuracies = []

    for benchmark, result in results.items():
        if result.best_strategy:
            strategy_wins[result.best_strategy] += 1

        for strategy, metrics in result.strategies.items():
            strategy_accuracies[strategy].append(metrics["best_accuracy"])
            caa_accuracies.append(metrics["caa_accuracy"])
            hp_accuracies.append(metrics["hyperplane_accuracy"])

    # Find overall best strategy
    overall_best = max(strategy_wins.keys(), key=lambda s: strategy_wins[s]) if strategy_wins else None

    return {
        "strategy_win_counts": dict(strategy_wins),
        "strategy_avg_accuracy": {
            s: float(np.mean(accs)) for s, accs in strategy_accuracies.items()
        },
        "overall_best_strategy": overall_best,
        "accuracy_by_method": {
            "caa_mean": float(np.mean(caa_accuracies)) if caa_accuracies else 0.0,
            "hyperplane_mean": float(np.mean(hp_accuracies)) if hp_accuracies else 0.0,
            "caa_wins": sum(1 for c, h in zip(caa_accuracies, hp_accuracies) if c > h),
            "hyperplane_wins": sum(1 for c, h in zip(caa_accuracies, hp_accuracies) if h > c),
        },
        "num_benchmarks_analyzed": len(results),
    }


def print_results(results: Dict[str, BenchmarkResults], summary: Dict[str, Any]):
    """Print formatted results to stdout."""
    print("\n" + "=" * 60)
    print("Q1: EXTRACTION STRATEGY COMPARISON")
    print("=" * 60)

    print("\nStrategy Win Counts (best strategy per benchmark):")
    print("-" * 40)
    for strategy, count in sorted(summary["strategy_win_counts"].items(), key=lambda x: -x[1]):
        print(f"  {strategy}: {count} benchmarks")

    print(f"\nOverall Best Strategy: {summary['overall_best_strategy']}")

    print("\nAverage Accuracy by Strategy:")
    print("-" * 40)
    for strategy, acc in sorted(summary["strategy_avg_accuracy"].items(), key=lambda x: -x[1]):
        print(f"  {strategy}: {acc:.3f}")

    print("\nCAA vs Hyperplane Method:")
    print("-" * 40)
    method = summary["accuracy_by_method"]
    print(f"  CAA mean accuracy: {method['caa_mean']:.3f}")
    print(f"  Hyperplane mean accuracy: {method['hyperplane_mean']:.3f}")
    print(f"  CAA wins: {method['caa_wins']}, Hyperplane wins: {method['hyperplane_wins']}")


def run_q1_analysis(model_name: str, layer: int = None, output_path: str = None) -> Dict[str, Any]:
    """
    Run complete Q1 analysis.

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
    results = analyze_strategy_performance(activations)
    summary = summarize_strategy_results(results)

    print_results(results, summary)

    output = {
        "model": model_name,
        "layer": layer,
        "question": "Q1: Which extraction strategy outperforms all the others?",
        "summary": summary,
        "per_benchmark": {
            b: {
                "best_strategy": r.best_strategy,
                "best_accuracy": r.best_accuracy,
                "all_strategies": r.strategies,
            }
            for b, r in results.items()
        },
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Q1: Which extraction strategy outperforms all the others?"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--layer", type=int, default=None, help="Layer (default: middle)")
    parser.add_argument("--output", type=str, default="q1_results.json", help="Output file")
    args = parser.parse_args()

    run_q1_analysis(args.model, args.layer, args.output)


if __name__ == "__main__":
    main()
