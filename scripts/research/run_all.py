#!/usr/bin/env python3
"""
Run All Research Analyses

Execute all four research questions in sequence and generate a combined report.

Usage:
    python -m scripts.research.run_all --model "Qwen/Qwen3-8B" --output results/
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from .common import load_activations_from_db
from .q1_strategy_comparison import analyze_strategy_performance, summarize_strategy_results
from .q2_repscan_correlation import analyze_repscan_correlation, interpret_correlations
from .q3_benchmark_maximum import analyze_per_benchmark_maximum
from .q4_unified_direction import analyze_unified_direction


def run_full_analysis(model_name: str, layer: int = None, output_dir: str = None) -> Dict[str, Any]:
    """
    Run complete research analysis for all 4 questions.

    Args:
        model_name: Model to analyze
        layer: Layer to use (default: middle)
        output_dir: Directory to save individual question results

    Returns:
        Combined results dict
    """
    print(f"\n{'='*60}")
    print(f"Research Analysis for {model_name}")
    print(f"{'='*60}\n")

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load data once
    print("Loading activations from database...")
    activations = load_activations_from_db(model_name, layer)
    total_activations = sum(len(v) for v in activations.values())
    num_benchmarks = len(activations)
    print(f"Loaded {total_activations} activations across {num_benchmarks} benchmarks\n")

    # Research Question 1
    print("=" * 60)
    print("Q1: Which extraction strategy outperforms all the others?")
    print("=" * 60)
    strategy_results = analyze_strategy_performance(activations)
    q1_summary = summarize_strategy_results(strategy_results)

    print("\nStrategy Win Counts:")
    for strategy, count in sorted(q1_summary["strategy_win_counts"].items(), key=lambda x: -x[1])[:5]:
        print(f"  {strategy}: {count} benchmarks")
    print(f"\nOverall Best: {q1_summary['overall_best_strategy']}")

    # Research Question 2
    print("\n" + "=" * 60)
    print("Q2: Is RepScan effective at predicting steering?")
    print("=" * 60)
    q2_results = analyze_repscan_correlation(activations)

    if "correlations" in q2_results:
        print("\nCorrelations with steering accuracy:")
        for metric, corr in q2_results["correlations"].items():
            if "pearson_r" in corr:
                sig = "*" if corr["is_significant_pearson"] else ""
                print(f"  {metric}: r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.4f}){sig}")

    # Research Question 3
    print("\n" + "=" * 60)
    print("Q3: What is the maximum we can achieve per benchmark?")
    print("=" * 60)
    q3_results = analyze_per_benchmark_maximum(strategy_results)

    if "summary" in q3_results:
        s = q3_results["summary"]
        print(f"\nMean best accuracy:   {s['mean_best_accuracy']:.3f}")
        print(f"Median best accuracy: {s['median_best_accuracy']:.3f}")
        print(f"Range: [{s['min_best_accuracy']:.3f}, {s['max_best_accuracy']:.3f}]")

    # Research Question 4
    print("\n" + "=" * 60)
    print("Q4: Is there a unified direction?")
    print("=" * 60)
    q4_results = analyze_unified_direction(activations)

    if "summary" in q4_results:
        s = q4_results["summary"]
        print(f"\nUnified mean accuracy:       {s['unified_mean_accuracy']:.3f}")
        print(f"Per-benchmark mean accuracy: {s['per_benchmark_mean_accuracy']:.3f}")
        print(f"Unified wins: {s['unified_wins_count']}/{s['total_benchmarks']} ({s['unified_win_rate']:.1%})")

    # Compile results
    results = {
        "model": model_name,
        "layer": layer,
        "timestamp": datetime.now().isoformat(),
        "data_summary": {
            "total_activations": total_activations,
            "num_benchmarks": num_benchmarks,
        },
        "research_questions": {
            "q1_strategy_comparison": {
                "description": "Which extraction strategy outperforms all the others?",
                "summary": q1_summary,
                "per_benchmark": {
                    b: {
                        "best_strategy": r.best_strategy,
                        "best_accuracy": r.best_accuracy,
                        "all_strategies": r.strategies,
                    }
                    for b, r in strategy_results.items()
                },
            },
            "q2_repscan_correlation": {
                "description": "Is RepScan effective at predicting steering?",
                "results": q2_results,
                "interpretations": interpret_correlations(q2_results.get("correlations", {})) if "correlations" in q2_results else {},
            },
            "q3_per_benchmark_maximum": {
                "description": "What is the maximum we can achieve per benchmark?",
                "results": q3_results,
            },
            "q4_unified_direction": {
                "description": "Is there a unified direction that improves performance over all benchmarks?",
                "results": q4_results,
            },
        },
    }

    # Save individual results
    if output_dir:
        for q_name, q_data in results["research_questions"].items():
            q_path = os.path.join(output_dir, f"{q_name}.json")
            with open(q_path, 'w') as f:
                json.dump(q_data, f, indent=2, default=str)
            print(f"Saved {q_path}")

        # Save combined results
        combined_path = os.path.join(output_dir, "all_results.json")
        with open(combined_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nCombined results saved to {combined_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("RESEARCH SUMMARY")
    print("=" * 60)

    print(f"\nModel: {model_name}")
    print(f"Layer: {layer}")
    print(f"Benchmarks analyzed: {num_benchmarks}")
    print(f"Total activation pairs: {total_activations}")

    print("\nKey Findings:")
    print(f"  Q1: Best extraction strategy is {q1_summary['overall_best_strategy']}")

    if "correlations" in q2_results:
        best_corr = max(
            [(m, c.get("pearson_r", 0)) for m, c in q2_results["correlations"].items() if "pearson_r" in c],
            key=lambda x: abs(x[1]),
            default=(None, 0)
        )
        if best_corr[0]:
            print(f"  Q2: Best predictor is {best_corr[0]} (r={best_corr[1]:.3f})")

    if "summary" in q3_results:
        print(f"  Q3: Mean achievable accuracy is {q3_results['summary']['mean_best_accuracy']:.3f}")

    if "summary" in q4_results:
        print(f"  Q4: Unified direction wins {q4_results['summary']['unified_win_rate']:.1%} of benchmarks")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run all research analyses")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'Qwen/Qwen3-8B')")
    parser.add_argument("--layer", type=int, default=None, help="Layer to analyze (default: middle)")
    parser.add_argument("--output", type=str, default="research_results", help="Output directory")
    args = parser.parse_args()

    run_full_analysis(args.model, args.layer, args.output)


if __name__ == "__main__":
    main()
