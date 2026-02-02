#!/usr/bin/env python3
"""Analyze checkpoint results from full strategy analysis."""

import json
import argparse
from collections import defaultdict


def analyze_model(data: dict, model: str):
    """Print summary metrics for a model."""
    if model not in data:
        print(f"Model {model} not found in checkpoint")
        return

    model_data = data[model]
    # Handle nested structure: model -> benchmarks -> benchmark_name -> strategies
    if "benchmarks" in model_data:
        results = model_data["benchmarks"]
    else:
        results = model_data

    strategies = ['chat_last', 'chat_first', 'chat_mean', 'role_play', 'mc_balanced', 'mc_completion', 'raw']

    # Aggregate metrics per strategy
    strategy_metrics = defaultdict(lambda: {'linear_acc': [], 'consistency': [], 'steer_acc': [], 'effect_size': []})

    for benchmark, bench_data in results.items():
        # Handle nested strategies dict
        strats = bench_data.get("strategies", bench_data) if isinstance(bench_data, dict) else {}
        for strat, metrics in strats.items():
            if metrics and isinstance(metrics, dict):
                for key in ['linear_acc', 'consistency', 'steer_acc', 'effect_size']:
                    if key in metrics and metrics[key] is not None:
                        val = metrics[key]
                        if isinstance(val, str):
                            val = float(val)
                        strategy_metrics[strat][key].append(val)

    print(f"\nResults for {model}")
    print("=" * 85)
    print(f"{'Strategy':<15} {'Linear Acc':>12} {'Consistency':>12} {'Steer Acc':>12} {'Effect Size':>12} {'N':>6}")
    print("-" * 85)

    for strat in strategies:
        m = strategy_metrics[strat]
        n = len(m['linear_acc'])
        if n > 0:
            la = sum(m['linear_acc']) / n
            co = sum(m['consistency']) / n
            sa = sum(m['steer_acc']) / n
            es = sum(m['effect_size']) / n
            print(f"{strat:<15} {la:>12.4f} {co:>12.4f} {sa:>12.4f} {es:>12.4f} {n:>6}")
        else:
            print(f"{strat:<15} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12} {0:>6}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="strategy_analysis_results/checkpoint.json")
    parser.add_argument("--model", help="Specific model to analyze (default: all)")
    args = parser.parse_args()

    with open(args.checkpoint) as f:
        data = json.load(f)

    print(f"Checkpoint contains {len(data)} models: {list(data.keys())}")

    if args.model:
        analyze_model(data, args.model)
    else:
        for model in data.keys():
            analyze_model(data, model)


if __name__ == "__main__":
    main()
