#!/usr/bin/env python3
"""
Test activation-based proxy metric for steering evaluation.

Instead of running full generation + evaluation (~50 min per method),
use held-out activation pairs to compute steering accuracy (~5 sec per method).

For one benchmark:
1. Load activations from DB
2. Split train/test
3. For each method: train steering vector, evaluate on test activations
4. Compute zwiad metrics
5. Compare: does zwiad predict the best method?
"""

from test_activation_proxy_helpers import (
    load_benchmark_activations,
    compute_zwiad_metrics,
    train_caa,
    train_ostrze,
    train_mlp,
    train_pca_based,
    evaluate_steering,
)


def run_benchmark_comparison(model_name: str, benchmark: str, layer: int = None):
    """Run full comparison for one benchmark."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {benchmark}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Load activations
    print("\nLoading activations from DB...")
    activations, num_layers = load_benchmark_activations(model_name, benchmark, layer)

    if not activations:
        print("No activations found!")
        return None

    # Use first available strategy
    strategy = list(activations.keys())[0]
    pos = activations[strategy]["pos"]
    neg = activations[strategy]["neg"]
    print(f"Using strategy: {strategy}")
    print(f"Pairs: {len(pos)}")

    # Train/test split
    n = len(pos)
    train_idx = int(n * 0.8)
    train_pos, test_pos = pos[:train_idx], pos[train_idx:]
    train_neg, test_neg = neg[:train_idx], neg[train_idx:]
    print(f"Train: {len(train_pos)}, Test: {len(test_pos)}")

    # Compute zwiad metrics
    print("\nComputing zwiad metrics...")
    zwiad = compute_zwiad_metrics(train_pos, train_neg)
    print(f"  Linear probe: {zwiad['linear_probe']:.3f}")
    print(f"  Signal strength: {zwiad['signal_strength']:.3f}")
    print(f"  Cluster separation: {zwiad['cluster_sep']:.3f}")

    # Repscan recommendation logic (simplified)
    if zwiad['linear_probe'] > 0.9:
        zwiad_pick = "CAA"
    elif zwiad['signal_strength'] > 2:
        zwiad_pick = "Ostrze"
    else:
        zwiad_pick = "PCA"  # proxy for TECZA/GROM
    print(f"  Repscan recommendation: {zwiad_pick}")

    # Train and evaluate each method
    print("\nTraining and evaluating methods...")
    results = {}

    vec = train_caa(train_pos, train_neg)
    acc = evaluate_steering(vec, test_pos, test_neg)
    results["CAA"] = acc
    print(f"  CAA: {acc:.3f}")

    vec = train_ostrze(train_pos, train_neg)
    acc = evaluate_steering(vec, test_pos, test_neg)
    results["Ostrze"] = acc
    print(f"  Ostrze: {acc:.3f}")

    vec = train_mlp(train_pos, train_neg)
    acc = evaluate_steering(vec, test_pos, test_neg)
    results["MLP"] = acc
    print(f"  MLP: {acc:.3f}")

    vec = train_pca_based(train_pos, train_neg)
    acc = evaluate_steering(vec, test_pos, test_neg)
    results["PCA"] = acc
    print(f"  PCA (TECZA proxy): {acc:.3f}")

    actual_best = max(results, key=results.get)
    print(f"\nActual best: {actual_best} ({results[actual_best]:.3f})")
    print(f"Repscan pick: {zwiad_pick}")
    print(f"Repscan correct: {actual_best == zwiad_pick}")

    return {
        "benchmark": benchmark,
        "zwiad_metrics": zwiad,
        "zwiad_pick": zwiad_pick,
        "method_accuracies": results,
        "actual_best": actual_best,
        "zwiad_correct": actual_best == zwiad_pick,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--benchmark", default="truthfulqa_custom", help="Benchmark name")
    parser.add_argument("--layer", type=int, default=None, help="Layer (default: middle)")
    args = parser.parse_args()

    result = run_benchmark_comparison(args.model, args.benchmark, args.layer)

    if result:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Repscan predicted: {result['zwiad_pick']}")
        print(f"Actual best: {result['actual_best']}")
        print(f"Prediction correct: {result['zwiad_correct']}")
