#!/usr/bin/env python3
"""
Signal detection script for contrastive pairs.

Checks if a concept has extractable geometric structure in activation space.
Tests both linear and nonlinear separability.

Usage:
    python scripts/check_signal.py --task truthfulqa --model meta-llama/Llama-3.2-1B-Instruct
    python scripts/check_signal.py --task livecodebench --model Qwen/Qwen3-8B
    python scripts/check_signal.py --task sentiment --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse

from wisent.core.utils.config_tools.constants import PAIR_GENERATORS_DEFAULT_N

from .check_signal_helpers import (
    load_model,
    get_truthfulqa_pairs,
    get_livecodebench_pairs,
    get_sentiment_pairs,
    collect_activations,
    compute_signal_metrics,
    evaluate_signal,
)


def main():
    parser = argparse.ArgumentParser(description="Check signal in contrastive pairs")
    parser.add_argument("--task", type=str, required=True,
                        choices=["truthfulqa", "livecodebench", "sentiment"])
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--n-samples", type=int, default=PAIR_GENERATORS_DEFAULT_N)
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=["chat_mean", "role_play", "mc_balanced"])
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Load data
    print(f"\nLoading {args.task} data...")
    if args.task == "truthfulqa":
        pairs = get_truthfulqa_pairs(args.n_samples)
    elif args.task == "livecodebench":
        pairs = get_livecodebench_pairs(args.n_samples)
    elif args.task == "sentiment":
        pairs = get_sentiment_pairs(args.n_samples)

    print(f"Loaded {len(pairs)} pairs")

    # Load model
    model, tokenizer = load_model(args.model, args.device)
    num_layers = model.config.num_hidden_layers
    device = next(model.parameters()).device

    print(f"\n{'='*80}")
    print(f"SIGNAL CHECK: {args.task.upper()} on {args.model}")
    print(f"{'='*80}")

    for strategy in args.strategies:
        print(f"\n--- Strategy: {strategy} ---")

        # Collect activations
        print("Collecting activations...")
        pos_acts, neg_acts = collect_activations(
            model, tokenizer, pairs, strategy, device, num_layers
        )

        # Find best layer by diff cosine
        best_layer = 0
        best_cosine = -1
        layer_results = []

        for layer in range(num_layers + 1):
            results = compute_signal_metrics(pos_acts[layer], neg_acts[layer])
            layer_results.append((layer, results))
            if results["diff_cosine_mean"] > best_cosine:
                best_cosine = results["diff_cosine_mean"]
                best_layer = layer

        print(f"\nBest layer: {best_layer} (cosine: {best_cosine:.4f})")

        # Full analysis on best layer
        results = compute_signal_metrics(pos_acts[best_layer], neg_acts[best_layer])

        print(f"\n{'Metric':<35} {'Value':>10} {'Threshold':>12} {'Pass':>6}")
        print("-" * 65)

        checks = evaluate_signal(results)
        for name, passed, value in checks:
            status = "YES" if passed else "NO"
            if isinstance(value, float):
                print(f"{name:<35} {value:>10.4f} {'-':>12} {status:>6}")
            else:
                print(f"{name:<35} {str(value):>10} {'-':>12} {status:>6}")

        # Summary
        linear_signal = results["diff_cosine_mean"] > 0.2
        nonlinear_signal = results["mlp_accuracy"] > 0.7 or results["svm_accuracy"] > 0.7

        print(f"\n{'='*65}")
        if linear_signal:
            print(f"RESULT: LINEAR SIGNAL EXISTS (cosine={results['diff_cosine_mean']:.3f})")
            print(f"        Activation steering should work.")
        elif nonlinear_signal:
            print(f"RESULT: NONLINEAR SIGNAL EXISTS (MLP={results['mlp_accuracy']:.3f})")
            print(f"        Linear steering may not work. Consider nonlinear methods.")
        else:
            print(f"RESULT: NO CLEAR SIGNAL DETECTED")
            print(f"        This concept may not have geometric structure in this model.")
        print(f"{'='*65}")

        # Layer-by-layer summary
        print(f"\nLayer-by-layer cosine (top 5):")
        sorted_layers = sorted(layer_results, key=lambda x: x[1]["diff_cosine_mean"], reverse=True)[:5]
        for layer, res in sorted_layers:
            print(f"  Layer {layer:>2}: cosine={res['diff_cosine_mean']:.4f}, "
                  f"PC1={res['pc1_variance']:.2%}, MLP={res['mlp_accuracy']:.2%}")


if __name__ == "__main__":
    main()
