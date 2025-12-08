#!/usr/bin/env python3
"""
Optimize abliteration parameters using Optuna for intelligent parameter search.

This example demonstrates how to use the `optimize-weights` CLI command
to find optimal weight modification parameters for any task.

Usage:
    # For benchmark-based optimization:
    wisent optimize-weights \
        --task hellaswag \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --trials 30 \
        --output-dir ./data/modified_models/optuna

    # For refusal optimization:
    wisent optimize-weights \
        --task refusal \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --target-metric compliance_rate \
        --target-value 0.95 \
        --trials 30 \
        --output-dir ./data/modified_models/optuna

    # For personalization optimization:
    wisent optimize-weights \
        --task personalization \
        --trait "a pirate who speaks in nautical terms" \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --trials 30 \
        --output-dir ./data/modified_models/optuna

    # For custom evaluator optimization:
    wisent optimize-weights \
        --task custom \
        --trait "human-like writing style" \
        --custom-evaluator wisent.core.evaluators.custom.examples.gptzero \
        --custom-evaluator-kwargs '{"api_key": "YOUR_KEY"}' \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --trials 30 \
        --output-dir ./data/modified_models/optuna

    # Or run this script for a complete example:
    python optimize_abliteration_optuna.py --task hellaswag --trials 30
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Optimize abliteration parameters using the optimize-weights command"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        required=True,
        help="Task type: 'refusal', 'personalization', benchmark name (e.g., 'hellaswag'), or comma-separated benchmarks",
    )
    parser.add_argument(
        "--trait",
        type=str,
        default=None,
        help="Trait description (required when --task personalization)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/modified_models/optuna",
        help="Output directory for optimized model",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=100,
        help="Number of contrastive pairs for training",
    )
    parser.add_argument(
        "--num-eval-prompts",
        type=int,
        default=50,
        help="Number of prompts for evaluation",
    )

    args = parser.parse_args()

    # Validate personalization requires trait
    if args.task.lower() == "personalization" and not args.trait:
        print("Error: --trait is required when --task personalization")
        sys.exit(1)

    print("=" * 80)
    print("ABLITERATION PARAMETER OPTIMIZATION (OPTUNA)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    if args.trait:
        print(f"Trait: {args.trait}")
    print(f"Trials: {args.trials}")
    print("=" * 80 + "\n")

    # Build the optimize-weights command
    cmd = [
        "python", "-m", "wisent.core.main", "optimize-weights",
        args.model,
        "--trials", str(args.trials),
        "--num-pairs", str(args.num_pairs),
        "--num-eval-prompts", str(args.num_eval_prompts),
        "--output-dir", args.output_dir,
        "--method", "abliteration",
        "--early-stop",
    ]

    cmd.extend(["--task", args.task])
    if args.trait:
        cmd.extend(["--trait", args.trait])

    print(f"Running: {' '.join(cmd)}\n")

    # Run the command
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✅ Optimization complete!")
        print(f"Optimized model saved to: {args.output_dir}")
    else:
        print(f"\n❌ Optimization failed with return code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
