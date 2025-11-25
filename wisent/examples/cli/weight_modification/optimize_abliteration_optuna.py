#!/usr/bin/env python3
"""
Optimize abliteration parameters using Optuna for intelligent parameter search.

This uses the Optuna framework (same as used for steering optimization) to
intelligently search the parameter space for maximum performance gains.

Key differences from grid search:
- Optuna uses TPE sampler to intelligently explore promising regions
- Can run many more trials efficiently
- Automatically balances exploration vs exploitation
- Same optimization framework used for steering in Wisent

Usage:
    python optimize_abliteration_optuna.py --n-trials 30 --task hellaswag
"""

import argparse
import subprocess
import json
import os
import sys
from pathlib import Path

# Add wisent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from wisent.core.weight_modification.abliteration_optimizer import (
    AbliterationOptimizer,
)


def evaluate_model_on_hellaswag(model_path: str, limit: int = 500) -> float:
    """
    Evaluate a model on HellaSwag using lm-evaluation-harness.

    Args:
        model_path: Path to model directory
        limit: Number of examples to evaluate (for speed)

    Returns:
        acc_norm score (0.0 to 1.0)
    """
    # Create output path
    eval_output_dir = "./data/evals"
    os.makedirs(eval_output_dir, exist_ok=True)
    eval_file = f"{eval_output_dir}/{Path(model_path).name}_hellaswag.json"

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", "hellaswag",
        "--limit", str(limit),
        "--batch_size", "8",
        "--device", "mps",
        "--output_path", eval_file,
    ]

    try:
        print(f"Evaluating {Path(model_path).name}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"Evaluation failed: {result.stderr[:500]}")
            return -1.0

        # Parse results from stdout (table format)
        # Format is:
        # |  Tasks  |Version|Filter|n-shot| Metric |   |Value|   |Stderr|
        # |hellaswag|      1|none  |     0|acc     |↑  |  0.4|±  |0.0492|
        # |         |       |none  |     0|acc_norm|↑  |  0.5|±  |0.0503|
        lines = result.stdout.split('\n')
        for line in lines:
            # Look for acc_norm in the Metric column (index 4 after split)
            if 'acc_norm' in line:
                parts = line.split('|')
                # Find the Value column (comes after the ↑ symbol)
                for j, part in enumerate(parts):
                    if '↑' in part.strip() and j + 1 < len(parts):
                        try:
                            value_str = parts[j + 1].strip()
                            acc_norm = float(value_str)
                            print(f"  acc_norm: {acc_norm:.4f}")
                            return acc_norm
                        except (ValueError, IndexError):
                            pass

                # Alternative: find numeric value after acc_norm
                parts_stripped = [p.strip() for p in parts if p.strip()]
                for j, part in enumerate(parts_stripped):
                    if part == 'acc_norm':
                        # Look for a float in remaining parts
                        for k in range(j + 1, len(parts_stripped)):
                            try:
                                acc_norm = float(parts_stripped[k])
                                if 0.0 <= acc_norm <= 1.0:
                                    print(f"  acc_norm: {acc_norm:.4f}")
                                    return acc_norm
                            except ValueError:
                                continue

        print(f"Could not parse acc_norm from output")
        return -1.0

    except subprocess.TimeoutExpired:
        print(f"Evaluation timed out")
        return -1.0
    except Exception as e:
        print(f"Evaluation exception: {e}")
        return -1.0


def main():
    parser = argparse.ArgumentParser(
        description="Optimize abliteration parameters using Optuna"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Model name",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="hellaswag",
        help="Task name for optimization",
    )
    parser.add_argument(
        "--trait-label",
        type=str,
        default="correctness",
        help="Trait label for contrastive pairs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/modified_models/optuna",
        help="Base output directory",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=10,
        help="Number of random startup trials",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=500,
        help="Number of examples to evaluate (for speed)",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=0.44,
        help="Baseline model accuracy for gain calculation",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ABLITERATION PARAMETER OPTIMIZATION (OPTUNA)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Trials: {args.n_trials}")
    print(f"Baseline: {args.baseline:.1%}")
    print(f"Eval limit: {args.eval_limit} examples")
    print("=" * 80 + "\n")

    # Create optimizer
    optimizer = AbliterationOptimizer(
        model_name=args.model,
        task=args.task,
        trait_label=args.trait_label,
        base_output_dir=args.output_dir,
        evaluate_fn=lambda path: evaluate_model_on_hellaswag(path, args.eval_limit),
        num_layers=16,  # Llama-3.2-1B
        direction="maximize",
    )

    # Run optimization
    print("Starting Optuna optimization...")
    print("This will intelligently explore the parameter space.\n")

    result = optimizer.optimize(
        n_trials=args.n_trials,
        n_startup_trials=args.n_startup_trials,
        show_progress=True,
    )

    # Print results
    optimizer.print_results(result)

    # Calculate gain
    gain = (result.best_score - args.baseline) * 100
    print(f"\nPerformance Gain: {gain:+.1f} percentage points")
    print(f"  Baseline: {args.baseline:.1%}")
    print(f"  Optimized: {result.best_score:.1%}")

    # Show improvement path
    if gain >= 40:
        print(f"\n🎉 SUCCESS! Achieved {gain:+.1f}% gain (target: +40%)")
    elif gain >= 20:
        print(f"\n✅ Strong gain of {gain:+.1f}% (target: +40%, getting close)")
    elif gain >= 10:
        print(f"\n📈 Good gain of {gain:+.1f}% (target: +40%, need more optimization)")
    elif gain >= 5:
        print(f"\n📊 Moderate gain of {gain:+.1f}% (target: +40%, need different approach)")
    else:
        print(f"\n⚠️  Small gain of {gain:+.1f}% (target: +40%, may need alternative method)")

    # Save best model location
    best_model_dir = os.path.join(
        args.output_dir,
        f"{args.task}_optuna_trial_{result.best_trial.number}"
    )

    print(f"\nBest model: {best_model_dir}")
    print(f"\nTo evaluate on full dataset:")
    print(f"  lm_eval --model hf --model_args pretrained={best_model_dir} --tasks {args.task} --device mps")


if __name__ == "__main__":
    main()
