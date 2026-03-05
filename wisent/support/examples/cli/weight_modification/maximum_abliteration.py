#!/usr/bin/env python3
"""MAXIMUM ABLITERATION: Push abliteration to its absolute limits."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from wisent.core.utils.config_tools.constants import SEPARATOR_WIDTH_REPORT
from wisent.examples.cli.weight_modification.maximum_abliteration_helpers import (
    AbliterationConfig,
    evaluate_model,
    run_abliteration,
    binary_search_strength,
    grid_search_components,
    grid_search_kernel_shape,
)


def maximum_abliteration(
    task: str,
    model: str,
    output_dir: str,
    baseline_acc: float,
    *,
    full_eval_limit: int,
):
    """
    Run maximum abliteration optimization.

    This performs multiple optimization passes:
    1. Component search
    2. Kernel shape search
    3. Strength calibration
    4. Final validation
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * SEPARATOR_WIDTH_REPORT)
    print("MAXIMUM ABLITERATION OPTIMIZATION")
    print("=" * SEPARATOR_WIDTH_REPORT)
    print(f"Task: {task}")
    print(f"Model: {model}")
    print(f"Baseline accuracy: {baseline_acc:.1%}")
    print("=" * SEPARATOR_WIDTH_REPORT)

    # Phase 1: Find best components
    print("\n" + "=" * SEPARATOR_WIDTH_REPORT)
    print("PHASE 1: COMPONENT OPTIMIZATION")
    print("=" * SEPARATOR_WIDTH_REPORT)
    best_components, _ = grid_search_components(
        task, model, output_dir, baseline_acc,
        limit=full_eval_limit,
    )

    # Phase 2: Find best kernel shape
    print("\n" + "=" * SEPARATOR_WIDTH_REPORT)
    print("PHASE 2: KERNEL SHAPE OPTIMIZATION")
    print("=" * SEPARATOR_WIDTH_REPORT)
    best_position, best_distance, _, _ = grid_search_kernel_shape(
        task, model, output_dir, baseline_acc,
        best_components=best_components,
        limit=full_eval_limit,
    )

    # Phase 3: Fine-tune strength
    print("\n" + "=" * SEPARATOR_WIDTH_REPORT)
    print("PHASE 3: STRENGTH CALIBRATION")
    print("=" * SEPARATOR_WIDTH_REPORT)
    best_strength, _, _ = binary_search_strength(
        task, model, output_dir, baseline_acc,
        limit=full_eval_limit,
    )

    # Phase 4: Final model with all optimized parameters
    print("\n" + "=" * SEPARATOR_WIDTH_REPORT)
    print("PHASE 4: FINAL OPTIMIZATION")
    print("=" * SEPARATOR_WIDTH_REPORT)

    # Try different num_pairs with optimized config
    best_acc = baseline_acc
    best_config = None
    best_model_path = None

    for num_pairs in [300, 400, 500, 600]:
        config = AbliterationConfig(
            max_weight=1.8,
            min_weight=0.4,
            strength=best_strength,
            num_pairs=num_pairs,
            max_weight_position=best_position,
            min_weight_distance=best_distance,
            components=best_components,
        )

        model_path = f"{output_dir}/final_pairs{num_pairs}"
        run_abliteration(config, task, model, model_path)

        acc, acc_norm = evaluate_model(model_path, task, limit=full_eval_limit)
        gain = (acc - baseline_acc) * 100

        print(f"  num_pairs={num_pairs}: acc={acc:.4f} (gain={gain:+.2f}%)")

        if acc > best_acc:
            best_acc = acc
            best_config = config
            best_model_path = model_path

    # Final report
    print("\n" + "=" * SEPARATOR_WIDTH_REPORT)
    print("MAXIMUM ABLITERATION COMPLETE")
    print("=" * SEPARATOR_WIDTH_REPORT)

    final_gain = (best_acc - baseline_acc) * 100

    print(f"\nBest Configuration:")
    print(f"  max_weight: {best_config.max_weight}")
    print(f"  min_weight: {best_config.min_weight}")
    print(f"  strength: {best_config.strength}")
    print(f"  num_pairs: {best_config.num_pairs}")
    print(f"  max_weight_position: {best_config.max_weight_position}")
    print(f"  min_weight_distance: {best_config.min_weight_distance}")
    print(f"  components: {best_config.components}")

    print(f"\nResults:")
    print(f"  Baseline accuracy: {baseline_acc:.1%}")
    print(f"  Best accuracy: {best_acc:.1%}")
    print(f"  Performance gain: {final_gain:+.2f} percentage points")
    print(f"\nBest model saved to: {best_model_path}")

    if final_gain >= 40:
        print("\n🎉 SUCCESS! Achieved +40% goal!")
    elif final_gain >= 10:
        print(f"\n✅ Strong gain of {final_gain:+.2f}%")
    elif final_gain >= 5:
        print(f"\n📈 Good gain of {final_gain:+.2f}%")
    else:
        print(f"\n⚠️ Limited gain of {final_gain:+.2f}% - technique may have hit ceiling")

    return best_model_path, best_acc, best_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maximum abliteration optimization")
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--baseline", type=float, required=True, help="Baseline accuracy")
    parser.add_argument("--eval-limit", type=int, required=True, help="Evaluation limit")

    args = parser.parse_args()

    maximum_abliteration(
        task=args.task,
        model=args.model,
        output_dir=args.output_dir,
        baseline_acc=args.baseline,
        full_eval_limit=args.eval_limit,
    )
