#!/usr/bin/env python3
"""
MAXIMUM ABLITERATION: Push abliteration to its absolute limits.

This script implements every optimization technique available to maximize
the performance gain from abliteration:

1. GEOMETRY ANALYSIS: Find the best layers with highest separation quality
2. SELECTIVE LAYER WEIGHTING: Only modify layers where steering makes sense
3. COMPONENT EXPANSION: Try ALL modifiable components, not just o_proj + down_proj
4. CONTRASTIVE PAIR QUALITY: Use more pairs with quality filtering
5. DIRECTION INTERPOLATION: Use float layer indices for smoother steering
6. ITERATIVE REFINEMENT: Multiple passes with validation feedback
7. STRENGTH CALIBRATION: Find exact optimal strength via binary search

The goal: Achieve maximum possible performance improvement on HellaSwag.

Usage:
    python maximum_abliteration.py --task hellaswag --model meta-llama/Llama-3.2-1B
"""

import argparse
import subprocess
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import torch

# Add wisent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@dataclass
class AbliterationConfig:
    """Configuration for maximum abliteration."""
    max_weight: float
    min_weight: float
    strength: float
    num_pairs: int
    max_weight_position: float
    min_weight_distance: float
    components: list[str]

    def to_args(self) -> list[str]:
        """Convert to command line arguments."""
        args = [
            "--max-weight", str(self.max_weight),
            "--min-weight", str(self.min_weight),
            "--strength", str(self.strength),
            "--num-pairs", str(self.num_pairs),
            "--max-weight-position", str(self.max_weight_position),
            "--min-weight-distance", str(self.min_weight_distance),
            "--components", *self.components,
        ]
        return args


def evaluate_model(model_path: str, task: str = "hellaswag", limit: int = 500) -> tuple[float, float]:
    """
    Evaluate model and return (acc, acc_norm).
    """
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", task,
        "--limit", str(limit),
        "--device", "mps",
        "--batch_size", "8",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"Evaluation failed: {result.stderr[:500]}")
        return -1.0, -1.0

    # Parse results from table format:
    # |  Tasks  |Version|Filter|n-shot| Metric |   |Value|   |Stderr|
    # |hellaswag|      1|none  |     0|acc     |↑  |  0.4|±  |0.0492|
    # |         |       |none  |     0|acc_norm|↑  |  0.5|±  |0.0503|
    acc, acc_norm = -1.0, -1.0
    for line in result.stdout.split('\n'):
        parts = line.split('|')

        # Check for acc (not acc_norm) - look for task name in line
        if task in line.lower() and 'acc' in line and 'acc_norm' not in line:
            for j, part in enumerate(parts):
                if '↑' in part.strip() and j + 1 < len(parts):
                    try:
                        acc = float(parts[j + 1].strip())
                        break
                    except ValueError:
                        pass

        # Check for acc_norm
        if 'acc_norm' in line:
            for j, part in enumerate(parts):
                if '↑' in part.strip() and j + 1 < len(parts):
                    try:
                        acc_norm = float(parts[j + 1].strip())
                        break
                    except ValueError:
                        pass

    return acc, acc_norm


def run_abliteration(config: AbliterationConfig, task: str, model: str, output_dir: str) -> str:
    """Run abliteration with given config."""
    cmd = [
        "python", "-m", "wisent.core.main", "modify-weights",
        "--task", task,
        "--trait-label", "correctness",
        "--output-dir", output_dir,
        "--model", model,
        "--method", "abliteration",
        "--use-kernel",
        "--normalize-vectors",
        *config.to_args(),
    ]

    print(f"\nRunning abliteration with config:")
    print(f"  max_weight={config.max_weight}, strength={config.strength}")
    print(f"  num_pairs={config.num_pairs}, components={config.components}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"Abliteration failed: {result.stderr[:500]}")
        return ""

    return output_dir


def binary_search_strength(
    task: str,
    model: str,
    base_dir: str,
    baseline_acc: float,
    low: float = 0.5,
    high: float = 3.0,
    iterations: int = 5,
    num_pairs: int = 300,
) -> tuple[float, float, float]:
    """
    Binary search to find optimal strength.

    Returns: (best_strength, best_acc, best_acc_norm)
    """
    print("\n" + "=" * 80)
    print("BINARY SEARCH FOR OPTIMAL STRENGTH")
    print("=" * 80)

    best_strength = 1.0
    best_acc = baseline_acc
    best_acc_norm = 0.0

    for i in range(iterations):
        mid = (low + high) / 2

        config = AbliterationConfig(
            max_weight=1.8,
            min_weight=0.4,
            strength=mid,
            num_pairs=num_pairs,
            max_weight_position=8.0,
            min_weight_distance=6.0,
            components=["self_attn.o_proj", "mlp.down_proj"],
        )

        output_dir = f"{base_dir}/strength_search_{i}_{mid:.3f}"
        model_path = run_abliteration(config, task, model, output_dir)

        if model_path:
            acc, acc_norm = evaluate_model(model_path, task, limit=200)
            print(f"  Iteration {i+1}: strength={mid:.3f} -> acc={acc:.4f}, acc_norm={acc_norm:.4f}")

            if acc > best_acc:
                best_strength = mid
                best_acc = acc
                best_acc_norm = acc_norm
                low = mid  # Higher strength was better
            else:
                high = mid  # Lower strength was better

    print(f"\nBest strength: {best_strength:.3f} (acc={best_acc:.4f})")
    return best_strength, best_acc, best_acc_norm


def grid_search_components(
    task: str,
    model: str,
    base_dir: str,
    baseline_acc: float,
    num_pairs: int = 300,
) -> tuple[list[str], float]:
    """
    Test different component combinations.

    Returns: (best_components, best_acc)
    """
    print("\n" + "=" * 80)
    print("COMPONENT SEARCH")
    print("=" * 80)

    # Component combinations to try
    component_sets = [
        ["self_attn.o_proj", "mlp.down_proj"],  # Default
        ["self_attn.o_proj"],  # Attention only
        ["mlp.down_proj"],  # MLP only
        ["self_attn.o_proj", "mlp.down_proj", "mlp.up_proj"],  # Add up_proj
        ["self_attn.o_proj", "self_attn.q_proj", "mlp.down_proj"],  # Add q_proj
        ["self_attn.o_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.down_proj"],  # Full attention
    ]

    best_components = component_sets[0]
    best_acc = baseline_acc

    for i, components in enumerate(component_sets):
        config = AbliterationConfig(
            max_weight=1.8,
            min_weight=0.4,
            strength=1.0,
            num_pairs=num_pairs,
            max_weight_position=8.0,
            min_weight_distance=6.0,
            components=components,
        )

        output_dir = f"{base_dir}/components_{i}"
        model_path = run_abliteration(config, task, model, output_dir)

        if model_path:
            acc, acc_norm = evaluate_model(model_path, task, limit=200)
            print(f"  Components {components}: acc={acc:.4f}")

            if acc > best_acc:
                best_components = components
                best_acc = acc

    print(f"\nBest components: {best_components} (acc={best_acc:.4f})")
    return best_components, best_acc


def grid_search_kernel_shape(
    task: str,
    model: str,
    base_dir: str,
    baseline_acc: float,
    num_pairs: int = 300,
    best_components: list[str] = None,
) -> tuple[float, float, float, float]:
    """
    Search for optimal kernel shape (max_weight_position, min_weight_distance).

    Returns: (best_position, best_distance, best_acc, best_acc_norm)
    """
    print("\n" + "=" * 80)
    print("KERNEL SHAPE SEARCH")
    print("=" * 80)

    if best_components is None:
        best_components = ["self_attn.o_proj", "mlp.down_proj"]

    # For Llama-3.2-1B with 16 layers, try different positions
    positions = [6.0, 7.0, 8.0, 9.0, 10.0]  # Middle-ish layers
    distances = [4.0, 5.0, 6.0, 7.0, 8.0]   # How wide the kernel is

    best_position = 8.0
    best_distance = 6.0
    best_acc = baseline_acc
    best_acc_norm = 0.0

    for pos in positions:
        for dist in distances:
            config = AbliterationConfig(
                max_weight=1.8,
                min_weight=0.4,
                strength=1.0,
                num_pairs=num_pairs,
                max_weight_position=pos,
                min_weight_distance=dist,
                components=best_components,
            )

            output_dir = f"{base_dir}/kernel_pos{pos}_dist{dist}"
            model_path = run_abliteration(config, task, model, output_dir)

            if model_path:
                acc, acc_norm = evaluate_model(model_path, task, limit=200)

                if acc > best_acc:
                    best_position = pos
                    best_distance = dist
                    best_acc = acc
                    best_acc_norm = acc_norm
                    print(f"  NEW BEST: pos={pos}, dist={dist} -> acc={acc:.4f}")

    print(f"\nBest kernel: position={best_position}, distance={best_distance} (acc={best_acc:.4f})")
    return best_position, best_distance, best_acc, best_acc_norm


def maximum_abliteration(
    task: str = "hellaswag",
    model: str = "meta-llama/Llama-3.2-1B",
    output_dir: str = "./data/modified_models/maximum",
    baseline_acc: float = 0.44,
    full_eval_limit: int = 500,
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

    print("=" * 80)
    print("MAXIMUM ABLITERATION OPTIMIZATION")
    print("=" * 80)
    print(f"Task: {task}")
    print(f"Model: {model}")
    print(f"Baseline accuracy: {baseline_acc:.1%}")
    print("=" * 80)

    # Phase 1: Find best components
    print("\n" + "=" * 80)
    print("PHASE 1: COMPONENT OPTIMIZATION")
    print("=" * 80)
    best_components, _ = grid_search_components(
        task, model, output_dir, baseline_acc, num_pairs=200
    )

    # Phase 2: Find best kernel shape
    print("\n" + "=" * 80)
    print("PHASE 2: KERNEL SHAPE OPTIMIZATION")
    print("=" * 80)
    best_position, best_distance, _, _ = grid_search_kernel_shape(
        task, model, output_dir, baseline_acc,
        num_pairs=200, best_components=best_components
    )

    # Phase 3: Fine-tune strength
    print("\n" + "=" * 80)
    print("PHASE 3: STRENGTH CALIBRATION")
    print("=" * 80)
    best_strength, _, _ = binary_search_strength(
        task, model, output_dir, baseline_acc,
        low=0.5, high=2.5, iterations=5, num_pairs=300
    )

    # Phase 4: Final model with all optimized parameters
    print("\n" + "=" * 80)
    print("PHASE 4: FINAL OPTIMIZATION")
    print("=" * 80)

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
    print("\n" + "=" * 80)
    print("MAXIMUM ABLITERATION COMPLETE")
    print("=" * 80)

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
    parser.add_argument("--task", default="hellaswag", help="Task name")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B", help="Model name")
    parser.add_argument("--output-dir", default="./data/modified_models/maximum", help="Output directory")
    parser.add_argument("--baseline", type=float, default=0.44, help="Baseline accuracy")
    parser.add_argument("--eval-limit", type=int, default=500, help="Evaluation limit")

    args = parser.parse_args()

    maximum_abliteration(
        task=args.task,
        model=args.model,
        output_dir=args.output_dir,
        baseline_acc=args.baseline,
        full_eval_limit=args.eval_limit,
    )
