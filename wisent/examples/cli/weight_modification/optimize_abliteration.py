"""
Optimize abliteration parameters using grid search and evaluation feedback.

This script systematically searches for optimal abliteration parameters by:
1. Testing multiple parameter combinations
2. Evaluating each on the target benchmark
3. Finding the best performing configuration

The goal: Achieve maximum performance gains through systematic parameter optimization.
"""

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
import sys


@dataclass
class AbliterationConfig:
    """Configuration for abliteration parameters."""
    max_weight: float
    min_weight: float
    strength: float
    num_pairs: int
    max_weight_position: float = 8.0
    min_weight_distance: float = 6.0

    def __str__(self):
        return f"max_weight={self.max_weight}_min_weight={self.min_weight}_strength={self.strength}_pairs={self.num_pairs}"


@dataclass
class EvaluationResult:
    """Results from evaluating a configuration."""
    config: AbliterationConfig
    accuracy: float
    acc_norm: float
    model_path: str

    def gain(self, baseline: float) -> float:
        """Calculate performance gain over baseline."""
        return (self.accuracy - baseline) * 100


def run_modification(config: AbliterationConfig, task: str, output_dir: str, model: str) -> str:
    """
    Run weight modification with given configuration.

    Returns:
        Path to modified model
    """
    model_name = f"{task}_{config}"
    model_path = f"{output_dir}/{model_name}"

    cmd = [
        "python", "-m", "wisent.core.main", "modify-weights",
        "--task", task,
        "--trait-label", "correctness",
        "--output-dir", model_path,
        "--model", model,
        "--num-pairs", str(config.num_pairs),
        "--method", "abliteration",
        "--strength", str(config.strength),
        "--components", "self_attn.o_proj", "mlp.down_proj",
        "--use-kernel",
        "--max-weight", str(config.max_weight),
        "--max-weight-position", str(config.max_weight_position),
        "--min-weight", str(config.min_weight),
        "--min-weight-distance", str(config.min_weight_distance),
        "--normalize-vectors",
        "--verbose"
    ]

    print(f"\n{'='*80}")
    print(f"MODIFYING WEIGHTS: {config}")
    print(f"{'='*80}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Weight modification failed")
        print(result.stderr)
        sys.exit(1)

    return model_path


def evaluate_model(model_path: str, task: str, limit: int = 500) -> tuple[float, float]:
    """
    Evaluate model on benchmark.

    Returns:
        (accuracy, acc_norm) tuple
    """
    eval_output = f"./data/evals/{Path(model_path).name}.json"

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", task,
        "--limit", str(limit),
        "--batch_size", "8",
        "--output_path", eval_output,
        "--device", "mps"
    ]

    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_path}")
    print(f"{'='*80}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Evaluation failed")
        print(result.stderr)
        return 0.0, 0.0

    # Parse accuracy from output
    # Format: |hellaswag|      1|none  |     0|acc     |↑  |0.452|±  |0.0223|
    lines = result.stdout.split('\n')
    for line in lines:
        if '|hellaswag|' in line and '|acc' in line:
            parts = line.split('|')
            # Find the accuracy value
            for i, part in enumerate(parts):
                if 'acc' in part and i + 2 < len(parts):
                    try:
                        acc = float(parts[i + 2].strip())
                        # Get acc_norm from next line
                        next_line_idx = lines.index(line) + 1
                        if next_line_idx < len(lines):
                            next_line = lines[next_line_idx]
                            if 'acc_norm' in next_line:
                                next_parts = next_line.split('|')
                                for j, p in enumerate(next_parts):
                                    if 'acc_norm' in p and j + 2 < len(next_parts):
                                        acc_norm = float(next_parts[j + 2].strip())
                                        return acc, acc_norm
                        return acc, 0.0
                    except (ValueError, IndexError):
                        continue

    print(f"WARNING: Could not parse accuracy from output")
    return 0.0, 0.0


def optimize_abliteration(
    task: str,
    model: str,
    baseline_accuracy: float,
    output_dir: str = "./data/modified_models",
    eval_limit: int = 500
) -> EvaluationResult:
    """
    Optimize abliteration parameters through grid search.

    Args:
        task: Benchmark task name (e.g., "hellaswag")
        model: Base model name
        baseline_accuracy: Baseline model accuracy on task
        output_dir: Directory to save modified models
        eval_limit: Number of examples to evaluate

    Returns:
        Best performing configuration and results
    """
    # Define search grid - focusing on promising ranges
    # Based on previous experiments:
    # - max_weight=1.5: +2% (best so far)
    # - max_weight=3.5: +1.2%
    # - max_weight=10.0: -20% (too aggressive)

    # Let's search around the 1.5-2.5 range with different configurations
    search_grid = [
        # Conservative approaches (around proven best)
        AbliterationConfig(max_weight=1.5, min_weight=0.3, strength=1.0, num_pairs=100),
        AbliterationConfig(max_weight=1.5, min_weight=0.5, strength=1.0, num_pairs=200),
        AbliterationConfig(max_weight=1.8, min_weight=0.4, strength=1.0, num_pairs=200),
        AbliterationConfig(max_weight=2.0, min_weight=0.5, strength=1.0, num_pairs=200),

        # More focused kernel (narrower distribution)
        AbliterationConfig(max_weight=2.0, min_weight=0.8, strength=1.0, num_pairs=200),
        AbliterationConfig(max_weight=2.5, min_weight=1.0, strength=1.0, num_pairs=200),

        # Different component selection strategies (via strength)
        AbliterationConfig(max_weight=1.5, min_weight=0.3, strength=1.5, num_pairs=200),
        AbliterationConfig(max_weight=1.8, min_weight=0.4, strength=1.5, num_pairs=200),

        # More pairs for better vector estimation
        AbliterationConfig(max_weight=1.5, min_weight=0.3, strength=1.0, num_pairs=400),
        AbliterationConfig(max_weight=2.0, min_weight=0.5, strength=1.0, num_pairs=400),
    ]

    results = []
    best_result = None

    print(f"\n{'='*80}")
    print(f"ABLITERATION OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Task: {task}")
    print(f"Model: {model}")
    print(f"Baseline accuracy: {baseline_accuracy:.1%}")
    print(f"Configurations to test: {len(search_grid)}")
    print(f"{'='*80}\n")

    for i, config in enumerate(search_grid, 1):
        print(f"\n{'#'*80}")
        print(f"CONFIGURATION {i}/{len(search_grid)}")
        print(f"{'#'*80}")
        print(f"Parameters: {config}")
        print(f"{'#'*80}\n")

        # Modify weights
        model_path = run_modification(config, task, output_dir, model)

        # Evaluate
        accuracy, acc_norm = evaluate_model(model_path, task, eval_limit)

        # Store results
        result = EvaluationResult(
            config=config,
            accuracy=accuracy,
            acc_norm=acc_norm,
            model_path=model_path
        )
        results.append(result)

        # Track best
        if best_result is None or result.accuracy > best_result.accuracy:
            best_result = result

        # Report
        gain = result.gain(baseline_accuracy)
        print(f"\n{'='*80}")
        print(f"RESULTS FOR CONFIGURATION {i}")
        print(f"{'='*80}")
        print(f"Accuracy: {result.accuracy:.1%} (gain: {gain:+.1f}%)")
        print(f"Acc_norm: {result.acc_norm:.1%}")
        print(f"Current best: {best_result.accuracy:.1%} (gain: {best_result.gain(baseline_accuracy):+.1f}%)")
        print(f"{'='*80}\n")

    # Final report
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results (sorted by accuracy):")
    print(f"{'='*80}")

    for i, result in enumerate(sorted(results, key=lambda r: r.accuracy, reverse=True), 1):
        gain = result.gain(baseline_accuracy)
        print(f"{i}. Accuracy: {result.accuracy:.1%} (gain: {gain:+.1f}%)")
        print(f"   Config: {result.config}")
        print(f"   Model: {result.model_path}")
        print()

    print(f"{'='*80}")
    print(f"BEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"Accuracy: {best_result.accuracy:.1%}")
    print(f"Gain: {best_result.gain(baseline_accuracy):+.1f}%")
    print(f"Config: {best_result.config}")
    print(f"Model: {best_result.model_path}")
    print(f"{'='*80}\n")

    return best_result


if __name__ == "__main__":
    # Run optimization
    best = optimize_abliteration(
        task="hellaswag",
        model="meta-llama/Llama-3.2-1B-Instruct",
        baseline_accuracy=0.44,  # 44% baseline
        output_dir="./data/modified_models/optimized",
        eval_limit=500
    )

    print(f"\n✅ Optimization complete!")
    print(f"Best model: {best.model_path}")
    print(f"Best accuracy: {best.accuracy:.1%} ({best.gain(0.44):+.1f}% gain)")
