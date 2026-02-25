"""Helpers for maximum_abliteration."""

import subprocess
from dataclasses import dataclass

from wisent.core.constants import (
    EXTRACTION_DEFAULT_PAIR_LIMIT,
    EXTRACTION_SINGLE_PAIR_LIMIT,
    ABLITERATION_NUM_PAIRS,
    ABLITERATION_DEFAULT_POSITION,
    ABLITERATION_DEFAULT_DISTANCE,
    ABLITERATION_BINARY_SEARCH_LOW,
    ABLITERATION_BINARY_SEARCH_HIGH,
    ABLITERATION_BINARY_SEARCH_ITERS,
    SUBPROCESS_TIMEOUT_LONG,
    DISPLAY_TRUNCATION_LARGE,
    SEPARATOR_WIDTH_REPORT,
)


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


def evaluate_model(model_path: str, task: str = "hellaswag", limit: int = EXTRACTION_DEFAULT_PAIR_LIMIT) -> tuple[float, float]:
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

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_LONG)

    if result.returncode != 0:
        print(f"Evaluation failed: {result.stderr[:DISPLAY_TRUNCATION_LARGE]}")
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

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_LONG)

    if result.returncode != 0:
        print(f"Abliteration failed: {result.stderr[:DISPLAY_TRUNCATION_LARGE]}")
        return ""

    return output_dir


def binary_search_strength(
    task: str,
    model: str,
    base_dir: str,
    baseline_acc: float,
    low: float = ABLITERATION_BINARY_SEARCH_LOW,
    high: float = ABLITERATION_BINARY_SEARCH_HIGH,
    iterations: int = ABLITERATION_BINARY_SEARCH_ITERS,
    num_pairs: int = ABLITERATION_NUM_PAIRS,
) -> tuple[float, float, float]:
    """
    Binary search to find optimal strength.

    Returns: (best_strength, best_acc, best_acc_norm)
    """
    print("\n" + "=" * SEPARATOR_WIDTH_REPORT)
    print("BINARY SEARCH FOR OPTIMAL STRENGTH")
    print("=" * SEPARATOR_WIDTH_REPORT)

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
            max_weight_position=ABLITERATION_DEFAULT_POSITION,
            min_weight_distance=ABLITERATION_DEFAULT_DISTANCE,
            components=["self_attn.o_proj", "mlp.down_proj"],
        )

        output_dir = f"{base_dir}/strength_search_{i}_{mid:.3f}"
        model_path = run_abliteration(config, task, model, output_dir)

        if model_path:
            acc, acc_norm = evaluate_model(model_path, task, limit=EXTRACTION_SINGLE_PAIR_LIMIT)
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
    num_pairs: int = ABLITERATION_NUM_PAIRS,
) -> tuple[list[str], float]:
    """
    Test different component combinations.

    Returns: (best_components, best_acc)
    """
    print("\n" + "=" * SEPARATOR_WIDTH_REPORT)
    print("COMPONENT SEARCH")
    print("=" * SEPARATOR_WIDTH_REPORT)

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
            max_weight_position=ABLITERATION_DEFAULT_POSITION,
            min_weight_distance=ABLITERATION_DEFAULT_DISTANCE,
            components=components,
        )

        output_dir = f"{base_dir}/components_{i}"
        model_path = run_abliteration(config, task, model, output_dir)

        if model_path:
            acc, acc_norm = evaluate_model(model_path, task, limit=EXTRACTION_SINGLE_PAIR_LIMIT)
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
    num_pairs: int = ABLITERATION_NUM_PAIRS,
    best_components: list[str] = None,
) -> tuple[float, float, float, float]:
    """
    Search for optimal kernel shape (max_weight_position, min_weight_distance).

    Returns: (best_position, best_distance, best_acc, best_acc_norm)
    """
    print("\n" + "=" * SEPARATOR_WIDTH_REPORT)
    print("KERNEL SHAPE SEARCH")
    print("=" * SEPARATOR_WIDTH_REPORT)

    if best_components is None:
        best_components = ["self_attn.o_proj", "mlp.down_proj"]

    # For Llama-3.2-1B with 16 layers, try different positions
    positions = [6.0, 7.0, 8.0, 9.0, 10.0]  # Middle-ish layers
    distances = [4.0, 5.0, 6.0, 7.0, 8.0]   # How wide the kernel is

    best_position = ABLITERATION_DEFAULT_POSITION
    best_distance = ABLITERATION_DEFAULT_DISTANCE
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
                acc, acc_norm = evaluate_model(model_path, task, limit=EXTRACTION_SINGLE_PAIR_LIMIT)

                if acc > best_acc:
                    best_position = pos
                    best_distance = dist
                    best_acc = acc
                    best_acc_norm = acc_norm
                    print(f"  NEW BEST: pos={pos}, dist={dist} -> acc={acc:.4f}")

    print(f"\nBest kernel: position={best_position}, distance={best_distance} (acc={best_acc:.4f})")
    return best_position, best_distance, best_acc, best_acc_norm

