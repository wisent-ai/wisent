"""Shared helpers for exhaustive geometry analysis tests."""

import json
import os
import subprocess
import sys
import time
import torch
from datetime import datetime
from typing import Dict, List

from wisent.core.utils.config_tools.constants import DISPLAY_TRUNCATION_COMPACT, PROGRESS_CALLBACK_THRESHOLD, SEPARATOR_WIDTH_REPORT, JSON_INDENT


def detect_model_layers(model: str) -> int:
    """Auto-detect model layer count from config."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    return (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "n_layer", None)
        or getattr(config, "num_layers", None)
        or 32
    )


def generate_pairs_cli(task: str, pairs_file: str, num_pairs: int):
    """Generate contrastive pairs using CLI."""
    result = subprocess.run(
        [
            sys.executable, "-m", "wisent.core.primitives.model_interface.core.main",
            "generate-pairs-from-task",
            task,
            "--output", pairs_file,
            "--limit", str(num_pairs),
        ],
        capture_output=True,
        text=True,
        timeout=600
    )
    if result.returncode != 0:
        raise RuntimeError(f"Pair generation failed: {result.stderr}")
    return result


def extract_activations_cli(
    pairs_file: str,
    activations_file: str,
    model: str,
    layers_str: str,
    token_aggregation: str,
    prompt_strategy: str | None = None,
):
    """Extract activations using CLI."""
    cmd = [
        sys.executable, "-m", "wisent.core.primitives.model_interface.core.main",
        "get-activations",
        pairs_file,
        "--output", activations_file,
        "--model", model,
        "--layers", layers_str,
        "--token-aggregation", token_aggregation,
    ]
    if prompt_strategy:
        cmd.extend(["--prompt-strategy", prompt_strategy])
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Activation extraction failed: {result.stderr}"
        )
    return result


def load_activations_as_tensors(
    activations_file: str,
    max_layers: int | None = None,
) -> tuple[
    Dict[int, torch.Tensor], Dict[int, torch.Tensor], int
]:
    """Load activations JSON and convert to tensors by layer.

    Returns (pos_tensors, neg_tensors, num_layers).
    """
    with open(activations_file, "r") as f:
        data = json.load(f)

    pairs_list = data.get("pairs", [])
    pos_by_layer: Dict[int, List[torch.Tensor]] = {}
    neg_by_layer: Dict[int, List[torch.Tensor]] = {}

    for pair in pairs_list:
        pos_la = pair.get("positive_response", {}).get(
            "layers_activations", {}
        )
        neg_la = pair.get("negative_response", {}).get(
            "layers_activations", {}
        )
        for layer_key in pos_la:
            layer = int(layer_key)
            if max_layers is not None and layer > max_layers:
                continue
            if layer not in pos_by_layer:
                pos_by_layer[layer] = []
                neg_by_layer[layer] = []
            if layer_key in pos_la and layer_key in neg_la:
                pos_by_layer[layer].append(
                    torch.tensor(pos_la[layer_key]).reshape(-1)
                )
                neg_by_layer[layer].append(
                    torch.tensor(neg_la[layer_key]).reshape(-1)
                )

    pos_tensors: Dict[int, torch.Tensor] = {}
    neg_tensors: Dict[int, torch.Tensor] = {}
    for layer in sorted(pos_by_layer.keys()):
        if pos_by_layer[layer] and neg_by_layer[layer]:
            pos_tensors[layer] = torch.stack(pos_by_layer[layer])
            neg_tensors[layer] = torch.stack(neg_by_layer[layer])

    return pos_tensors, neg_tensors, len(pos_tensors)


def make_progress_callback(start_time: float, threshold: int = PROGRESS_CALLBACK_THRESHOLD):
    """Create a progress callback for combination analysis."""
    last_report = [0, time.time()]

    def callback(current: int, total: int):
        now = time.time()
        if (
            current - last_report[0] >= threshold
            or now - last_report[1] >= 30
            or current == total
        ):
            elapsed = now - start_time
            rate = current / elapsed if elapsed > 0 else 0
            remaining = (
                (total - current) / rate if rate > 0 else 0
            )
            pct = 100 * current / total
            print(
                f"    Progress: {current:,}/{total:,}"
                f" ({pct:.1f}%) - {rate:.1f} combos/sec"
                f" - ETA: {remaining:.0f}s"
            )
            last_report[0] = current
            last_report[1] = now

    return callback


def print_analysis_results(result):
    """Print standard analysis results."""
    print("\n" + "=" * SEPARATOR_WIDTH_REPORT)
    print("RESULTS")
    print("=" * SEPARATOR_WIDTH_REPORT)
    print(f"\nTotal combinations tested: {result.total_combinations}")
    print(f"\nBest combination: {result.best_combination}")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Best structure: {result.best_structure.value}")
    print(f"\nBest single layer: L{result.single_layer_best}")
    print(
        f"Best single layer score: "
        f"{result.single_layer_best_score:.4f}"
    )
    print(
        f"Combination beats single: "
        f"{result.combination_beats_single}"
    )
    print(
        f"Improvement over single: "
        f"{result.improvement_over_single:.4f}"
    )


def save_analysis_results(
    result,
    output_dir: str,
    filename_prefix: str,
    extra_fields: dict | None = None,
) -> str:
    """Save analysis results to JSON and return file path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        output_dir, f"{filename_prefix}_{timestamp}.json"
    )
    results_json = {
        "total_combinations": result.total_combinations,
        "best_combination": list(result.best_combination),
        "best_score": result.best_score,
        "best_structure": result.best_structure.value,
        "single_layer_best": result.single_layer_best,
        "single_layer_best_score": result.single_layer_best_score,
        "combination_beats_single": result.combination_beats_single,
        "improvement_over_single": result.improvement_over_single,
        "top_10": [
            {
                "layers": list(r.layers),
                "best_score": r.best_score,
                "best_structure": r.best_structure.value,
                "all_scores": r.all_scores,
            }
            for r in result.top_10
        ],
        "top_100": [
            {
                "layers": list(r.layers),
                "best_score": r.best_score,
                "best_structure": r.best_structure.value,
            }
            for r in result.all_results[:DISPLAY_TRUNCATION_COMPACT]
        ],
        "patterns": {
            k: v if not isinstance(v, float) or not (v != v)
            else None
            for k, v in result.patterns.items()
        },
        "recommendation": result.recommendation,
    }
    if extra_fields:
        results_json.update(extra_fields)
    with open(output_file, "w") as f:
        json.dump(results_json, f, indent=JSON_INDENT)
    print(f"\nResults saved to: {output_file}")
    return output_file
