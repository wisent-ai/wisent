"""
Exhaustive layer combination analysis.

Tests all power-of-two minus one layer combinations to find optimal layer
subsets for geometry detection. Uses CLI commands for pair generation and
activation extraction.

A prior Qwen-8B run became unresponsive after starting the exhaustive step.
If investigating future failures: check dmesg BEFORE rebooting for OOM
messages, check kern.log, try SSH to run top/free before assuming crash.
Do NOT assume the root cause without actual evidence from logs/metrics.
"""

import os
import sys
import tempfile
import time
from typing import Dict

from wisent.core.utils.config_tools.constants import (
    PAIR_GENERATORS_DEFAULT_N, TEST_MAX_COMBO_SIZE, SEPARATOR_WIDTH_REPORT,
    COMBO_BASE, COMBO_OFFSET,
    PROGRESS_CALLBACK_THRESHOLD_EXHAUSTIVE,
    PROGRESS_CALLBACK_THRESHOLD_CONTIGUOUS,
)
from wisent.tests._exhaustive_helpers import (
    detect_model_layers,
    generate_pairs_cli,
    extract_activations_cli,
    load_activations_as_tensors,
    make_progress_callback,
    print_analysis_results,
    save_analysis_results,
)

TOKEN_AGGREGATIONS = [
    "final", "average", "first", "max", "min", "max_score"
]
PROMPT_STRATEGIES = [
    "chat_template", "direct_completion",
    "instruction_following", "multiple_choice", "role_playing"
]


def run_exhaustive_layer_analysis(
    task: str,
    model: str,
    token_aggregation: str,
    num_pairs: int = PAIR_GENERATORS_DEFAULT_N,
    max_layers: int | None = None,
    output_dir: str | None = None,
):
    """Run exhaustive layer combination analysis (all power-of-two minus one combos)."""
    from wisent.core.primitives.contrastive_pairs.diagnostics.control_vectors import (
        detect_geometry_exhaustive,
    )
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * SEPARATOR_WIDTH_REPORT)
    print("EXHAUSTIVE LAYER COMBINATION ANALYSIS")
    print("=" * SEPARATOR_WIDTH_REPORT)
    print(f"Task: {task}, Model: {model}, Pairs: {num_pairs}")
    start = time.time()
    model_layers = detect_model_layers(model)
    print(f"    Model has {model_layers} layers ({time.time()-start:.1f}s)")
    num_layers = min(max_layers, model_layers) if max_layers else model_layers
    combo_count = COMBO_BASE**num_layers - COMBO_OFFSET
    print(f"    Combinations to test: {combo_count:,}")
    with tempfile.TemporaryDirectory() as tmpdir:
        pairs_file = os.path.join(tmpdir, "pairs.json")
        acts_file = os.path.join(tmpdir, "activations.json")
        layers_str = ",".join(str(i) for i in range(COMBO_OFFSET, num_layers + COMBO_OFFSET))
        print(f"\n[step-one] Generating {num_pairs} pairs...")
        generate_pairs_cli(task, pairs_file, num_pairs)
        print(f"\n[step-two] Extracting activations for {num_layers} layers...")
        extract_activations_cli(
            pairs_file, acts_file, model, layers_str, token_aggregation,
        )
        print("\n[step-three] Loading and converting to tensors...")
        pos_t, neg_t, nl = load_activations_as_tensors(acts_file, max_layers)
        actual_combos = COMBO_BASE ** nl - COMBO_OFFSET
        print(f"    {nl} layers -> {actual_combos} combinations")
        print("\n[step-four] Running exhaustive analysis...")
        start = time.time()
        cb = make_progress_callback(start, threshold=PROGRESS_CALLBACK_THRESHOLD_EXHAUSTIVE)
        result = detect_geometry_exhaustive(
            pos_t, neg_t, max_layers=nl,
            combination_method="concat", progress_callback=cb,
        )
        elapsed = time.time() - start
        print(f"    Done in {elapsed:.1f}s ({actual_combos/elapsed:.1f} c/s)")
        print_analysis_results(result)
        save_analysis_results(
            result, output_dir, f"exhaustive_geometry_{task}",
            {"task": task, "model": model, "num_pairs": num_pairs,
             "max_layers": num_layers, "elapsed_seconds": elapsed},
        )
        return result


def run_limited_layer_analysis(
    task: str,
    model: str,
    token_aggregation: str,
    num_pairs: int = PAIR_GENERATORS_DEFAULT_N,
    max_combo_size: int = TEST_MAX_COMBO_SIZE,
    output_dir: str | None = None,
):
    """Run limited layer combination analysis (small-layer combos + all)."""
    from wisent.core.primitives.contrastive_pairs.diagnostics.control_vectors import (
        detect_geometry_limited,
    )
    from math import comb
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * SEPARATOR_WIDTH_REPORT)
    print("LIMITED LAYER COMBINATION ANALYSIS")
    print("=" * SEPARATOR_WIDTH_REPORT)
    start = time.time()
    model_layers = detect_model_layers(model)
    print(f"    Model has {model_layers} layers ({time.time()-start:.1f}s)")
    total = sum(
        comb(model_layers, r)
        for r in range(COMBO_OFFSET, min(max_combo_size, model_layers) + COMBO_OFFSET)
    )
    if max_combo_size < model_layers:
        total += COMBO_OFFSET
    print(f"    Will test {total:,} combinations")
    with tempfile.TemporaryDirectory() as tmpdir:
        pairs_file = os.path.join(tmpdir, "pairs.json")
        acts_file = os.path.join(tmpdir, "activations.json")
        layers_str = ",".join(str(i) for i in range(COMBO_OFFSET, model_layers + COMBO_OFFSET))
        generate_pairs_cli(task, pairs_file, num_pairs)
        extract_activations_cli(
            pairs_file, acts_file, model, layers_str, token_aggregation,
        )
        pos_t, neg_t, nl = load_activations_as_tensors(acts_file)
        start = time.time()
        cb = make_progress_callback(start)
        result = detect_geometry_limited(
            pos_t, neg_t, max_combo_size=max_combo_size,
            combination_method="concat", progress_callback=cb,
        )
        print_analysis_results(result)
        save_analysis_results(
            result, output_dir, f"geometry_limited_{task}",
            {"task": task, "model": model, "num_pairs": num_pairs,
             "max_combo_size": max_combo_size},
        )
        return result


def run_contiguous_layer_analysis(
    task: str,
    model: str,
    token_aggregation: str,
    num_pairs: int = PAIR_GENERATORS_DEFAULT_N,
    output_dir: str | None = None,
):
    """Run contiguous layer combination analysis (adjacent layers only)."""
    from wisent.core.primitives.contrastive_pairs.diagnostics.control_vectors import (
        detect_geometry_contiguous,
    )
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * SEPARATOR_WIDTH_REPORT)
    print("CONTIGUOUS LAYER COMBINATION ANALYSIS")
    print("=" * SEPARATOR_WIDTH_REPORT)
    start = time.time()
    model_layers = detect_model_layers(model)
    print(f"    Model has {model_layers} layers ({time.time()-start:.1f}s)")
    total = model_layers * (model_layers + COMBO_OFFSET) // COMBO_BASE
    print(f"    Will test {total:,} contiguous combinations")
    with tempfile.TemporaryDirectory() as tmpdir:
        pairs_file = os.path.join(tmpdir, "pairs.json")
        acts_file = os.path.join(tmpdir, "activations.json")
        layers_str = ",".join(str(i) for i in range(COMBO_OFFSET, model_layers + COMBO_OFFSET))
        generate_pairs_cli(task, pairs_file, num_pairs)
        extract_activations_cli(
            pairs_file, acts_file, model, layers_str, token_aggregation,
        )
        pos_t, neg_t, nl = load_activations_as_tensors(acts_file)
        start = time.time()
        cb = make_progress_callback(start, threshold=PROGRESS_CALLBACK_THRESHOLD_CONTIGUOUS)
        result = detect_geometry_contiguous(
            pos_t, neg_t,
            combination_method="concat", progress_callback=cb,
        )
        print_analysis_results(result)
        save_analysis_results(
            result, output_dir, f"geometry_contiguous_{task}",
            {"task": task, "model": model, "num_pairs": num_pairs,
             "mode": "contiguous"},
        )
        return result


def run_smart_layer_analysis(
    task: str,
    model: str,
    token_aggregation: str,
    prompt_strategy: str,
    num_pairs: int = PAIR_GENERATORS_DEFAULT_N,
    max_combo_size: int = TEST_MAX_COMBO_SIZE,
    output_dir: str | None = None,
):
    """Run smart analysis (contiguous + limited combos)."""
    from wisent.core.primitives.contrastive_pairs.diagnostics.control_vectors import (
        detect_geometry_smart,
    )
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * SEPARATOR_WIDTH_REPORT)
    print("SMART LAYER COMBINATION ANALYSIS")
    print("=" * SEPARATOR_WIDTH_REPORT)
    start = time.time()
    model_layers = detect_model_layers(model)
    print(f"    Model has {model_layers} layers ({time.time()-start:.1f}s)")
    with tempfile.TemporaryDirectory() as tmpdir:
        pairs_file = os.path.join(tmpdir, "pairs.json")
        acts_file = os.path.join(tmpdir, "activations.json")
        layers_str = ",".join(
            str(i) for i in range(COMBO_OFFSET, model_layers + COMBO_OFFSET)
        )
        generate_pairs_cli(task, pairs_file, num_pairs)
        extract_activations_cli(
            pairs_file, acts_file, model, layers_str,
            token_aggregation, prompt_strategy,
        )
        pos_t, neg_t, nl = load_activations_as_tensors(acts_file)
        start = time.time()
        cb = make_progress_callback(start)
        result = detect_geometry_smart(
            pos_t, neg_t, max_combo_size=max_combo_size,
            combination_method="concat", progress_callback=cb,
        )
        print_analysis_results(result)
        save_analysis_results(
            result, output_dir,
            f"geometry_smart_{task}_{token_aggregation}_{prompt_strategy}",
            {"task": task, "model": model, "num_pairs": num_pairs,
             "mode": "smart", "max_combo_size": max_combo_size,
             "token_aggregation": token_aggregation,
             "prompt_strategy": prompt_strategy},
        )
        return result


if __name__ == "__main__":
    from wisent.tests._exhaustive_cli import main
    main()
