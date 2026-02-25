"""
Exhaustive layer combination analysis.

Tests all 2^N - 1 layer combinations to find optimal layer subsets
for geometry detection.

Uses CLI commands for pair generation and activation extraction.

===============================================================================
DEBUGGING NOTES - READ BEFORE MAKING ASSUMPTIONS
===============================================================================

On Dec 15, 2025, a Qwen3-8B run (36 layers = 68 billion combinations) became
unresponsive after starting step [5]. The instance lost SSM connection, SSH
timed out, and required a reboot.

WHAT WE KNOW (facts with evidence):
- Step [5] started: "Running exhaustive analysis (68719476735 combinations)..."
- No further output after that line
- Instance became unreachable (SSM ConnectionLost, SSH timeout)
- After reboot, dmesg.0 showed NO OOM messages
- kern.log had no errors between 18:30 (step 5 start) and 19:58 (reboot)

WHAT WE DO NOT KNOW (no evidence):
- Whether the process was running or stuck
- Whether memory was exhausted (no OOM in logs)
- Whether CPU was pegged
- The actual cause of unresponsiveness

DO NOT ASSUME:
- That 68 billion combinations is "too many" without measuring
- That the list allocation caused OOM (no evidence)
- That the loop is slow (no benchmarks)
- ANY root cause without actual evidence from logs/metrics

If investigating future failures:
1. Check dmesg BEFORE rebooting for OOM messages
2. Check /var/log/kern.log for errors
3. Try to SSH and run 'top', 'free -h', 'ps aux' before assuming crash
4. Get actual memory/CPU metrics, don't guess

The instance may have been working fine but just not producing output.
===============================================================================
"""

import os
import sys
import tempfile
import time
from typing import Dict

from wisent.core.constants import PARSER_DEFAULT_NUM_PAIRS, TEST_MAX_COMBO_SIZE, SEPARATOR_WIDTH_REPORT
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
    task: str = "truthfulqa_gen",
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    num_pairs: int = PARSER_DEFAULT_NUM_PAIRS,
    max_layers: int | None = None,
    output_dir: str = "/home/ubuntu/output",
):
    """
    Run exhaustive layer combination analysis.

    Tests all 2^N - 1 layer combinations.

    WARNING: DO NOT SET max_layers TO REDUCE THE NUMBER OF LAYERS.
    The whole point is to test ALL layer combinations.
    max_layers exists ONLY for debugging/testing purposes.
    """
    from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
        detect_geometry_exhaustive,
    )
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * SEPARATOR_WIDTH_REPORT)
    print("EXHAUSTIVE LAYER COMBINATION ANALYSIS")
    print("=" * SEPARATOR_WIDTH_REPORT)
    print(f"Task: {task}, Model: {model}, Pairs: {num_pairs}")

    print(f"\n[0] Detecting model layer count from config...")
    start = time.time()
    model_layers = detect_model_layers(model)
    print(f"    Model has {model_layers} layers ({time.time()-start:.1f}s)")

    num_layers = min(max_layers, model_layers) if max_layers else model_layers
    print(f"    Combinations to test: {2**num_layers - 1:,}")

    with tempfile.TemporaryDirectory() as tmpdir:
        pairs_file = os.path.join(tmpdir, "pairs.json")
        acts_file = os.path.join(tmpdir, "activations.json")
        layers_str = ",".join(str(i) for i in range(1, num_layers + 1))

        print(f"\n[1] Generating {num_pairs} pairs...")
        generate_pairs_cli(task, pairs_file, num_pairs)
        print(f"\n[2] Extracting activations layers 1-{num_layers}...")
        extract_activations_cli(pairs_file, acts_file, model, layers_str)
        print("\n[3-4] Loading and converting to tensors...")
        pos_t, neg_t, nl = load_activations_as_tensors(acts_file, max_layers)
        actual_combos = 2 ** nl - 1
        print(f"    {nl} layers -> {actual_combos} combinations")

        print(f"\n[5] Running exhaustive analysis...")
        start = time.time()
        cb = make_progress_callback(start, threshold=10000)
        result = detect_geometry_exhaustive(
            pos_t, neg_t, max_layers=nl,
            combination_method="concat", progress_callback=cb,
        )
        elapsed = time.time() - start
        print(f"    Done in {elapsed:.1f}s ({actual_combos/elapsed:.1f} c/s)")
        print_analysis_results(result)
        save_analysis_results(
            result, output_dir,
            f"exhaustive_geometry_{task}",
            {"task": task, "model": model, "num_pairs": num_pairs,
             "max_layers": num_layers, "elapsed_seconds": elapsed},
        )
        return result


def run_limited_layer_analysis(
    task: str = "truthfulqa_gen",
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    num_pairs: int = PARSER_DEFAULT_NUM_PAIRS,
    max_combo_size: int = TEST_MAX_COMBO_SIZE,
    output_dir: str = "/home/ubuntu/output",
):
    """Run limited layer combination analysis (1,2,3-layer combos + all)."""
    from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
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
        for r in range(1, min(max_combo_size, model_layers) + 1)
    )
    if max_combo_size < model_layers:
        total += 1
    print(f"    Will test {total:,} combinations")

    with tempfile.TemporaryDirectory() as tmpdir:
        pairs_file = os.path.join(tmpdir, "pairs.json")
        acts_file = os.path.join(tmpdir, "activations.json")
        layers_str = ",".join(str(i) for i in range(1, model_layers + 1))

        generate_pairs_cli(task, pairs_file, num_pairs)
        extract_activations_cli(pairs_file, acts_file, model, layers_str)
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
    task: str = "truthfulqa_gen",
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    num_pairs: int = PARSER_DEFAULT_NUM_PAIRS,
    output_dir: str = "/home/ubuntu/output",
):
    """Run contiguous layer combination analysis (adjacent layers only)."""
    from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
        detect_geometry_contiguous,
    )
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * SEPARATOR_WIDTH_REPORT)
    print("CONTIGUOUS LAYER COMBINATION ANALYSIS")
    print("=" * SEPARATOR_WIDTH_REPORT)

    start = time.time()
    model_layers = detect_model_layers(model)
    print(f"    Model has {model_layers} layers ({time.time()-start:.1f}s)")
    total = model_layers * (model_layers + 1) // 2
    print(f"    Will test {total:,} contiguous combinations")

    with tempfile.TemporaryDirectory() as tmpdir:
        pairs_file = os.path.join(tmpdir, "pairs.json")
        acts_file = os.path.join(tmpdir, "activations.json")
        layers_str = ",".join(str(i) for i in range(1, model_layers + 1))

        generate_pairs_cli(task, pairs_file, num_pairs)
        extract_activations_cli(pairs_file, acts_file, model, layers_str)
        pos_t, neg_t, nl = load_activations_as_tensors(acts_file)

        start = time.time()
        cb = make_progress_callback(start, threshold=50)
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
    task: str = "truthfulqa_gen",
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    num_pairs: int = PARSER_DEFAULT_NUM_PAIRS,
    max_combo_size: int = TEST_MAX_COMBO_SIZE,
    token_aggregation: str = "final",
    prompt_strategy: str = "chat_template",
    output_dir: str = "/home/ubuntu/output",
):
    """Run smart analysis (contiguous + limited combos)."""
    from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
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
            str(i) for i in range(1, model_layers + 1)
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
            combination_method="concat",
            progress_callback=cb,
        )
        print_analysis_results(result)
        save_analysis_results(
            result, output_dir,
            f"geometry_smart_{task}_{token_aggregation}"
            f"_{prompt_strategy}",
            {
                "task": task, "model": model,
                "num_pairs": num_pairs,
                "mode": "smart",
                "max_combo_size": max_combo_size,
                "token_aggregation": token_aggregation,
                "prompt_strategy": prompt_strategy,
            },
        )
        return result


if __name__ == "__main__":
    from wisent.tests._exhaustive_cli import main
    main()
