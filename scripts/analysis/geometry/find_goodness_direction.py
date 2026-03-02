#!/usr/bin/env python3
"""
Find a universal "goodness" direction using CLI commands.
Tests multiple models to see if goodness has consistent geometry across models.

Usage:
    python scripts/find_goodness_direction.py
"""
import json
import os
import sys
from pathlib import Path

from .find_goodness_direction_helpers import (
    generate_pairs,
    get_model_short_name,
    run_for_model,
)

# Configuration
DEVICE = os.environ.get('DEVICE', 'mps')
OUTPUT_BASE_DIR = Path(os.environ.get('OUTPUT_DIR', './goodness_direction_output'))
PAIRS_PER_BENCHMARK = int(os.environ.get('PAIRS_PER_BENCHMARK', '100'))

MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen3-8B",
    "openai/gpt-oss-20b",
]

BENCHMARKS = [
    'truthfulqa_gen', 'arc_easy', 'arc_challenge', 'hellaswag',
    'winogrande', 'piqa', 'boolq', 'openbookqa',
]

STRATEGIES = ['chat_last', 'chat_mean', 'mc_balanced']


def main():
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    pairs_dir = OUTPUT_BASE_DIR / 'pairs'
    pairs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FINDING UNIVERSAL GOODNESS DIRECTION")
    print("=" * 70)
    print(f"Models: {len(MODELS)}, Benchmarks: {len(BENCHMARKS)}")
    print(f"Strategies: {STRATEGIES}, Device: {DEVICE}")

    # Step 1: Generate pairs (shared across models)
    print("\nStep 1: Generating pairs for each benchmark...")
    benchmark_pairs = {}
    for bench in BENCHMARKS:
        pairs_path = pairs_dir / f"{bench}_pairs.json"
        if pairs_path.exists():
            print(f"  [{bench}] Pairs already exist")
            benchmark_pairs[bench] = str(pairs_path)
        else:
            success = generate_pairs(bench, str(pairs_path), PAIRS_PER_BENCHMARK)
            if success and pairs_path.exists():
                print(f"  [{bench}] Generated pairs")
                benchmark_pairs[bench] = str(pairs_path)
            else:
                print(f"  [{bench}] Failed")

    if not benchmark_pairs:
        print("\nNo pairs generated. Exiting.")
        sys.exit(1)

    print(f"\nGenerated pairs for {len(benchmark_pairs)}/{len(BENCHMARKS)} benchmarks")

    # Step 2: Run analysis for each model
    all_model_results = {}
    for model in MODELS:
        model_short = get_model_short_name(model)
        model_output_dir = OUTPUT_BASE_DIR / model_short
        try:
            model_results = run_for_model(model, pairs_dir, model_output_dir,
                                          BENCHMARKS, STRATEGIES, DEVICE, PAIRS_PER_BENCHMARK)
            all_model_results[model] = model_results
        except Exception as e:
            print(f"\nError with {model}: {e}")
            import traceback
            traceback.print_exc()

    # Step 3: Cross-model summary
    print("\n" + "=" * 70)
    print("CROSS-MODEL SUMMARY")
    print("=" * 70)

    print("\nPooled geometry per model:")
    for model, results in all_model_results.items():
        pooled = results.get('pooled_geometry', {})
        if pooled:
            verdict = pooled.get('verdict', 'unknown')
            linear = pooled.get('best_linear_score', 0)
            layer = pooled.get('best_layer', '?')
            all_results = pooled.get('all_results', [])
            best_struct = all_results[0].get('best_structure', 'unknown') if all_results else 'unknown'
            print(f"  {model}: verdict={verdict}, linear={linear:.3f}, layer={layer}, structure={best_struct}")

    print("\nPer-benchmark best config (chat_last):")
    print(f"  {'Benchmark':<20}", end="")
    for model in all_model_results:
        print(f" {get_model_short_name(model)[:15]:<15}", end="")
    print()

    for bench in BENCHMARKS:
        print(f"  {bench:<20}", end="")
        for model, results in all_model_results.items():
            chat_last = results.get('per_benchmark_results', {}).get('chat_last', {})
            bench_data = chat_last.get(bench, {})
            best_config = bench_data.get('best_config', '?')[:12]
            linear = bench_data.get('best_linear_score', 0)
            print(f" {best_config}({linear:.2f})", end="")
        print()

    # Save cross-model summary
    cross_model_summary = {
        'models': MODELS, 'benchmarks': BENCHMARKS, 'strategies': STRATEGIES,
        'all_results': {m: r for m, r in all_model_results.items()},
    }
    summary_path = OUTPUT_BASE_DIR / 'cross_model_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(cross_model_summary, f, indent=2)
    print(f"\nCross-model summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
