"""Pre-populate all Benchmark Debug tab data for a single benchmark.

Runs extraction/evaluation test, generates visualization cache for all
available layers, and loads baseline + find-best-method results from HF.

Usage:
    python -m wisent.support.examples.scripts.visualization.cache.debug_single_benchmark \
        truthfulqa_custom --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import sys
import time

from wisent.core.utils.config_tools.constants import TRIALS_PER_DIMENSION_MULTIPLIER

_SEP = "=" * 60


def run(task_name: str, model_name: str, output_dir: str,
        trials_multiplier: int, backend: str):
    """Fill in all debug tab data for a benchmark."""
    overall_start = time.time()

    print(f"\n{_SEP}")
    print(f"BENCHMARK DEBUG: {task_name} / {model_name}")
    print(_SEP)

    # --- 1. Extraction + Evaluation ---
    print(f"\n[1/4] Testing extraction + evaluation...")
    from wisent.support.examples.scripts.discovery.validation.test_single_benchmark import (
        test_benchmark,
    )
    result = test_benchmark(task_name)
    ext = result.get("extraction", {})
    evl = result.get("evaluator", {})
    print(f"  Extraction: {ext.get('status', '?')} ({ext.get('pair_count', 0)} pairs)")
    print(f"  Evaluator:  {evl.get('status', '?')}")
    evaluation = evl.get("evaluation", {})
    if evaluation:
        metrics = evaluation.get("aggregated_metrics", {})
        for k, v in metrics.items():
            val = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"    {k}: {val}")

    # --- 2. Visualizations ---
    print(f"\n[2/4] Generating visualization cache...")
    from wisent.support.examples.scripts.visualization.cache.generate_benchmark_viz import (
        generate_viz,
    )
    try:
        viz_results = generate_viz(task_name, model_name)
        for layer_num, status in sorted(viz_results.items()):
            print(f"  Layer {layer_num}: {status}")
    except Exception as exc:
        print(f"  No activations available: {exc}")
        viz_results = {}

    # --- 3. Find Best Method ---
    print(f"\n[3/4] Running find-best-method optimization...")
    from wisent.core.utils.cli.commands.optimize_steering.pipeline.find_best.handler import (
        execute_find_best_method,
    )
    from types import SimpleNamespace
    find_best_args = SimpleNamespace(
        model=model_name, task=task_name, output_dir=output_dir,
        trials_multiplier=trials_multiplier, backend=backend,
    )
    execute_find_best_method(find_best_args)

    # --- 4. Baseline + Find-Best Results ---
    print(f"\n[4/4] Loading baseline + optimization results from HF...")
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
        load_baseline_metadata_from_hf,
        load_best_method_from_hf,
    )

    baseline = load_baseline_metadata_from_hf(model_name, task_name)
    if baseline:
        print(f"  Baseline: {baseline.get('accuracy', 0):.2%} "
              f"({baseline.get('total_pairs', 0)} pairs, "
              f"{str(baseline.get('timestamp', ''))[:10]})")
    else:
        print(f"  Baseline: not available")

    best = load_best_method_from_hf(model_name, task_name)
    if best:
        winner = best.get("winner", "?")
        score = best.get("winner_score", 0)
        delta = best.get("winner_delta", 0)
        print(f"  Best method: {winner.upper()} — {score:.4f} ({delta:+.4f})")
        for r in best.get("ranking", []):
            m = r.get("method", "?").upper()
            s = r.get("score", 0)
            d = r.get("delta", 0)
            marker = " <-- WINNER" if m == winner.upper() else ""
            print(f"    {m:>15}: {s:.4f} ({d:+.4f}){marker}")
    else:
        print(f"  Best method: no optimization results yet")

    # --- Summary ---
    total_time = time.time() - overall_start
    print(f"\n{_SEP}")
    print(f"SUMMARY: {task_name} / {model_name} ({total_time:.1f}s)")
    print(f"  Extraction:     {ext.get('status', '?')}")
    print(f"  Evaluator:      {evl.get('status', '?')}")
    print(f"  Viz layers:     {len(viz_results)}")
    print(f"  Baseline:       {'yes' if baseline else 'no'}")
    print(f"  Optimization:   {'yes' if best else 'no'}")
    print(_SEP)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-populate all debug tab data for a benchmark.",
    )
    parser.add_argument("task", help="Benchmark task name")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--output-dir", default="debug_results",
        help="Directory for find-best-method results (default: debug_results)",
    )
    parser.add_argument(
        "--trials-multiplier", type=int,
        default=TRIALS_PER_DIMENSION_MULTIPLIER,
        help="Trials per dimension for find-best-method",
    )
    parser.add_argument(
        "--backend", default="optuna", choices=["hyperopt", "optuna"],
        help="Optimizer backend (default: optuna)",
    )
    args = parser.parse_args()
    run(args.task, args.model, args.output_dir,
        args.trials_multiplier, args.backend)
    sys.exit(0)


if __name__ == "__main__":
    main()
