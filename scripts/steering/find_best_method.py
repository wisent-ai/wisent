"""Find the best steering method for a given benchmark.

Uses the UnifiedOptimizer with distribution-based search spaces
to optimize each steering method independently and rank them.

Generates ALL contrastive pairs once, splits into train/test,
trains steering on train set, evaluates on held-out test set.

Trials per method = dimensions * TRIALS_MULTIPLIER.

Environment variables (all required unless noted):
    MODEL_NAME: HuggingFace model ID
    BENCHMARK: Benchmark task name
    OUTPUT_DIR: Directory for results

Optional:
    TRIALS_MULTIPLIER: Trials per dimension (default: TRIALS_PER_DIMENSION_MULTIPLIER)
    BACKEND: Optimizer backend ('hyperopt' or 'optuna', default: hyperopt)
"""
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from wisent.core.utils.config_tools.constants import (
    COMBO_OFFSET,
    EXIT_CODE_ERROR,
    JSON_INDENT,
    SCORE_RANGE_MIN,
    SEPARATOR_WIDTH_REPORT,
    SEPARATOR_WIDTH_WIDE,
    SPLIT_RATIO_TRAIN_DEFAULT,
    TRIALS_PER_DIMENSION_MULTIPLIER,
)
from wisent.core.control.steering_methods.registry import (
    SteeringMethodRegistry,
)
from wisent.core.utils.cli.optimize_steering.search_space import (
    get_method_space,
)
from wisent.core.utils.cli.optimize_steering.pipeline import (
    create_objective,
)
from wisent.core.utils.services.optimization.core.unified_optimizer import (
    UnifiedOptimizer,
)


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: {name} environment variable is required")
        sys.exit(EXIT_CODE_ERROR)
    return val


def main():
    model_name = _require_env("MODEL_NAME")
    benchmark = _require_env("BENCHMARK")
    output_dir = _require_env("OUTPUT_DIR")
    trials_mult = int(os.environ.get(
        "TRIALS_MULTIPLIER", str(TRIALS_PER_DIMENSION_MULTIPLIER),
    ))
    backend = os.environ.get("BACKEND", "hyperopt")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from transformers import AutoConfig as _AC
    cfg = _AC.from_pretrained(model_name, trust_remote_code=True)
    num_layers = cfg.num_hidden_layers

    all_methods = SteeringMethodRegistry.list_methods()

    # Generate all pairs once, split into train/test
    train_file, test_file, n_train, n_test = _generate_and_split_pairs(
        benchmark, output_dir,
    )

    print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
    print("FIND BEST STEERING METHOD")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}")
    print(f"   Model:         {model_name}")
    print(f"   Layers:        {num_layers}")
    print(f"   Benchmark:     {benchmark}")
    print(f"   Train pairs:   {n_train}")
    print(f"   Test pairs:    {n_test}")
    print(f"   Methods:       {len(all_methods)}")
    print(f"   Trials/dim:    {trials_mult}x")
    print(f"   Backend:       {backend}")
    print(f"   Output:        {output_dir}")

    for method_name in all_methods:
        space = get_method_space(method_name.upper(), num_layers)
        dims = len(space)
        n_trials = dims * trials_mult
        print(f"   {method_name.upper()}: {dims} dims, {n_trials} trials")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}\n")

    method_results = {}
    overall_start = time.time()

    for method_idx, method_name in enumerate(all_methods):
        _run_method(
            method_idx + COMBO_OFFSET, len(all_methods),
            method_name, model_name, benchmark, num_layers,
            trials_mult, backend, method_results, output_dir,
            train_file, test_file,
        )

    _save_final_report(
        method_results, model_name, benchmark, output_dir, overall_start,
    )


def _generate_and_split_pairs(benchmark, output_dir):
    """Generate all pairs for the benchmark and split into train/test."""
    from wisent.extractors.lm_eval.lm_task_pairs_generation import (
        build_contrastive_pairs,
    )

    print(f"Generating all contrastive pairs for {benchmark}...", flush=True)
    pairs = build_contrastive_pairs(
        task_name=benchmark, train_ratio=SPLIT_RATIO_TRAIN_DEFAULT,
    )
    if not pairs:
        print(f"ERROR: No pairs generated for {benchmark}")
        sys.exit(EXIT_CODE_ERROR)

    split_idx = math.floor(len(pairs) * SPLIT_RATIO_TRAIN_DEFAULT)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]

    def _save_pairs(pair_list, path, task_name):
        data = {
            "task_name": task_name,
            "num_pairs": len(pair_list),
            "pairs": [p.to_dict() for p in pair_list],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=JSON_INDENT)

    train_path = os.path.join(output_dir, f"train_pairs_{benchmark}.json")
    test_path = os.path.join(output_dir, f"test_pairs_{benchmark}.json")
    _save_pairs(train_pairs, train_path, benchmark)
    _save_pairs(test_pairs, test_path, benchmark)
    print(f"   Total: {len(pairs)}, Train: {len(train_pairs)}, Test: {len(test_pairs)}")
    return train_path, test_path, len(train_pairs), len(test_pairs)


def _run_method(
    method_idx, total, method_name, model_name, benchmark,
    num_layers, trials_mult, backend, method_results, output_dir,
    train_pairs_file, test_pairs_file,
):
    """Run optimization for a single method."""
    method_upper = method_name.upper()
    space = get_method_space(method_upper, num_layers)
    n_trials = len(space) * trials_mult

    print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
    print(f"[{method_idx}/{total}] {method_upper} ({len(space)} dims, {n_trials} trials)")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}")

    method_start = time.time()
    optimizer = UnifiedOptimizer(backend=backend, direction="maximize")

    with tempfile.TemporaryDirectory() as work_dir:
        objective = create_objective(
            method=method_upper, model=model_name, task=benchmark,
            num_layers=num_layers, limit=SCORE_RANGE_MIN, device=None,
            work_dir=work_dir,
            train_pairs_file=train_pairs_file,
            test_pairs_file=test_pairs_file,
        )
        result = optimizer.optimize(objective, space, n_trials)

    method_time = time.time() - method_start
    entry = {
        "method": method_name,
        "best_score": result.best_score,
        "best_params": result.best_params,
        "n_trials": result.n_trials,
        "backend": result.backend,
        "time_seconds": method_time,
        "all_trials": result.all_trials,
    }
    method_results[method_name] = entry

    print(
        f"\n   {method_upper}: "
        f"score={result.best_score:.4f} "
        f"in {method_time:.1f}s"
    )

    incremental_path = os.path.join(
        output_dir, f"incremental_{benchmark}_{method_name}.json",
    )
    with open(incremental_path, "w") as f:
        json.dump(method_results, f, indent=JSON_INDENT, default=str)
    print(f"   Saved: {incremental_path}")


def _save_final_report(
    method_results, model_name, benchmark, output_dir, overall_start,
):
    """Determine winner, save final JSON, print summary."""
    total_time = time.time() - overall_start
    scored = {
        name: r["best_score"]
        for name, r in method_results.items()
        if "best_score" in r
    }

    if scored:
        winner = max(scored, key=scored.get)
        winner_score = scored[winner]
    else:
        winner, winner_score = None, SCORE_RANGE_MIN

    final_results = {
        "model": model_name,
        "benchmark": benchmark,
        "winner": winner,
        "winner_score": winner_score,
        "total_time_seconds": total_time,
        "timestamp": datetime.now().isoformat(),
        "method_results": method_results,
        "ranking": sorted(
            [
                {"method": n, "score": r.get("best_score", SCORE_RANGE_MIN)}
                for n, r in method_results.items()
            ],
            key=lambda x: x["score"],
            reverse=True,
        ),
    }

    final_path = os.path.join(
        output_dir, f"best_method_{benchmark}.json",
    )
    with open(final_path, "w") as f:
        json.dump(final_results, f, indent=JSON_INDENT, default=str)

    print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
    print(f"RESULTS: {benchmark}")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}")
    for rank_entry in final_results["ranking"]:
        marker = " <-- WINNER" if rank_entry["method"] == winner else ""
        name = rank_entry["method"].rjust(SEPARATOR_WIDTH_REPORT)
        print(f"   {name}: {rank_entry['score']:.4f}{marker}")
    print(f"\n   Total time: {total_time:.1f}s")
    print(f"   Results: {final_path}")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}\n")


if __name__ == "__main__":
    main()
