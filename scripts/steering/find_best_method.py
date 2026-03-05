"""Find the best steering method for a given benchmark.

Loads per-method JSON configs to determine extraction requirements
and required parameters. Methods that need Q/K projection extraction
(SZLAK, PRZELOM) are skipped since the current activation pipeline
only supports residual stream extraction.

Environment variables:
    MODEL_NAME: HuggingFace model ID (required)
    BENCHMARK: Benchmark task name (required)
    OUTPUT_DIR: Directory for results (required)
    TRAIN_RATIO: Train/test split ratio as float (required)

Usage via run_on_gcp.sh:
    MODEL_NAME=... BENCHMARK=... OUTPUT_DIR=... \
    ./run_on_gcp.sh --model $MODEL_NAME \
        "python scripts/steering/find_best_method.py"
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from wisent.core.utils.config_tools.constants import (
    SEPARATOR_WIDTH_WIDE,
    JSON_INDENT,
    SCORE_RANGE_MIN,
    SEPARATOR_WIDTH_REPORT,
    EXIT_CODE_ERROR,
)
from wisent.core.control.steering_methods.configs.loader import (
    load_method_config,
    ALL_METHOD_NAMES,
)

_RESIDUAL_COMPONENT = "residual_stream"


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: {name} environment variable is required")
        sys.exit(EXIT_CODE_ERROR)
    return val


def _can_run_method(config: dict) -> bool:
    """Check if method can run with current activation pipeline."""
    component = config["extraction"]["component"]
    return component == _RESIDUAL_COMPONENT


def main():
    model_name = _require_env("MODEL_NAME")
    benchmark = _require_env("BENCHMARK")
    output_dir = _require_env("OUTPUT_DIR")
    train_ratio = float(_require_env("TRAIN_RATIO"))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
    print("FIND BEST STEERING METHOD")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}")
    print(f"   Model:     {model_name}")
    print(f"   Benchmark: {benchmark}")
    print(f"   Methods:   {len(ALL_METHOD_NAMES)}")
    print(f"   Output:    {output_dir}")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}\n")

    # Load all method configs up front — fail early if any missing
    method_configs = {}
    for name in ALL_METHOD_NAMES:
        method_configs[name] = load_method_config(name)

    # Determine which methods can run
    runnable = []
    skipped = []
    for name in ALL_METHOD_NAMES:
        if _can_run_method(method_configs[name]):
            runnable.append(name)
        else:
            skipped.append(name)
            component = method_configs[name]["extraction"]["component"]
            print(f"   SKIP {name}: needs {component} extraction")

    print(f"   Runnable: {len(runnable)}, Skipped: {len(skipped)}\n")

    from wisent.core.primitives.models.wisent_model import WisentModel
    from wisent.extractors.lm_eval.lm_task_pairs_generation import (
        build_contrastive_pairs,
    )
    from wisent.core.utils.cli.commands.optimization.core.engine.method_optimizer import (
        MethodOptimizer,
    )
    from wisent.core.reading.evaluators.rotator import EvaluatorRotator

    print("Loading model...", flush=True)
    wisent_model = WisentModel(model_name)
    print(f"Model loaded: {wisent_model.num_layers} layers\n")

    # build_contrastive_pairs handles train/test split via train_ratio
    print(f"Generating pairs for {benchmark}...", flush=True)
    pair_result = build_contrastive_pairs(
        task_name=benchmark,
        train_ratio=train_ratio,
    )
    if not pair_result:
        print(f"ERROR: No pairs generated for {benchmark}")
        sys.exit(EXIT_CODE_ERROR)

    train_pairs = pair_result.train
    test_pairs = pair_result.test
    print(f"Train: {len(train_pairs.pairs)}, Test: {len(test_pairs.pairs)}\n")

    evaluator = EvaluatorRotator(task_name=benchmark)
    print(f"Evaluator: {evaluator.__class__.__name__}\n")

    method_results = {}
    overall_start = time.time()

    for method_idx, method_name in enumerate(runnable, start=EXIT_CODE_ERROR):
        _run_method(
            method_idx, len(runnable), method_name,
            method_configs[method_name], wisent_model,
            train_pairs, test_pairs, evaluator, benchmark,
            method_results, output_dir,
        )

    # Record skipped methods
    for name in skipped:
        component = method_configs[name]["extraction"]["component"]
        method_results[name] = {
            "method": name,
            "skipped": True,
            "reason": f"Needs {component} extraction (not yet supported)",
        }

    _save_final_report(
        method_results, model_name, benchmark,
        output_dir, overall_start,
    )


def _run_method(
    method_idx, total, method_name, config, wisent_model,
    train_pairs, test_pairs, evaluator, benchmark,
    method_results, output_dir,
):
    """Run optimization for a single method. Fails loudly."""
    from wisent.core.utils.cli.commands.optimization.core.engine.method_optimizer import (
        MethodOptimizer,
    )

    print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
    print(f"[{method_idx}/{total}] {method_name.upper()}")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}")
    print(f"   Extraction: {config['extraction']['component']}")
    print(f"   Layer mode: {config['extraction']['layer_mode']}")
    print(f"   Application: {config['application']['mode']}\n")

    method_start = time.time()

    optimizer = MethodOptimizer(
        model=wisent_model,
        method_name=method_name,
        verbose=True,
    )

    # This will raise ValueError if required params are missing
    # — that's intentional. No silent failures.
    summary = optimizer.optimize(
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        evaluator=evaluator,
        task_name=benchmark,
    )

    method_time = time.time() - method_start
    best = summary.best_result
    entry = {
        "method": method_name,
        "best_score": best.score if best else SCORE_RANGE_MIN,
        "baseline_score": summary.baseline_score,
        "improvement": (
            best.score - summary.baseline_score
            if best else SCORE_RANGE_MIN
        ),
        "configs_tested": summary.configs_tested,
        "time_seconds": method_time,
        "best_config": _extract_config(best),
        "extraction": config["extraction"],
        "application": config["application"],
    }
    method_results[method_name] = entry

    print(
        f"\n   {method_name}: "
        f"score={entry['best_score']:.4f} "
        f"(baseline={entry['baseline_score']:.4f}, "
        f"delta={entry['improvement']:+.4f}) "
        f"in {method_time:.1f}s"
    )

    incremental_path = os.path.join(
        output_dir,
        f"incremental_{benchmark}_{method_name}.json",
    )
    with open(incremental_path, "w") as f:
        json.dump(method_results, f, indent=JSON_INDENT, default=str)
    print(f"   Saved: {incremental_path}")


def _extract_config(best):
    """Extract serializable config from best result."""
    if not best:
        return {}
    return {
        "layers": best.config.layers,
        "strength": best.config.strength,
        "token_aggregation": best.config.token_aggregation.value,
        "strategy": best.config.strategy,
        "method_params": best.config.method_params,
    }


def _save_final_report(
    method_results, model_name, benchmark,
    output_dir, overall_start,
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
        winner = None
        winner_score = SCORE_RANGE_MIN

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
