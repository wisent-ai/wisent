"""Find the best steering method for a given benchmark.

Env vars (all required): MODEL_NAME, BENCHMARK, OUTPUT_DIR, TRIALS_MULTIPLIER, BACKEND
"""
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from wisent.core.utils.config_tools.constants import (
    COMBO_OFFSET, EXIT_CODE_ERROR, JSON_INDENT,
    OPTIMIZATION_TRIAL_PAIRS_CAP, SCORE_RANGE_MIN,
    SEPARATOR_WIDTH_REPORT, SEPARATOR_WIDTH_WIDE, SPLIT_RATIO_TRAIN_DEFAULT,
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
from wisent.core.utils.services.optimization.core.atoms import (
    BaseOptimizer,
    HPOConfig,
)
from wisent.core.utils.cli.optimize_steering.pipeline.comprehensive import baseline_cache
from wisent.core.utils.services.benchmarks import validate_benchmark


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: {name} environment variable is required")
        sys.exit(EXIT_CODE_ERROR)
    return val


def main():
    model_name = _require_env("MODEL_NAME")
    benchmark = _require_env("BENCHMARK")
    validate_benchmark(benchmark)
    output_dir = _require_env("OUTPUT_DIR")
    trials_mult = int(_require_env("TRIALS_MULTIPLIER"))
    backend = _require_env("BACKEND")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from transformers import AutoConfig as _AC
    cfg = _AC.from_pretrained(model_name, trust_remote_code=True)
    num_layers = cfg.num_hidden_layers

    # Load the model ONCE and reuse across all trials/methods
    from wisent.core.primitives.models.wisent_model import WisentModel
    print(f"Loading model {model_name}...", flush=True)
    cached_model = WisentModel(model_name, device=None)
    print(f"   Model loaded", flush=True)

    all_methods = SteeringMethodRegistry.list_methods()

    train_file, test_file, n_train, n_test = _load_pairs_from_hf(
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

    # --- Baseline (unsteered) evaluation ---
    baseline_score = _run_baseline(
        model_name, benchmark, test_file, output_dir, cached_model,
    )
    print(f"   Baseline:      {baseline_score:.4f}")

    method_results = {}
    overall_start = time.time()

    for method_idx, method_name in enumerate(all_methods):
        _run_method(
            method_idx + COMBO_OFFSET, len(all_methods),
            method_name, model_name, benchmark, num_layers,
            trials_mult, backend, method_results, output_dir,
            train_file, test_file, n_train, cached_model,
            baseline_score, n_test,
        )

    _save_final_report(
        method_results, model_name, benchmark, output_dir,
        overall_start, baseline_score, train_file, test_file,
    )


def _load_pairs_from_hf(benchmark, output_dir):
    """Load pairs from HF. Train = cached activation pairs, test = extra pairs."""
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
        load_pair_texts_from_hf,
    )
    print(f"Loading cached pairs for {benchmark} from HF...", flush=True)
    n_train = OPTIMIZATION_TRIAL_PAIRS_CAP
    n_test = math.ceil(n_train / SPLIT_RATIO_TRAIN_DEFAULT) - n_train
    total_needed = n_train + n_test
    hf_pairs = load_pair_texts_from_hf(benchmark, limit=total_needed)
    if not hf_pairs:
        print(f"ERROR: No cached pairs on HF for {benchmark}")
        sys.exit(EXIT_CODE_ERROR)
    sorted_ids = sorted(hf_pairs.keys())

    def _to_pair(pid):
        p = hf_pairs[pid]
        d = {"prompt": p["prompt"],
             "positive_response": {"model_response": p["positive"]},
             "negative_response": {"model_response": p["negative"]}}
        if p.get("metadata"):
            d["metadata"] = p["metadata"]
        return d

    train_pairs = [_to_pair(pid) for pid in sorted_ids[:n_train]]
    test_pairs = [_to_pair(pid) for pid in sorted_ids[n_train:total_needed]]

    def _save(pair_list, path):
        with open(path, "w") as f:
            json.dump({"task_name": benchmark, "num_pairs": len(pair_list),
                        "pairs": pair_list}, f, indent=JSON_INDENT)

    train_path = os.path.join(output_dir, f"train_pairs_{benchmark}.json")
    test_path = os.path.join(output_dir, f"test_pairs_{benchmark}.json")
    _save(train_pairs, train_path)
    _save(test_pairs, test_path)
    print(f"   Train: {len(train_pairs)} (cached activations), Test: {len(test_pairs)}")
    return train_path, test_path, len(train_pairs), len(test_pairs)


def _run_method(
    method_idx, total, method_name, model_name, benchmark,
    num_layers, trials_mult, backend, method_results, output_dir,
    train_pairs_file, test_pairs_file, n_train, cached_model=None,
    baseline_score=None, n_test=None,
):
    """Run optimization for a single method."""
    method_upper = method_name.upper()
    space = get_method_space(method_upper, num_layers)
    n_trials = len(space) * trials_mult

    print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
    print(f"[{method_idx}/{total}] {method_upper} ({len(space)} dims, {n_trials} trials)")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}")

    method_start = time.time()
    method_dir = os.path.join(output_dir, "trials", method_name)
    workspace = os.path.join(method_dir, "_workspace")
    os.makedirs(workspace, exist_ok=True)

    objective = create_objective(
        method=method_upper, model=model_name, task=benchmark,
        num_layers=num_layers, limit=n_train, device=None,
        work_dir=workspace,
        train_pairs_file=train_pairs_file,
        test_pairs_file=test_pairs_file,
        cached_model=cached_model,
    )

    trial_counter = []

    def persisted_objective(params):
        score = objective(params)
        trial_idx = len(trial_counter)
        trial_counter.append(trial_idx)
        trial_dir = os.path.join(method_dir, f"trial_{trial_idx:04d}")
        os.makedirs(trial_dir, exist_ok=True)
        for fname in ("responses.json", "scores.json", "steering.pt"):
            src = os.path.join(workspace, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(trial_dir, fname))
        with open(os.path.join(trial_dir, "trial_meta.json"), "w") as f:
            json.dump({"params": params, "score": score, "trial": trial_idx},
                      f, indent=JSON_INDENT, default=str)
        return score

    optimizer = BaseOptimizer()
    optimizer.direction = "maximize"
    result = optimizer.optimize_fn(
        persisted_objective, space, n_trials, cfg=HPOConfig(backend=backend),
        model=model_name, benchmark=benchmark, method=method_upper,
    )
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

    delta = result.best_score - baseline_score if baseline_score is not None else None
    delta_str = f" delta={delta:+.4f}" if delta is not None else ""
    print(f"\n   {method_upper}: score={result.best_score:.4f}{delta_str} in {method_time:.1f}s")
    incremental = {
        "benchmark": benchmark, "model": model_name,
        "baseline_score": baseline_score, "n_train": n_train, "n_test": n_test,
        "methods": method_results,
    }
    incremental_path = os.path.join(
        output_dir, f"incremental_{benchmark}_{method_name}.json",
    )
    with open(incremental_path, "w") as f:
        json.dump(incremental, f, indent=JSON_INDENT, default=str)
    print(f"   Saved: {incremental_path}")


def _run_baseline(model_name, benchmark, test_file, output_dir, cached_model=None):
    """Evaluate unsteered model on test pairs, with per-pair HF cache."""
    acc, _, _ = baseline_cache.generate_baseline_with_cache(
        model_name, benchmark, test_file, None, cached_model=cached_model,
    )
    return acc


def _save_final_report(
    method_results, model_name, benchmark, output_dir, overall_start,
    baseline_score, train_file, test_file,
):
    """Determine winner, save final JSON with delta, diff, and activation effect."""
    from scripts.steering.find_best_method_diff import build_winner_diff
    from scripts.steering.find_best.activations import measure_activation_space_effect
    total_time = time.time() - overall_start
    scored = {n: r["best_score"] for n, r in method_results.items() if "best_score" in r}
    if not scored:
        print("ERROR: No method produced a score")
        sys.exit(EXIT_CODE_ERROR)
    winner = max(scored, key=scored.get)
    ranking = sorted(
        [{"method": n, "score": s, "delta": s - baseline_score} for n, s in scored.items()],
        key=lambda x: x["score"], reverse=True,
    )
    diff = build_winner_diff(output_dir, winner, model_name, benchmark)
    act_effect = measure_activation_space_effect(
        output_dir, winner, method_results[winner]["best_params"],
        model_name, benchmark, train_file, test_file,
    )
    final_results = {
        "model": model_name, "benchmark": benchmark,
        "baseline_score": baseline_score, "winner": winner,
        "winner_score": scored[winner], "winner_delta": scored[winner] - baseline_score,
        "winner_response_diff": diff, "activation_space_effect": act_effect,
        "total_time_seconds": total_time,
        "timestamp": datetime.now().isoformat(),
        "method_results": method_results, "ranking": ranking,
    }
    final_path = os.path.join(output_dir, f"best_method_{benchmark}.json")
    with open(final_path, "w") as f:
        json.dump(final_results, f, indent=JSON_INDENT, default=str)
    _print_final_report(ranking, winner, baseline_score, benchmark, diff, act_effect, total_time, final_path)


def _print_final_report(ranking, winner, baseline, benchmark, diff, act, total_time, path):
    """Print summary to stdout."""
    sep = "=" * SEPARATOR_WIDTH_WIDE
    print(f"\n{sep}\nRESULTS: {benchmark} (baseline: {baseline:.4f})\n{sep}")
    for r in ranking:
        sign = "+" if r["delta"] >= SCORE_RANGE_MIN else ""
        mk = " <-- WINNER" if r["method"] == winner else ""
        print(f"   {r['method'].rjust(SEPARATOR_WIDTH_REPORT)}: {r['score']:.4f} ({sign}{r['delta']:.4f}){mk}")
    if "error" not in diff:
        print(f"\n   Response diff ({winner}): +{diff['flipped_correct']} -{diff['flipped_wrong']} ={diff['unchanged']} net={diff['net_improvement']}")
    if "error" not in act:
        print(f"\n   Activation effect ({winner}): acc={act['classifier_accuracy']:.4f} auc={act['classifier_auc']:.4f}")
        print(f"     prob: base={act['base_mean_prob']:.4f} steered={act['steered_mean_prob']:.4f} shift={act['prob_shift']:.4f} region_shift={act['region_shift']}")
    print(f"\n   Time: {total_time:.1f}s  Results: {path}\n{sep}\n")


if __name__ == "__main__":
    main()
