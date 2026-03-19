"""Find the best steering method for a given benchmark.

Uses the BaseOptimizer with distribution-based search spaces
to optimize each steering method independently and rank them.
Generates ALL contrastive pairs once, splits into train/test,
trains steering on train set, evaluates on held-out test set.
Trials per method = dimensions * trials_multiplier.
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
    COMBO_OFFSET, EXIT_CODE_ERROR, JSON_INDENT, OPTIMIZATION_TRIAL_PAIRS_CAP,
    SCORE_RANGE_MIN, SEPARATOR_WIDTH_REPORT, SEPARATOR_WIDTH_WIDE,
    SPLIT_RATIO_TRAIN_DEFAULT,
)
from wisent.core.control.steering_methods.registry import (
    SteeringMethodRegistry,
)
from wisent.core.utils.cli.optimize_steering.search_space import (
    get_method_space,
)
from wisent.core.utils.cli.optimize_steering.pipeline import create_objective
from wisent.core.utils.services.optimization.core.atoms import (
    BaseOptimizer, HPOConfig,
)
from wisent.core.utils.cli.optimize_steering.pipeline.comprehensive import (
    baseline_cache,
)


def execute_find_best_method(args):
    """Execute the find-best-method command."""
    model_name, benchmark = args.model, args.task
    output_dir, trials_mult, backend = (
        args.output_dir, args.trials_multiplier, args.backend,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    from transformers import AutoConfig as _AC
    cfg = _AC.from_pretrained(model_name, trust_remote_code=True)
    num_layers = cfg.num_hidden_layers
    all_methods = SteeringMethodRegistry.list_methods()
    train_file, test_file, n_train, n_test = _generate_and_split_pairs(
        benchmark, output_dir,
    )
    sep = "=" * SEPARATOR_WIDTH_WIDE
    print(f"\n{sep}\nFIND BEST STEERING METHOD\n{sep}")
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
        print(f"   {method_name.upper()}: {len(space)} dims, "
              f"{len(space) * trials_mult} trials")
    print(f"{sep}\n")
    baseline_score = _run_baseline(
        model_name, benchmark, test_file, output_dir,
    )
    print(f"   Baseline:      {baseline_score:.4f}")
    method_results = {}
    overall_start = time.time()
    for method_idx, method_name in enumerate(all_methods):
        _run_method(
            method_idx + COMBO_OFFSET, len(all_methods),
            method_name, model_name, benchmark, num_layers,
            trials_mult, backend, method_results, output_dir,
            train_file, test_file, n_train,
        )
    _save_final_report(
        method_results, model_name, benchmark, output_dir,
        overall_start, baseline_score, train_file, test_file,
    )


def _generate_and_split_pairs(benchmark, output_dir):
    """Generate all pairs for the benchmark and split train/test."""
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
    train_pairs, test_pairs = pairs[:split_idx], pairs[split_idx:]

    def _save(pair_list, path, task_name):
        data = {"task_name": task_name, "num_pairs": len(pair_list),
                "pairs": [p.to_dict() for p in pair_list]}
        with open(path, "w") as f:
            json.dump(data, f, indent=JSON_INDENT)

    train_path = os.path.join(output_dir, f"train_pairs_{benchmark}.json")
    test_path = os.path.join(output_dir, f"test_pairs_{benchmark}.json")
    _save(train_pairs, train_path, benchmark)
    _save(test_pairs, test_path, benchmark)
    print(f"   Total: {len(pairs)}, "
          f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")
    return train_path, test_path, len(train_pairs), len(test_pairs)


def _run_method(
    method_idx, total, method_name, model_name, benchmark,
    num_layers, trials_mult, backend, method_results, output_dir,
    train_pairs_file, test_pairs_file, n_train,
):
    """Run optimization for a single method."""
    method_upper = method_name.upper()
    space = get_method_space(method_upper, num_layers)
    n_trials = len(space) * trials_mult
    sep = "=" * SEPARATOR_WIDTH_WIDE
    print(f"\n{sep}\n[{method_idx}/{total}] {method_upper} "
          f"({len(space)} dims, {n_trials} trials)\n{sep}")
    method_start = time.time()
    method_dir = os.path.join(output_dir, "trials", method_name)
    workspace = os.path.join(method_dir, "_workspace")
    os.makedirs(workspace, exist_ok=True)
    trial_limit = min(n_train, OPTIMIZATION_TRIAL_PAIRS_CAP)
    objective = create_objective(
        method=method_upper, model=model_name, task=benchmark,
        num_layers=num_layers, limit=trial_limit, device=None,
        work_dir=workspace, train_pairs_file=train_pairs_file,
        test_pairs_file=test_pairs_file,
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
    )
    method_time = time.time() - method_start
    method_results[method_name] = {
        "method": method_name, "best_score": result.best_score,
        "best_params": result.best_params, "n_trials": result.n_trials,
        "backend": result.backend, "time_seconds": method_time,
        "all_trials": result.all_trials,
    }
    print(f"\n   {method_upper}: score={result.best_score:.4f} "
          f"in {method_time:.1f}s")
    incremental_path = os.path.join(
        output_dir, f"incremental_{benchmark}_{method_name}.json",
    )
    with open(incremental_path, "w") as f:
        json.dump(method_results, f, indent=JSON_INDENT, default=str)
    print(f"   Saved: {incremental_path}")


def _run_baseline(model_name, benchmark, test_file, output_dir):
    """Evaluate unsteered model, using HF cache if available."""
    if baseline_cache.check_baseline_exists(model_name, benchmark):
        print("   Loading cached baseline from HuggingFace...")
        _, scores, meta = baseline_cache.load_baseline_from_hf(
            model_name, benchmark,
        )
        return meta.get(
            "accuracy", sum(s["correct"] for s in scores) / len(scores),
        )
    print("   Generating baseline (no cache found)...")
    acc, _, _ = baseline_cache.generate_and_upload_baseline(
        model_name, benchmark, test_file, None,
    )
    return acc


def _save_final_report(
    method_results, model_name, benchmark, output_dir, overall_start,
    baseline_score, train_file, test_file,
):
    """Determine winner, save final JSON with delta, diff, and act."""
    from .diff import build_winner_diff
    from .activations import measure_activation_space_effect
    total_time = time.time() - overall_start
    scored = {n: r["best_score"]
              for n, r in method_results.items() if "best_score" in r}
    if not scored:
        print("ERROR: No method produced a score")
        sys.exit(EXIT_CODE_ERROR)
    winner = max(scored, key=scored.get)
    ranking = sorted(
        [{"method": n, "score": s, "delta": s - baseline_score}
         for n, s in scored.items()],
        key=lambda x: x["score"], reverse=True,
    )
    diff = build_winner_diff(output_dir, winner, model_name, benchmark)
    act_effect = measure_activation_space_effect(
        output_dir, winner, method_results[winner]["best_params"],
        model_name, benchmark, train_file, test_file,
    )
    final = {
        "model": model_name, "benchmark": benchmark,
        "baseline_score": baseline_score, "winner": winner,
        "winner_score": scored[winner],
        "winner_delta": scored[winner] - baseline_score,
        "winner_response_diff": diff, "activation_space_effect": act_effect,
        "total_time_seconds": total_time,
        "timestamp": datetime.now().isoformat(),
        "method_results": method_results, "ranking": ranking,
    }
    final_path = os.path.join(output_dir, f"best_method_{benchmark}.json")
    with open(final_path, "w") as f:
        json.dump(final, f, indent=JSON_INDENT, default=str)
    _print_final(ranking, winner, baseline_score, benchmark,
                 diff, act_effect, total_time, final_path)


def _print_final(ranking, winner, baseline, benchmark,
                 diff, act, total_time, path):
    """Print summary to stdout."""
    sep = "=" * SEPARATOR_WIDTH_WIDE
    print(f"\n{sep}\nRESULTS: {benchmark} (baseline: {baseline:.4f})\n{sep}")
    for r in ranking:
        sign = "+" if r["delta"] >= SCORE_RANGE_MIN else ""
        mk = " <-- WINNER" if r["method"] == winner else ""
        print(f"   {r['method'].rjust(SEPARATOR_WIDTH_REPORT)}: "
              f"{r['score']:.4f} ({sign}{r['delta']:.4f}){mk}")
    if "error" not in diff:
        print(f"\n   Response diff ({winner}): "
              f"+{diff['flipped_correct']} -{diff['flipped_wrong']} "
              f"={diff['unchanged']} net={diff['net_improvement']}")
    if "error" not in act:
        print(f"\n   Activation effect ({winner}): "
              f"acc={act['classifier_accuracy']:.4f} "
              f"auc={act['classifier_auc']:.4f}")
        print(f"     prob: base={act['base_mean_prob']:.4f} "
              f"steered={act['steered_mean_prob']:.4f} "
              f"shift={act['prob_shift']:.4f} "
              f"region_shift={act['region_shift']}")
    print(f"\n   Time: {total_time:.1f}s  Results: {path}\n{sep}\n")
