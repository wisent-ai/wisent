"""Comprehensive multi-method steering optimization entry point."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from wisent.core.utils.config_tools.constants import (
    SEPARATOR_WIDTH_REPORT, SEPARATOR_WIDTH_WIDE,
    JSON_INDENT, COMBO_OFFSET, RECURSION_INITIAL_DEPTH,
    SPLIT_RATIO_TRAIN_DEFAULT,
)
from wisent.core.control.steering_methods.configs.optimal import (
    get_optimal, get_optimal_extraction_strategy,
)

logger = logging.getLogger(__name__)


def execute_comprehensive_optimization(args) -> Dict[str, Any]:
    """Run comprehensive multi-method steering optimization.

    For each task, trains and evaluates all requested methods using the
    UnifiedOptimizer, then compares results across methods.
    """
    methods = [m.upper() for m in args.methods]
    tasks = args.tasks
    if not tasks:
        raise ValueError("--tasks is required (no default task list)")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = getattr(args, "device", None)
    verbose = getattr(args, "verbose", False)
    backend = getattr(args, "backend", "hyperopt")

    from transformers import AutoConfig as _AC
    _cfg = _AC.from_pretrained(args.model, trust_remote_code=True)
    num_layers = _cfg.num_hidden_layers

    if verbose:
        sep = "=" * SEPARATOR_WIDTH_WIDE
        print(f"\n{sep}")
        print("COMPREHENSIVE STEERING OPTIMIZATION")
        print(f"{sep}")
        print(f"   Model: {args.model}")
        print(f"   Tasks: {tasks}")
        print(f"   Methods: {methods}")
        print(f"   Backend: {backend}")
        print(f"{sep}\n")

    all_task_results = {}

    for task_name in tasks:
        if verbose:
            print(f"\n{'=' * SEPARATOR_WIDTH_REPORT}")
            print(f"TASK: {task_name}")
            print(f"{'=' * SEPARATOR_WIDTH_REPORT}")

        task_result = _optimize_task(
            model=args.model, task_name=task_name, methods=methods,
            num_layers=num_layers, device=device,
            verbose=verbose, backend=backend,
            args=args,
        )
        all_task_results[task_name] = task_result

        task_output = os.path.join(output_dir, f"{task_name}.json")
        with open(task_output, "w") as f:
            json.dump(task_result, f, indent=JSON_INDENT, default=str)

        if verbose:
            _print_task_summary(task_name, task_result)

    summary = _build_summary(args.model, methods, all_task_results)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=JSON_INDENT, default=str)

    if verbose:
        _print_final_summary(summary)

    if not getattr(args, "no_save", False):
        _save_best_configs(args.model, all_task_results)

    return summary


def _optimize_task(
    model: str, task_name: str, methods: List[str],
    num_layers: int, device: Optional[str],
    verbose: bool, backend: str,
    args: Any,
) -> Dict[str, Any]:
    """Optimize all methods for a single task."""
    from .runner import run_method_search

    pairs_file = _generate_task_pairs(task_name)

    baseline_score = None
    if getattr(args, "compute_baseline", False):
        from .evaluation import compute_baseline_score
        baseline_score = compute_baseline_score(
            model, task_name, pairs_file, device, verbose,
        )
        if verbose:
            print(f"   Baseline (unsteered): {baseline_score:.4f}")

    method_results = {}
    for method in methods:
        if verbose:
            print(f"\n   --- Method: {method} ---")
        _output_dir = getattr(args, "output_dir", None)
        method_output = os.path.join(_output_dir, task_name) if _output_dir else None
        result = run_method_search(
            model=model, task_name=task_name, method=method,
            pairs_file=pairs_file, num_layers=num_layers,
            device=device, verbose=verbose,
            backend=backend,
            search_overrides=_extract_search_overrides(args, method),
            early_rejection_config=_extract_early_rejection(args),
            output_dir=method_output,
        )
        method_results[method] = result

    best_method = max(
        method_results,
        key=lambda m: method_results[m]["best_score"],
    )

    return {
        "task": task_name, "baseline_score": baseline_score,
        "method_results": method_results, "best_method": best_method,
        "best_score": method_results[best_method]["best_score"],
        "best_params": method_results[best_method]["best_params"],
    }


def _generate_task_pairs(task_name: str) -> str:
    """Generate contrastive pairs and save to temp file."""
    from wisent.extractors.lm_eval.lm_task_pairs_generation import (
        build_contrastive_pairs,
    )
    pairs = build_contrastive_pairs(
        task_name=task_name, limit=None,
        train_ratio=SPLIT_RATIO_TRAIN_DEFAULT,
    )
    if not pairs:
        raise ValueError(f"No contrastive pairs generated for {task_name}")
    pairs_data = {"task": task_name, "pairs": [p.to_dict() for p in pairs]}
    fd, path = tempfile.mkstemp(
        suffix=".json", prefix=f"pairs_{task_name}_",
    )
    with os.fdopen(fd, "w") as f:
        json.dump(pairs_data, f, indent=JSON_INDENT)
    return path


def _extract_search_overrides(args, method: str) -> Dict[str, Any]:
    """Extract --search-* CLI overrides into a dict."""
    overrides = {}
    for attr in ("search_layers", "search_strengths", "search_strategies",
                 "search_token_aggregations", "search_prompt_constructions"):
        val = getattr(args, attr, None)
        if val is not None:
            overrides[attr] = val
    method_specific = {
        "TECZA": ("search_num_directions", "search_direction_weighting",
                  "search_retain_weight"),
        "TETNO": ("search_sensor_layer", "search_steering_layers",
                  "search_threshold", "search_gate_temp",
                  "search_max_alpha"),
        "GROM": ("search_sensor_layer", "search_steering_layers",
                 "search_max_alpha", "search_gate_hidden",
                 "search_intensity_hidden", "search_behavior_weight",
                 "search_sparse_weight"),
    }
    for attr in method_specific.get(method, ()):
        val = getattr(args, attr, None)
        if val is not None:
            overrides[attr] = val
    return overrides


def _extract_early_rejection(args) -> Dict[str, Any]:
    """Extract early rejection configuration."""
    return {
        "enabled": not getattr(args, "disable_early_rejection", False),
        "cv_threshold": args.early_rejection_cv_threshold,
    }


def _build_summary(
    model: str, methods: List[str], results: Dict,
) -> Dict:
    """Build cross-task summary."""
    per_method_wins = {
        m: RECURSION_INITIAL_DEPTH for m in methods
    }
    for task_result in results.values():
        winner = task_result.get("best_method")
        if winner in per_method_wins:
            per_method_wins[winner] += COMBO_OFFSET
    return {
        "model": model, "methods_tested": methods,
        "tasks": list(results.keys()), "per_task_results": results,
        "method_wins": per_method_wins,
        "overall_best_method": max(
            per_method_wins, key=per_method_wins.get),
    }


def _print_task_summary(task_name: str, result: Dict) -> None:
    """Print summary for one task."""
    print(f"\n   Task: {task_name}")
    if result.get("baseline_score") is not None:
        print(f"   Baseline: {result['baseline_score']:.4f}")
    for method, mr in result["method_results"].items():
        print(f"   {method:8s}: {mr['best_score']:.4f}")
    print(f"   Winner: {result['best_method']} ({result['best_score']:.4f})")


def _print_final_summary(summary: Dict) -> None:
    """Print final cross-task summary."""
    sep = "=" * SEPARATOR_WIDTH_WIDE
    print(f"\n{sep}")
    print("COMPREHENSIVE OPTIMIZATION COMPLETE")
    print(f"{sep}")
    for method, wins in summary["method_wins"].items():
        print(f"   {method:8s}: {wins} task wins")
    print(f"   Overall best: {summary['overall_best_method']}")


def _save_best_configs(model: str, results: Dict) -> None:
    """Save best configs to model config."""
    from wisent.core.utils.config_tools.config import ModelConfigManager
    config_manager = ModelConfigManager()
    config = config_manager.load_model_config(model) or {
        "model_name": model,
    }
    config.setdefault("comprehensive_optimization", {})
    for task_name, task_result in results.items():
        config["comprehensive_optimization"][task_name] = {
            "best_method": task_result["best_method"],
            "best_score": task_result["best_score"],
            "best_params": task_result["best_params"],
        }
    config_manager.save_model_config(model, **config)
