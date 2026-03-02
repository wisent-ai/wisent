"""
Comparison of steering methods: Ours vs SAE-based.

This script:
1. Creates steering vectors using train split of pooled data
2. Runs base evaluation on test split (no overlap)
3. Runs steered evaluation on same test split
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from lm_eval import evaluator
from lm_eval.models.hf_steered import SteeredModel

from wisent.core.primitives.models.wisent_model import WisentModel
from wisent.core.utils.config_tools.constants import (
    COMPARISON_NUM_PAIRS,
    COMPARISON_MAX_BATCH_SIZE,
    COMPARISON_STEERING_LAYER,
    DEFAULT_SPLIT_RATIO,
)
from wisent.comparison import ours
from wisent.comparison import sae
from wisent.comparison import fgaa
from wisent.comparison.utils import (
    load_steering_vector,
    apply_steering_to_model,
    remove_steering,
    convert_to_lm_eval_format,
    create_test_only_task,
    extract_accuracy,
    run_lm_eval_evaluation,
    run_ll_evaluation,
)
from wisent.comparison._helpers.cli.main_helpers import (
    run_comparison,
    main as _main,
)

# Map method names to modules
METHOD_MODULES = {
    "caa": ours,
    "sae": sae,
    "fgaa": fgaa,
}


def run_single_task(
    model_name: str,
    task: str,
    bos_features_source: str,
    device: str,
    methods: list[str] = None,
    num_pairs: int = COMPARISON_NUM_PAIRS,
    steering_scales: list[float] = None,
    batch_size: int | str = 1,
    max_batch_size: int = COMPARISON_MAX_BATCH_SIZE,
    eval_limit: int | None = None,
    vectors_dir: Path = None,
    train_ratio: float = DEFAULT_SPLIT_RATIO,
    caa_layers: str = str(COMPARISON_STEERING_LAYER),
    sae_layers: str = str(COMPARISON_STEERING_LAYER),
    extraction_strategies: list[str] = None,
) -> list[dict]:
    """
    Run comparison for a single task with multiple methods, scales,
    and extraction strategies.

    Returns list of result dicts, one per method/scale/strategy combination.
    """
    if methods is None:
        methods = ["caa"]
    if steering_scales is None:
        raise ValueError("steering_scales must be provided explicitly")
    if extraction_strategies is None:
        extraction_strategies = ["mc_balanced"]

    results_list = []

    # Step 1: Create test task
    test_pct = round((1 - train_ratio) * 100)
    print(f"\n{'='*60}")
    print(f"Creating test task for: {task}")
    print(f"(using {test_pct}% of pooled data)")
    print(f"{'='*60}")

    task_dict = create_test_only_task(task, train_ratio=train_ratio)

    # Step 2: Generate ALL steering vectors FIRST
    steering_vectors_data = {}
    train_pct = round(train_ratio * 100)

    for method in methods:
        if method not in METHOD_MODULES:
            print(f"WARNING: Method '{method}' not implemented, skipping")
            continue
        method_module = METHOD_MODULES[method]

        for extraction_strategy in (extraction_strategies if method == "caa" else [None]):
            print(f"\n{'@'*60}")
            print(f"@ METHOD: {method}, EXTRACTION STRATEGY: {extraction_strategy or 'N/A'}")
            print(f"{'@'*60}")

            method_layers = caa_layers if method == "caa" else sae_layers
            print(f"\n{'='*60}")
            print(f"Generating steering vector for: {task} (method={method})")
            print(f"(using {train_pct}% of pooled data - no overlap with test)")
            print(f"Layers: {method_layers}")
            print(f"{'='*60}")

            suffix = f"_{extraction_strategy}" if extraction_strategy else ""
            vector_path = vectors_dir / f"{task}_{method}{suffix}_steering_vector.json"

            kwargs = {
                "task": task, "model_name": model_name,
                "output_path": vector_path, "num_pairs": num_pairs,
                "device": device, "layers": method_layers,
                "trait_label": task, "method": method,
            }
            if extraction_strategy:
                kwargs["extraction_strategy"] = extraction_strategy
            if method == "fgaa":
                kwargs["bos_features_source"] = bos_features_source

            method_module.generate_steering_vector(**kwargs)

            steering_data = load_steering_vector(vector_path, default_method=method)
            if extraction_strategy not in steering_vectors_data:
                steering_vectors_data[extraction_strategy] = {}
            steering_vectors_data[extraction_strategy][method] = steering_data

    # Step 3: Load model once for ALL evaluations
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")
    wisent_model = WisentModel(model_name=model_name, device=device)

    # Step 4: Run base evaluation
    base_results = run_lm_eval_evaluation(
        wisent_model=wisent_model, task_dict=task_dict, task_name=task,
        batch_size=batch_size, max_batch_size=max_batch_size, limit=eval_limit,
    )
    base_acc = extract_accuracy(base_results, task)

    base_ll_acc = run_ll_evaluation(
        wisent_model=wisent_model, task_dict=task_dict,
        task_name=task, limit=eval_limit,
    )

    # Step 5: Run ALL wisent steered evaluations
    wisent_results = _run_steered_evaluations(
        wisent_model, methods, extraction_strategies,
        steering_vectors_data, steering_scales, task_dict, task,
        batch_size, max_batch_size, eval_limit,
    )

    # Step 6: Free wisent_model to make room for SteeredModel
    del wisent_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Step 7: Run ALL lm-eval native steered evaluations
    _run_native_evaluations(
        results_list, methods, extraction_strategies,
        steering_vectors_data, steering_scales, model_name, task,
        device, batch_size, max_batch_size, eval_limit,
        vectors_dir, wisent_results, base_acc, base_ll_acc, num_pairs,
    )

    return results_list


def _run_steered_evaluations(
    wisent_model, methods, extraction_strategies,
    steering_vectors_data, steering_scales, task_dict, task,
    batch_size, max_batch_size, eval_limit,
):
    """Run all wisent steered evaluations (model stays loaded)."""
    wisent_results = {}
    for method in methods:
        for extraction_strategy in (extraction_strategies if method == "caa" else [None]):
            if extraction_strategy not in steering_vectors_data:
                continue
            if method not in steering_vectors_data[extraction_strategy]:
                continue
            steering_data = steering_vectors_data[extraction_strategy][method]
            for scale in steering_scales:
                apply_steering_to_model(wisent_model, steering_data, scale=scale)
                steered_results = run_lm_eval_evaluation(
                    wisent_model=wisent_model, task_dict=task_dict,
                    task_name=task, batch_size=batch_size,
                    max_batch_size=max_batch_size, limit=eval_limit,
                )
                steered_acc = extract_accuracy(steered_results, task)
                steered_ll_acc = run_ll_evaluation(
                    wisent_model=wisent_model, task_dict=task_dict,
                    task_name=task, limit=eval_limit,
                )
                remove_steering(wisent_model)
                wisent_results[(extraction_strategy, method, scale)] = {
                    "lm_eval": steered_acc, "ll": steered_ll_acc,
                }
    return wisent_results


def _run_native_evaluations(
    results_list, methods, extraction_strategies,
    steering_vectors_data, steering_scales, model_name, task,
    device, batch_size, max_batch_size, eval_limit,
    vectors_dir, wisent_results, base_acc, base_ll_acc, num_pairs,
):
    """Run all lm-eval native steered evaluations (one at a time)."""
    for method in methods:
        for extraction_strategy in (extraction_strategies if method == "caa" else [None]):
            if extraction_strategy not in steering_vectors_data:
                continue
            if method not in steering_vectors_data[extraction_strategy]:
                continue
            steering_data = steering_vectors_data[extraction_strategy][method]
            for scale in steering_scales:
                suffix = f"_{extraction_strategy}" if extraction_strategy else ""
                lm_eval_steer_path = vectors_dir / f"{task}_{method}{suffix}_lm_eval_steer_scale{scale}.pt"
                convert_to_lm_eval_format(steering_data, lm_eval_steer_path, scale=scale)
                lm_steered = SteeredModel(
                    pretrained=model_name, steer_path=str(lm_eval_steer_path),
                    device=device, batch_size=batch_size, max_batch_size=max_batch_size,
                )
                lm_eval_native_results = evaluator.evaluate(
                    lm=lm_steered, task_dict=task_dict, limit=eval_limit,
                )
                lm_eval_native_acc = extract_accuracy(lm_eval_native_results, task)
                del lm_steered
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                wisent_result = wisent_results[(extraction_strategy, method, scale)]
                results_list.append({
                    "task": task, "extraction_strategy": extraction_strategy or "N/A",
                    "method": method, "model": model_name,
                    "layers": steering_data['layers'], "num_pairs": num_pairs,
                    "steering_scale": scale, "base_accuracy_lm_eval": base_acc,
                    "base_accuracy_ll": base_ll_acc,
                    "steered_accuracy_lm_eval": wisent_result["lm_eval"],
                    "steered_accuracy_ll": wisent_result["ll"],
                    "steered_accuracy_lm_eval_native": lm_eval_native_acc,
                    "difference_lm_eval": wisent_result["lm_eval"] - base_acc,
                    "difference_ll": wisent_result["ll"] - base_ll_acc,
                    "difference_lm_eval_native": lm_eval_native_acc - base_acc,
                })


def main():
    _main(run_single_task_fn=run_single_task,
          run_comparison_fn=lambda **kw: run_comparison(
              **kw, run_single_task_fn=run_single_task))


if __name__ == "__main__":
    main()
