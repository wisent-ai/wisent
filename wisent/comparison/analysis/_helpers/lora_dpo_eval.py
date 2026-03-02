"""
LoRA DPO evaluation: evaluate trained adapter with optional steering.

Extracted from lora_dpo.py to keep files under 300 lines.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from wisent.core.utils.config_tools.constants import (
    COMPARISON_DEFAULT_BATCH_SIZE,
    COMPARISON_MAX_BATCH_SIZE,
    COMPARISON_NUM_PAIRS,
    COMPARISON_STEERING_LAYER,
    COMPARISON_STEERING_SCALES,
    DEFAULT_SPLIT_RATIO, JSON_INDENT,
)
from wisent.comparison.utils import (
    generate_contrastive_pairs,
    create_test_only_task,
    extract_accuracy,
    run_lm_eval_evaluation,
    run_ll_evaluation,
    apply_steering_to_model,
    remove_steering,
)

if TYPE_CHECKING:
    from wisent.core.primitives.models.wisent_model import WisentModel


def evaluate_lora_dpo(
    model_name: str,
    lora_path: str | Path,
    task: str,
    train_ratio: float = DEFAULT_SPLIT_RATIO,
    device: str = "cuda:0",
    batch_size: int = COMPARISON_DEFAULT_BATCH_SIZE,
    max_batch_size: int = COMPARISON_MAX_BATCH_SIZE,
    limit: int | None = None,
    output_dir: str | Path = None,
    num_train_pairs: int | None = None,
    num_epochs: int | None = None,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    lora_dropout: float | None = None,
    learning_rate: float | None = None,
    beta: float | None = None,
    max_length: int | None = None,
    max_prompt_length: int | None = None,
    with_steering: bool = False,
    steering_method: str = "caa",
    steering_layers: str = str(COMPARISON_STEERING_LAYER),
    steering_num_pairs: int = COMPARISON_NUM_PAIRS,
    steering_scales: list[float] | None = None,
    extraction_strategy: str = "mc_completion",
) -> dict:
    """Evaluate a trained DPO LoRA adapter."""
    from wisent.core.primitives.models.wisent_model import WisentModel
    from wisent.comparison.lora import apply_lora_to_model, remove_lora

    lora_path = Path(lora_path)
    if steering_scales is None:
        steering_scales = list(COMPARISON_STEERING_SCALES)

    task_dict = create_test_only_task(task, train_ratio=train_ratio)
    wisent_model = WisentModel(model_name=model_name, device=device)

    base_results = run_lm_eval_evaluation(wisent_model, task_dict, task, batch_size, max_batch_size, limit)
    base_acc_lm_eval = extract_accuracy(base_results, task)
    print(f"Base accuracy (lm-eval): {base_acc_lm_eval:.4f}")
    base_acc_ll = run_ll_evaluation(wisent_model, task_dict, task, limit)
    print(f"Base accuracy (LL): {base_acc_ll:.4f}")

    apply_lora_to_model(wisent_model, lora_path)
    lora_results = run_lm_eval_evaluation(wisent_model, task_dict, task, batch_size, max_batch_size, limit)
    lora_acc_lm_eval = extract_accuracy(lora_results, task)
    print(f"DPO-LoRA accuracy (lm-eval): {lora_acc_lm_eval:.4f}")
    lora_acc_ll = run_ll_evaluation(wisent_model, task_dict, task, limit)
    print(f"DPO-LoRA accuracy (LL): {lora_acc_ll:.4f}")

    results = {
        "task": task, "model": model_name, "training_method": "dpo",
        "lora_path": str(lora_path),
        "num_train_pairs": num_train_pairs, "num_epochs": num_epochs,
        "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
        "learning_rate": learning_rate, "beta": beta,
        "max_length": max_length, "max_prompt_length": max_prompt_length,
        "train_ratio": train_ratio, "eval_limit": limit,
        "base_accuracy_lm_eval": base_acc_lm_eval, "base_accuracy_ll": base_acc_ll,
        "lora_accuracy_lm_eval": lora_acc_lm_eval, "lora_accuracy_ll": lora_acc_ll,
        "lora_diff_lm_eval": lora_acc_lm_eval - base_acc_lm_eval,
        "lora_diff_ll": lora_acc_ll - base_acc_ll,
    }

    if with_steering:
        _run_steering_eval(
            wisent_model, task, task_dict, results,
            base_acc_lm_eval, lora_acc_lm_eval,
            steering_method, steering_layers, steering_num_pairs,
            steering_scales, extraction_strategy,
            batch_size, max_batch_size, limit, device,
        )

    remove_lora(wisent_model)
    del wisent_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _print_summary(results, base_acc_lm_eval, base_acc_ll, lora_acc_lm_eval, lora_acc_ll,
                   task, model_name, with_steering, steering_method)

    if output_dir:
        output_dir = Path(output_dir) / model_name.replace("/", "_")
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"{task}_lora_dpo_eval_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=JSON_INDENT)
        print(f"\nResults saved to: {results_file}")
    return results


def _run_steering_eval(
    wisent_model, task, task_dict, results,
    base_acc_lm_eval, lora_acc_lm_eval,
    steering_method, steering_layers, steering_num_pairs,
    steering_scales, extraction_strategy,
    batch_size, max_batch_size, limit, device,
):
    """Run optional steering evaluation on top of DPO-LoRA."""
    from wisent.core.weight_modification.trainers.steering_trainer import WisentSteeringTrainer
    from wisent.core.control.steering_methods import get_steering_method
    from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
    from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.primitives.contrastive_pairs.core.io.response import PositiveResponse, NegativeResponse
    import os

    pairs_data, pairs_file = generate_contrastive_pairs(task, steering_num_pairs)
    pairs = []
    for p in pairs_data:
        pair = ContrastivePair(
            prompt=p["prompt"],
            positive_response=PositiveResponse(model_response=p["positive_response"]["model_response"]),
            negative_response=NegativeResponse(model_response=p["negative_response"]["model_response"]),
        )
        pairs.append(pair)
    pair_set = ContrastivePairSet(pairs=pairs, name=f"{task}_dpo_lora_steering")

    steering_method_obj = get_steering_method(steering_method, device=device)
    strategy = ExtractionStrategy(extraction_strategy)
    trainer = WisentSteeringTrainer(
        model=wisent_model, pair_set=pair_set, steering_method=steering_method_obj,
    )
    result = trainer.run(layers_spec=steering_layers, strategy=strategy, accept_low_quality_vector=True)

    steering_vectors = {}
    for layer_name, tensor in result.steered_vectors.to_dict().items():
        if tensor is not None:
            steering_vectors[layer_name] = tensor.cpu().float().tolist()
    steering_data = {"steering_vectors": steering_vectors, "layers": list(steering_vectors.keys())}
    os.unlink(pairs_file)

    results["steering"] = {
        "method": steering_method, "layers": list(steering_vectors.keys()),
        "num_pairs": steering_num_pairs, "extraction_strategy": extraction_strategy,
        "scales": {},
    }
    for scale in steering_scales:
        apply_steering_to_model(wisent_model, steering_data, scale=scale)
        steer_results = run_lm_eval_evaluation(wisent_model, task_dict, task, batch_size, max_batch_size, limit)
        steer_acc_lm_eval = extract_accuracy(steer_results, task)
        steer_acc_ll = run_ll_evaluation(wisent_model, task_dict, task, limit)
        remove_steering(wisent_model)
        results["steering"]["scales"][str(scale)] = {
            "accuracy_lm_eval": steer_acc_lm_eval, "accuracy_ll": steer_acc_ll,
            "diff_from_base_lm_eval": steer_acc_lm_eval - base_acc_lm_eval,
            "diff_from_base_ll": steer_acc_ll - base_acc_lm_eval,
            "diff_from_lora_lm_eval": steer_acc_lm_eval - lora_acc_lm_eval,
            "diff_from_lora_ll": steer_acc_ll - lora_acc_lm_eval,
        }


def _print_summary(results, base_lm, base_ll, lora_lm, lora_ll, task, model, with_steering, method):
    """Print results summary table."""
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Task: {task}, Model: {model}, Training: DPO")
    print(f"{'-'*70}")
    print(f"{'Method':<25} {'lm-eval acc':<15} {'LL acc':<15} {'Diff (lm-eval)':<15}")
    print(f"{'-'*70}")
    print(f"{'Base':<25} {base_lm:<15.4f} {base_ll:<15.4f} {'':<15}")
    print(f"{'DPO-LoRA':<25} {lora_lm:<15.4f} {lora_ll:<15.4f} {lora_lm - base_lm:+.4f}")
    if with_steering:
        for scale, res in results["steering"]["scales"].items():
            label = f"DPO-LoRA+{method.upper()}@{scale}"
            print(f"{label:<25} {res['accuracy_lm_eval']:<15.4f} {res['accuracy_ll']:<15.4f} {res['diff_from_base_lm_eval']:+.4f}")
    print(f"{'='*70}")
