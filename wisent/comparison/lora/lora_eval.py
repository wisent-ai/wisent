"""LoRA evaluation: apply_lora_to_model, remove_lora, evaluate_lora."""
from __future__ import annotations
import gc
import json
from pathlib import Path
from typing import TYPE_CHECKING
import torch
from wisent.core.utils.config_tools.constants import JSON_INDENT
from wisent.comparison.utils import (
    create_test_only_task, extract_accuracy, run_lm_eval_evaluation,
    run_ll_evaluation, generate_contrastive_pairs, apply_steering_to_model, remove_steering,
)
if TYPE_CHECKING:
    from wisent.core.primitives.models.wisent_model import WisentModel

__all__ = ["apply_lora_to_model", "remove_lora", "evaluate_lora"]

def apply_lora_to_model(wisent_model: "WisentModel", lora_path: str | Path) -> None:
    """Apply a trained LoRA adapter to a WisentModel."""
    from peft import PeftModel
    lora_path = Path(lora_path)
    if hasattr(wisent_model.hf_model, 'peft_config'):
        wisent_model.hf_model.load_adapter(str(lora_path), adapter_name="steering")
        wisent_model.hf_model.set_adapter("steering")
    else:
        wisent_model.hf_model = PeftModel.from_pretrained(
            wisent_model.hf_model, str(lora_path), adapter_name="steering")
    print(f"LoRA adapter loaded from {lora_path}")

def remove_lora(wisent_model: "WisentModel") -> None:
    """Remove/disable LoRA adapter from a WisentModel."""
    if hasattr(wisent_model.hf_model, 'disable_adapters'):
        try:
            wisent_model.hf_model.disable_adapters()
            print("LoRA adapter disabled")
        except ValueError:
            pass
    elif hasattr(wisent_model.hf_model, 'base_model'):
        wisent_model.hf_model = wisent_model.hf_model.base_model.model
        print("LoRA adapter removed")

def _eval_lora_with_steering(wisent_model, task, task_dict, base_acc_lm_eval, base_acc_ll,
                              lora_acc_lm_eval, lora_acc_ll, steering_method, steering_layers,
                              steering_num_pairs, steering_scales, extraction_strategy,
                              device, batch_size, max_batch_size, results, log_interval: int,
                              min_norm_threshold: float):
    """Run LoRA + steering evaluation at multiple scales."""
    from wisent.core.weight_modification.trainers.steering_trainer import WisentSteeringTrainer
    from wisent.core.control.steering_methods import get_steering_method
    from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
    from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.primitives.contrastive_pairs.core.io.response import PositiveResponse, NegativeResponse
    pairs_data, pairs_file = generate_contrastive_pairs(task, steering_num_pairs)
    pairs = [ContrastivePair(prompt=p["prompt"],
                             positive_response=PositiveResponse(model_response=p["positive_response"]["model_response"]),
                             negative_response=NegativeResponse(model_response=p["negative_response"]["model_response"]))
             for p in pairs_data]
    pair_set = ContrastivePairSet(pairs=pairs, name=f"{task}_lora_steering")
    steering_method_obj = get_steering_method(steering_method, task_name=task, device=device)
    strategy = ExtractionStrategy(extraction_strategy)
    trainer = WisentSteeringTrainer(model=wisent_model, pair_set=pair_set, steering_method=steering_method_obj)
    result = trainer.run(layers_spec=steering_layers, min_norm_threshold=min_norm_threshold, strategy=strategy, accept_low_quality_vector=True)
    steering_vectors = {k: v.cpu().float().tolist() for k, v in result.steered_vectors.to_dict().items() if v is not None}
    steering_data = {"steering_vectors": steering_vectors, "layers": list(steering_vectors.keys())}
    import os
    os.unlink(pairs_file)
    results["steering"] = {"method": steering_method, "layers": list(steering_vectors.keys()),
                           "num_pairs": steering_num_pairs, "extraction_strategy": extraction_strategy, "scales": {}}
    for scale in steering_scales:
        apply_steering_to_model(wisent_model, steering_data, scale=scale, min_norm_threshold=min_norm_threshold)
        steer_results = run_lm_eval_evaluation(wisent_model, task_dict, task, batch_size, max_batch_size, None)
        steer_acc_lm_eval = extract_accuracy(steer_results, task)
        steer_acc_ll = run_ll_evaluation(wisent_model, task_dict, task, log_interval=log_interval, limit=None)
        remove_steering(wisent_model)
        results["steering"]["scales"][str(scale)] = {
            "accuracy_lm_eval": steer_acc_lm_eval, "accuracy_ll": steer_acc_ll,
            "diff_from_base_lm_eval": steer_acc_lm_eval - base_acc_lm_eval,
            "diff_from_base_ll": steer_acc_ll - base_acc_ll,
            "diff_from_lora_lm_eval": steer_acc_lm_eval - lora_acc_lm_eval,
            "diff_from_lora_ll": steer_acc_ll - lora_acc_ll,
        }
    return results

def evaluate_lora(
    model_name: str, lora_path: str | Path, task: str,
    extraction_strategy: str, device: str,
    batch_size: int, max_batch_size: int,
    log_interval: int,
    min_norm_threshold: float,
    *,
    train_ratio: float,
    output_dir: str | Path = None,
    num_train_pairs: int | None = None, num_epochs: int | None = None,
    lora_r: int | None = None, lora_alpha: int | None = None,
    lora_dropout: float | None = None, learning_rate: float | None = None,
    with_steering: bool = False, steering_method: str | None = None,
    steering_layers: str,
    steering_num_pairs: int,
    steering_scales: tuple[float, ...],
) -> dict:
    """Evaluate a trained LoRA adapter comparing base vs LoRA performance."""
    from wisent.core.primitives.models.wisent_model import WisentModel
    lora_path = Path(lora_path)
    print(f"\n{'='*60}\nCreating test task for: {task}\n{'='*60}")
    task_dict = create_test_only_task(task, train_ratio=train_ratio)
    print(f"\n{'='*60}\nLoading model: {model_name}\n{'='*60}")
    wisent_model = WisentModel(model_name=model_name, device=device)
    print(f"\n{'='*60}\nRunning BASE evaluation (no LoRA)\n{'='*60}")
    base_results = run_lm_eval_evaluation(wisent_model, task_dict, task, batch_size, max_batch_size, None)
    base_acc_lm_eval = extract_accuracy(base_results, task)
    print(f"Base accuracy (lm-eval): {base_acc_lm_eval:.4f}")
    base_acc_ll = run_ll_evaluation(wisent_model, task_dict, task, log_interval=log_interval, limit=None)
    print(f"Base accuracy (LL): {base_acc_ll:.4f}")
    print(f"\n{'='*60}\nApplying LoRA adapter from: {lora_path}\n{'='*60}")
    apply_lora_to_model(wisent_model, lora_path)
    print(f"\n{'='*60}\nRunning LORA evaluation\n{'='*60}")
    lora_results = run_lm_eval_evaluation(wisent_model, task_dict, task, batch_size, max_batch_size, None)
    lora_acc_lm_eval = extract_accuracy(lora_results, task)
    print(f"LoRA accuracy (lm-eval): {lora_acc_lm_eval:.4f}")
    lora_acc_ll = run_ll_evaluation(wisent_model, task_dict, task, log_interval=log_interval, limit=None)
    print(f"LoRA accuracy (LL): {lora_acc_ll:.4f}")
    results = {
        "task": task, "model": model_name, "lora_path": str(lora_path),
        "num_train_pairs": num_train_pairs, "num_epochs": num_epochs,
        "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
        "learning_rate": learning_rate, "train_ratio": train_ratio,
        "base_accuracy_lm_eval": base_acc_lm_eval, "base_accuracy_ll": base_acc_ll,
        "lora_accuracy_lm_eval": lora_acc_lm_eval, "lora_accuracy_ll": lora_acc_ll,
        "lora_diff_lm_eval": lora_acc_lm_eval - base_acc_lm_eval, "lora_diff_ll": lora_acc_ll - base_acc_ll,
    }
    if with_steering:
        results = _eval_lora_with_steering(
            wisent_model, task, task_dict, base_acc_lm_eval, base_acc_ll,
            lora_acc_lm_eval, lora_acc_ll, steering_method, steering_layers,
            steering_num_pairs, steering_scales, extraction_strategy, device, batch_size, max_batch_size, results,
            log_interval=log_interval, min_norm_threshold=min_norm_threshold)
    remove_lora(wisent_model)
    del wisent_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _print_lora_summary(task, model_name, lora_path, base_acc_lm_eval, base_acc_ll,
                        lora_acc_lm_eval, lora_acc_ll, with_steering, steering_method, results)
    if output_dir:
        output_dir = Path(output_dir) / model_name.replace("/", "_")
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"{task}_lora_eval_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=JSON_INDENT)
        print(f"\nResults saved to: {results_file}")
    return results

def _print_lora_summary(task, model_name, lora_path, base_lm, base_ll, lora_lm, lora_ll,
                         with_steering, steering_method, results):
    print(f"\n{'='*70}\nRESULTS SUMMARY\n{'='*70}")
    print(f"Task: {task}\nModel: {model_name}\nLoRA: {lora_path}")
    print(f"{'-'*70}\n{'Method':<25} {'lm-eval acc':<15} {'LL acc':<15} {'Diff(lm-eval)':<15}\n{'-'*70}")
    print(f"{'Base':<25} {base_lm:<15.4f} {base_ll:<15.4f} {'':<15}")
    print(f"{'LoRA':<25} {lora_lm:<15.4f} {lora_ll:<15.4f} {lora_lm - base_lm:+.4f}")
    if with_steering:
        for scale, res in results["steering"]["scales"].items():
            label = f"LoRA+{steering_method.upper()}@{scale}"
            print(f"{label:<25} {res['accuracy_lm_eval']:<15.4f} {res['accuracy_ll']:<15.4f} {res['diff_from_base_lm_eval']:+.4f}")
    print(f"{'='*70}")
