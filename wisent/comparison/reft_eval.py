"""ReFT evaluation: apply_reft_to_model, remove_reft, evaluate_reft."""
from __future__ import annotations
import gc
import json
from pathlib import Path
import torch
from wisent.comparison.utils import (
    create_test_only_task, run_ll_evaluation, apply_steering_to_model, remove_steering,
)
from wisent.core.models.wisent_model import WisentModel

__all__ = ["apply_reft_to_model", "remove_reft", "evaluate_reft"]


def apply_reft_to_model(wisent_model: "WisentModel", reft_path: str | Path) -> None:
    """Apply a trained ReFT intervention to a WisentModel."""
    import pyreft
    reft_path = Path(reft_path)
    reft_model = pyreft.ReftModel.load(str(reft_path), wisent_model.hf_model)
    reft_model.set_device(wisent_model.device)
    model_dtype = next(wisent_model.hf_model.parameters()).dtype
    for k, v in reft_model.interventions.items():
        intervention = v[0] if isinstance(v, (list, tuple)) else v
        if hasattr(intervention, 'to'):
            intervention.to(dtype=model_dtype)

    class ReftModelWrapper(torch.nn.Module):
        def __init__(self, reft_model, original_model):
            super().__init__()
            self._reft_model = reft_model
            self._original_model = original_model
            self.config = original_model.config
            self.device = next(original_model.parameters()).device
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            base = {}
            if input_ids is not None:
                base["input_ids"] = input_ids
            if attention_mask is not None:
                base["attention_mask"] = attention_mask
            base.update(kwargs)
            _, outputs = self._reft_model(base=base)
            return outputs
        def __getattr__(self, name):
            if name in ("_reft_model", "_original_model", "config", "device",
                        "training", "_parameters", "_buffers", "_modules"):
                return super().__getattr__(name)
            return getattr(self._original_model, name)

    wrapper = ReftModelWrapper(reft_model, wisent_model.hf_model)
    wisent_model._original_model = wisent_model.hf_model
    wisent_model.hf_model = wrapper
    print(f"ReFT intervention loaded from {reft_path}")


def remove_reft(wisent_model: "WisentModel") -> None:
    """Remove/disable ReFT intervention from a WisentModel."""
    if hasattr(wisent_model, '_original_model'):
        wisent_model.hf_model = wisent_model._original_model
        del wisent_model._original_model
        print("ReFT intervention removed")
    else:
        print("No ReFT intervention to remove")


def evaluate_reft(
    model_name: str, reft_path: str | Path, task: str,
    train_ratio: float = 0.8, device: str = "cuda:0",
    batch_size: int = 1, max_batch_size: int = 8, limit: int | None = None,
    output_dir: str | Path = None,
    num_train_pairs: int | None = None, num_epochs: int | None = None,
    low_rank_dimension: int | None = None, intervention_layers: list[int] | None = None,
    learning_rate: float | None = None,
    with_steering: bool = False, steering_method: str = "caa",
    steering_layers: str = "12", steering_num_pairs: int = 50,
    steering_scales: list[float] | None = None, extraction_strategy: str = "mc_completion",
) -> dict:
    """Evaluate a trained ReFT intervention comparing base vs ReFT performance."""
    reft_path = Path(reft_path)
    if steering_scales is None:
        steering_scales = [1.0, 2.0, 4.0]
    print(f"\n{'='*60}\nCreating test task for: {task}\n{'='*60}")
    task_dict = create_test_only_task(task, train_ratio=train_ratio)
    print(f"\n{'='*60}\nLoading model: {model_name}\n{'='*60}")
    wisent_model = WisentModel(model_name=model_name, device=device)
    print(f"\n{'='*60}\nRunning BASE evaluation (no ReFT)\n{'='*60}")
    base_acc_ll = run_ll_evaluation(wisent_model, task_dict, task, limit)
    print(f"Base accuracy: {base_acc_ll:.4f}")
    print(f"\n{'='*60}\nApplying ReFT intervention from: {reft_path}\n{'='*60}")
    apply_reft_to_model(wisent_model, reft_path)
    print(f"\n{'='*60}\nRunning REFT evaluation\n{'='*60}")
    reft_acc_ll = run_ll_evaluation(wisent_model, task_dict, task, limit)
    print(f"ReFT accuracy: {reft_acc_ll:.4f}")
    results = {
        "task": task, "model": model_name, "reft_path": str(reft_path),
        "num_train_pairs": num_train_pairs, "num_epochs": num_epochs,
        "low_rank_dimension": low_rank_dimension, "intervention_layers": intervention_layers,
        "learning_rate": learning_rate, "train_ratio": train_ratio, "eval_limit": limit,
        "base_accuracy": base_acc_ll, "reft_accuracy": reft_acc_ll, "reft_diff": reft_acc_ll - base_acc_ll,
    }
    if with_steering:
        results = _eval_reft_with_steering(
            wisent_model, task, task_dict, limit, base_acc_ll, reft_acc_ll,
            steering_method, steering_layers, steering_num_pairs, steering_scales,
            extraction_strategy, device, results,
        )
    remove_reft(wisent_model)
    del wisent_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _print_reft_summary(task, model_name, reft_path, base_acc_ll, reft_acc_ll, with_steering, steering_method, results)
    if output_dir:
        output_dir = Path(output_dir) / model_name.replace("/", "_")
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"{task}_reft_eval_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    return results


def _eval_reft_with_steering(wisent_model, task, task_dict, limit, base_acc_ll, reft_acc_ll,
                              steering_method, steering_layers, steering_num_pairs,
                              steering_scales, extraction_strategy, device, results):
    """Run ReFT + steering evaluation at multiple scales."""
    from wisent.core.trainers.steering_trainer import WisentSteeringTrainer
    from wisent.core.steering_methods import get_steering_method
    from wisent.core.activations import ExtractionStrategy
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.contrastive_pairs.core.io.response import PositiveResponse, NegativeResponse
    from wisent.comparison.utils import generate_contrastive_pairs
    pairs_data, pairs_file = generate_contrastive_pairs(task, steering_num_pairs)
    pairs = [
        ContrastivePair(
            prompt=p["prompt"],
            positive_response=PositiveResponse(model_response=p["positive_response"]["model_response"]),
            negative_response=NegativeResponse(model_response=p["negative_response"]["model_response"]),
        ) for p in pairs_data
    ]
    pair_set = ContrastivePairSet(pairs=pairs, name=f"{task}_reft_steering")
    steering_method_obj = get_steering_method(steering_method, device=device)
    strategy = ExtractionStrategy(extraction_strategy)
    trainer = WisentSteeringTrainer(model=wisent_model, pair_set=pair_set, steering_method=steering_method_obj)
    result = trainer.run(layers_spec=steering_layers, strategy=strategy, accept_low_quality_vector=True)
    steering_vectors = {k: v.cpu().float().tolist() for k, v in result.steered_vectors.to_dict().items() if v is not None}
    steering_data = {"steering_vectors": steering_vectors, "layers": list(steering_vectors.keys())}
    import os
    os.unlink(pairs_file)
    results["steering"] = {"method": steering_method, "layers": list(steering_vectors.keys()),
                           "num_pairs": steering_num_pairs, "extraction_strategy": extraction_strategy, "scales": {}}
    for scale in steering_scales:
        apply_steering_to_model(wisent_model, steering_data, scale=scale)
        steer_acc_ll = run_ll_evaluation(wisent_model, task_dict, task, limit)
        remove_steering(wisent_model)
        results["steering"]["scales"][str(scale)] = {
            "accuracy": steer_acc_ll, "diff_from_base": steer_acc_ll - base_acc_ll,
            "diff_from_reft": steer_acc_ll - reft_acc_ll,
        }
    return results


def _print_reft_summary(task, model_name, reft_path, base_acc_ll, reft_acc_ll, with_steering, steering_method, results):
    """Print ReFT results summary."""
    print(f"\n{'='*50}\nRESULTS SUMMARY\n{'='*50}")
    print(f"Task: {task}\nModel: {model_name}\nReFT: {reft_path}")
    print(f"{'-'*50}\n{'Method':<30} {'Accuracy':<12} {'Diff':<10}\n{'-'*50}")
    print(f"{'Base':<30} {base_acc_ll:<12.4f} {'':<10}")
    print(f"{'ReFT':<30} {reft_acc_ll:<12.4f} {reft_acc_ll - base_acc_ll:+.4f}")
    if with_steering:
        for scale, res in results["steering"]["scales"].items():
            label = f"ReFT+{steering_method.upper()}@{scale}"
            print(f"{label:<30} {res['accuracy']:<12.4f} {res['diff_from_base']:+.4f}")
    print(f"{'='*50}")
