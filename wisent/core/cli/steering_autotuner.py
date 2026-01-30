"""Automatic parameter tuning for steering - no hardcoded defaults."""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

from wisent.core.cli.steering_checkpoint import (
    SteeringConfig, AutotuneCheckpoint, log_config_result, setup_checkpoint_paths
)


def evaluate_steering_config(
    adapter,
    test_ids: List,
    pair_texts: Dict,
    evaluator,
    steering_vectors: Dict[str, torch.Tensor],
    layer_strengths: Dict[int, float],
    steering_method_name: str,
    pos_by_layer: Dict,
    neg_by_layer: Dict,
    max_new_tokens: int = 100,
) -> int:
    """Evaluate a steering config on test set, return number of truthful responses."""
    from wisent.core.cli.steering_behavioral import extract_response
    from wisent.core.cli.steering_viz_helpers import create_steering_method
    from wisent.core.activations.core.atoms import LayerActivations
    from wisent.core.adapters.base import SteeringConfig as AdapterSteeringConfig

    scales_dict = {f"layer.{layer}": strength for layer, strength in layer_strengths.items() if strength > 0}
    methods_dict = {}
    for layer, strength in layer_strengths.items():
        if strength > 0:
            layer_name = f"layer.{layer}"
            pos_acts = pos_by_layer.get(layer)
            neg_acts = neg_by_layer.get(layer)
            if pos_acts is not None and neg_acts is not None:
                method = create_steering_method(steering_method_name, strength, pos_acts, neg_acts)
                if method is not None:
                    methods_dict[layer_name] = method

    active_vectors = {k: v for k, v in steering_vectors.items() if k in scales_dict}
    if not active_vectors:
        return 0

    steering_vecs = LayerActivations(active_vectors)
    config = AdapterSteeringConfig(scale=scales_dict, method=methods_dict if methods_dict else None)

    truthful_count = 0
    for pair_key in test_ids:
        pair_data = pair_texts[pair_key]
        prompt = pair_data.get("prompt", "")
        pos_ref_text = pair_data.get("positive", "")
        neg_ref_text = pair_data.get("negative", "")
        correct_answers = [pos_ref_text] if pos_ref_text else []
        incorrect_answers = [neg_ref_text] if neg_ref_text else []

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = adapter.apply_chat_template(messages, add_generation_prompt=True)
        steered_response = extract_response(
            adapter.forward_with_steering(formatted_prompt, steering_vectors=steering_vecs, config=config)
        )
        result = evaluator.evaluate(steered_response, pos_ref_text,
            correct_answers=correct_answers, incorrect_answers=incorrect_answers)
        if result.ground_truth == "TRUTHFUL":
            truthful_count += 1

    return truthful_count


def search_optimal_steering_config(
    adapter, train_ids: List, val_ids: List, pair_texts: Dict, evaluator,
    pos_by_layer: Dict, neg_by_layer: Dict, layer_accuracies: Dict[int, float],
    steering_vectors: Dict[str, torch.Tensor], max_new_tokens: int = 100,
    checkpoint_dir: str = "./autotune_checkpoints", log_file: str = None,
) -> SteeringConfig:
    """Search for optimal steering parameters with checkpointing and resume capability."""
    checkpoint_file, log_file_path = setup_checkpoint_paths(checkpoint_dir, log_file)

    acc_values = sorted(layer_accuracies.values())
    acc_thresholds = [0.0] + [acc - 0.01 for acc in acc_values if acc > 0.5]
    strength_candidates = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    method_candidates = ["linear", "clamping", "contrast"]
    total_combos = len(acc_thresholds) * len(strength_candidates) * len(method_candidates)

    all_configs = []
    for acc_threshold in acc_thresholds:
        for base_strength in strength_candidates:
            for method in method_candidates:
                all_configs.append((acc_threshold, base_strength, method))

    checkpoint = AutotuneCheckpoint.load(checkpoint_file)
    if checkpoint is not None:
        print(f"  Resuming from checkpoint: {checkpoint.last_combo_idx}/{checkpoint.total_combos} completed")
        best_score = checkpoint.best_score
        best_config = SteeringConfig.from_dict(checkpoint.best_config) if checkpoint.best_config else None
        start_idx = checkpoint.last_combo_idx
        completed_configs = checkpoint.completed_configs
    else:
        print(f"  Searching {total_combos} configurations (checkpoint: {checkpoint_file})")
        best_score = -1
        best_config = None
        start_idx = 0
        completed_configs = []

    for combo_idx in range(start_idx, len(all_configs)):
        acc_threshold, base_strength, method = all_configs[combo_idx]

        layer_strengths = {}
        for layer, acc in layer_accuracies.items():
            if acc > acc_threshold:
                max_acc = max(layer_accuracies.values())
                scale = (acc - acc_threshold) / (max_acc - acc_threshold) if max_acc > acc_threshold else 1.0
                layer_strengths[layer] = base_strength * scale
            else:
                layer_strengths[layer] = 0.0

        if not any(s > 0 for s in layer_strengths.values()):
            score = 0
        else:
            score = evaluate_steering_config(
                adapter, val_ids, pair_texts, evaluator, steering_vectors,
                layer_strengths, method, pos_by_layer, neg_by_layer, max_new_tokens
            )

        result = {
            "combo_idx": combo_idx + 1, "total": total_combos, "acc_threshold": acc_threshold,
            "base_strength": base_strength, "method": method, "score": score,
            "val_size": len(val_ids), "active_layers": sum(1 for s in layer_strengths.values() if s > 0),
            "timestamp": datetime.now().isoformat(),
        }
        completed_configs.append(result)
        log_config_result(log_file_path, result)

        if score > best_score:
            best_score = score
            best_config = SteeringConfig(
                layer_strengths=layer_strengths.copy(), steering_method=method,
                accuracy_threshold=acc_threshold, base_strength=base_strength,
                val_score=score, layer_accuracies=layer_accuracies.copy(),
            )
            active_layers = sum(1 for s in layer_strengths.values() if s > 0)
            print(f"    [{combo_idx+1}/{total_combos}] New best: {score}/{len(val_ids)} "
                  f"(thresh={acc_threshold:.2f}, str={base_strength}, method={method}, layers={active_layers})")
        elif (combo_idx + 1) % 10 == 0:
            print(f"    [{combo_idx+1}/{total_combos}] Progress... best so far: {best_score}/{len(val_ids)}")

        ckpt = AutotuneCheckpoint(
            completed_configs=completed_configs,
            best_config=best_config.to_dict() if best_config else None,
            best_score=best_score, last_combo_idx=combo_idx + 1,
            total_combos=total_combos, timestamp=datetime.now().isoformat(),
        )
        ckpt.save(checkpoint_file)

    print(f"  Search complete. Log saved to: {log_file_path}")
    print(f"  Final checkpoint: {checkpoint_file}")
    return best_config


def run_autotune_multilayer(
    adapter, train_ids, test_ids, all_pair_texts, pos_by_layer, neg_by_layer,
    behavioral_acts_by_layer, behavioral_labels, val_split, max_new_tokens
):
    """Run full autotune flow for multi-layer steering."""
    from wisent.core.cli.steering_viz_helpers import (
        select_steering_direction, create_steering_method, create_all_layer_steering
    )
    from wisent.core.evaluators.rotator import EvaluatorRotator

    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.oracles")
    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")

    available_layers = sorted(pos_by_layer.keys())

    steering_vectors_dict, _, _ = create_all_layer_steering(
        pos_by_layer, neg_by_layer, "linear", 1.0, "behavioral",
        behavioral_acts_by_layer=behavioral_acts_by_layer, behavioral_labels=behavioral_labels
    )

    layer_accuracies = {}
    for layer in available_layers:
        beh_acts = behavioral_acts_by_layer.get(layer)
        _, _, acc = select_steering_direction(
            torch.from_numpy(pos_by_layer[layer]).float(),
            torch.from_numpy(neg_by_layer[layer]).float(),
            "behavioral", behavioral_activations=beh_acts, behavioral_labels=behavioral_labels
        )
        layer_accuracies[layer] = acc

    val_size = int(len(test_ids) * val_split)
    val_ids = test_ids[:val_size]
    final_test_ids = test_ids[val_size:]
    print(f"\nAUTOTUNE: Validation={len(val_ids)}, Test={len(final_test_ids)}")

    task_name = list(all_pair_texts.values())[0].get("task_name", "truthfulqa_custom") if all_pair_texts else "truthfulqa_custom"
    evaluator = EvaluatorRotator(evaluator=None, task_name=task_name).current

    print(f"\nPhase 2: Searching for optimal steering configuration...")
    optimal_config = search_optimal_steering_config(
        adapter, list(train_ids), val_ids, all_pair_texts, evaluator,
        pos_by_layer, neg_by_layer, layer_accuracies, steering_vectors_dict, max_new_tokens
    )

    print(f"\nOptimal config found:")
    print(f"  Threshold: {optimal_config.accuracy_threshold:.2f}")
    print(f"  Base strength: {optimal_config.base_strength}")
    print(f"  Method: {optimal_config.steering_method}")
    print(f"  Val score: {optimal_config.val_score}/{len(val_ids)}")
    active = [(l, s) for l, s in optimal_config.layer_strengths.items() if s > 0]
    print(f"  Active layers: {len(active)}")

    scales_dict = {f"layer.{l}": s for l, s in optimal_config.layer_strengths.items()}
    methods_dict = {}
    for layer, strength in optimal_config.layer_strengths.items():
        if strength > 0:
            method = create_steering_method(optimal_config.steering_method, strength,
                pos_by_layer[layer], neg_by_layer[layer])
            if method:
                methods_dict[f"layer.{layer}"] = method

    return optimal_config, steering_vectors_dict, methods_dict, scales_dict, final_test_ids
