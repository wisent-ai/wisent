"""Activation space effect measurement for find_best_method.

Measures how steering moves model representations toward/away from
the desired activation region using a trained classifier.
"""
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from wisent.core.utils.config_tools.constants import (
    COMBO_OFFSET,
    CLASSIFIER_LOG_DIVISOR,
    JSON_INDENT,
    RECURSION_INITIAL_DEPTH,
    VIZ_MLP_EPOCHS,
    VIZ_TRUTHFUL_REGION_THRESHOLD,
)
from wisent.core.utils.cli.optimize_steering.pipeline import (
    _build_config,
)
from wisent.core.utils.cli.optimize_steering.pipeline.find_best.diff import (
    _find_best_trial_dir,
)


def measure_activation_space_effect(
    output_dir: str,
    winner_method: str,
    best_params: dict,
    model_name: str,
    benchmark: str,
    train_pairs_file: str,
    test_pairs_file: str,
) -> Dict[str, Any]:
    """Measure how steering affects the activation space.

    Trains a classifier on reference (train) activations, then
    measures whether steered test activations move toward the
    positive region compared to unsteered base activations.
    """
    best_trial_dir = _find_best_trial_dir(output_dir, winner_method)
    if not best_trial_dir:
        return {"error": "No trial directories found"}

    steering_path = os.path.join(best_trial_dir, "steering.pt")
    if not os.path.exists(steering_path):
        return {"error": "No steering.pt in best trial dir"}

    config, strength = _build_config(winner_method.upper(), best_params)
    layer = _extract_layer(config)

    try:
        pos_ref, neg_ref = _get_reference_activations(
            train_pairs_file, model_name, layer, config.extraction_strategy,
        )
    except (ValueError, FileNotFoundError) as exc:
        return {"error": f"Reference activations: {exc}"}

    try:
        base_acts, steered_acts = _get_test_activations(
            model_name, test_pairs_file, steering_path, strength, layer,
        )
    except (ValueError, RuntimeError) as exc:
        return {"error": f"Test activations: {exc}"}

    metrics = _compute_metrics(pos_ref, neg_ref, base_acts, steered_acts)
    metrics.update({
        "layer": layer,
        "strength": strength,
        "method": winner_method,
    })
    steering_fig = _generate_steering_figure(
        pos_ref, neg_ref, base_acts, steered_acts,
        benchmark, winner_method, layer, strength,
    )
    if steering_fig:
        metrics["steering_figure"] = steering_fig

    effect_path = os.path.join(
        output_dir, f"activation_space_effect_{benchmark}.json",
    )
    with open(effect_path, "w") as f:
        json.dump(metrics, f, indent=JSON_INDENT)
    print(f"   Activation effect saved: {effect_path}")
    return metrics


def _extract_layer(config) -> int:
    """Extract the primary layer from a MethodConfig."""
    layer = getattr(config, "layer", None)
    if layer is not None:
        return int(layer)
    sensor = getattr(config, "sensor_layer", None)
    if sensor is not None:
        return int(sensor)
    raise ValueError("Config has no layer or sensor_layer")


def _get_reference_activations(
    train_pairs_file: str,
    model_name: str,
    layer: int,
    extraction_strategy: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract positive/negative reference activations from train pairs."""
    import argparse
    from wisent.core.utils.cli.optimize_steering.data.activations_data import (
        execute_get_activations,
    )

    work_dir = os.path.dirname(train_pairs_file)
    enriched_path = os.path.join(work_dir, "enriched_train_for_effect.json")

    if not os.path.exists(enriched_path):
        args = argparse.Namespace(
            pairs_file=train_pairs_file,
            model=model_name,
            output=enriched_path,
            layers=str(layer),
            extraction_strategy=extraction_strategy,
            device=None,
            verbose=False,
            timing=False,
            raw=False,
        )
        execute_get_activations(args)

    with open(enriched_path) as f:
        data = json.load(f)

    layer_key = str(layer)
    pos_acts: List[torch.Tensor] = []
    neg_acts: List[torch.Tensor] = []

    for pair in data.get("pairs", []):
        pos_layer = (
            pair.get("positive_response", {})
            .get("layers_activations", {})
            .get(layer_key)
        )
        neg_layer = (
            pair.get("negative_response", {})
            .get("layers_activations", {})
            .get(layer_key)
        )
        if pos_layer is not None:
            pos_acts.append(torch.tensor(pos_layer))
        if neg_layer is not None:
            neg_acts.append(torch.tensor(neg_layer))

    if not pos_acts or not neg_acts:
        raise ValueError(
            f"No activations at layer {layer} in enriched train pairs"
        )
    return torch.stack(pos_acts), torch.stack(neg_acts)


def _get_test_activations(
    model_name: str,
    test_pairs_file: str,
    steering_path: str,
    strength: float,
    layer: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract base and steered activations for test prompts."""
    from wisent.core.utils.visualization.steering.steering_viz_utils import (
        extract_base_and_steered_activations,
    )
    from wisent.core.primitives.model_interface.core.wisent import Wisent
    from wisent.core.control.steering_methods.implementations.steering_object import (
        load_steering_object,
    )

    with open(test_pairs_file) as f:
        test_data = json.load(f)

    prompts = [p.get("prompt", "") for p in test_data.get("pairs", [])]
    if not prompts:
        raise ValueError("No prompts in test pairs file")

    steering_obj = load_steering_object(steering_path)
    steering_vectors = steering_obj.to_raw_activation_map()

    wisent = Wisent.for_text(model_name)
    base_acts, steered_acts = extract_base_and_steered_activations(
        wisent, prompts, steering_vectors, layer, strength,
    )
    return base_acts, steered_acts


def _compute_metrics(
    pos_ref: torch.Tensor,
    neg_ref: torch.Tensor,
    base_acts: torch.Tensor,
    steered_acts: torch.Tensor,
) -> Dict[str, Any]:
    """Train classifier on reference data and measure activation shift."""
    from wisent.core.reading.classifiers.models.mlp import MLPClassifier
    from wisent.core.reading.classifiers.core.atoms import (
        ClassifierTrainConfig,
    )

    X = torch.cat(
        [pos_ref, neg_ref], dim=RECURSION_INITIAL_DEPTH,
    ).cpu().numpy()
    y = np.concatenate([np.ones(len(pos_ref)), np.zeros(len(neg_ref))])
    n_samples, n_features = X.shape

    clf = MLPClassifier.from_data_shape(
        n_samples, n_features,
        threshold=VIZ_TRUTHFUL_REGION_THRESHOLD, device="cpu",
    )
    config = ClassifierTrainConfig.from_data_shape(n_samples, n_features)
    log_freq = max(COMBO_OFFSET, VIZ_MLP_EPOCHS // CLASSIFIER_LOG_DIVISOR)
    report = clf.fit(X, y, log_frequency=log_freq, config=config)

    base_probs = clf.predict_proba(base_acts.cpu().numpy())
    steered_probs = clf.predict_proba(steered_acts.cpu().numpy())
    base_probs = (
        base_probs if isinstance(base_probs, list) else [base_probs]
    )
    steered_probs = (
        steered_probs if isinstance(steered_probs, list)
        else [steered_probs]
    )

    threshold = VIZ_TRUTHFUL_REGION_THRESHOLD
    base_in_pos = sum(p >= threshold for p in base_probs)
    steered_in_pos = sum(p >= threshold for p in steered_probs)
    total = len(base_probs)

    return {
        "classifier_accuracy": report.final.accuracy,
        "classifier_auc": report.final.auc,
        "base_mean_prob": float(np.mean(base_probs)),
        "steered_mean_prob": float(np.mean(steered_probs)),
        "prob_shift": float(
            np.mean(steered_probs) - np.mean(base_probs),
        ),
        "base_in_positive_region": base_in_pos,
        "steered_in_positive_region": steered_in_pos,
        "total_test_samples": total,
        "region_shift": steered_in_pos - base_in_pos,
    }


def _generate_steering_figure(
    pos_ref, neg_ref, base_acts, steered_acts,
    benchmark, method, layer, strength,
) -> str | None:
    """Generate multipanel steering effect figure. Returns base64 PNG."""
    try:
        from wisent.core.utils.visualization.steering.steering_multipanel import (
            create_steering_multipanel_figure,
        )
        title = (
            f"Steering Effect: {benchmark} — {method.upper()} "
            f"(layer {layer}, strength {strength:.2f})"
        )
        return create_steering_multipanel_figure(
            pos_activations=pos_ref, neg_activations=neg_ref,
            base_activations=base_acts, steered_activations=steered_acts,
            title=title,
        )
    except Exception as exc:
        print(f"   Warning: steering figure failed: {exc}")
        return None
