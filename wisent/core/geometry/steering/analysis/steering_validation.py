"""Validation of steering effectiveness.

Measures whether steering actually changes model behavior as expected
using Cohen's d effect size by comparing outputs before and after steering.
"""
from typing import Dict, List, Optional, Any
import numpy as np
import torch


def compute_steering_effect_size(
    outputs_before: List[str],
    outputs_after: List[str],
    target_direction: str = "increase",
    metric_fn: Optional[callable] = None,
) -> Dict[str, float]:
    """
    Compute effect size of steering on model outputs.

    Args:
        outputs_before: Model outputs before steering
        outputs_after: Model outputs after steering
        target_direction: "increase" or "decrease" for target behavior
        metric_fn: Optional function to score outputs (default: length)

    Returns:
        Dict with effect_size (Cohen's d), mean_before, mean_after, significant.
    """
    if metric_fn is None:
        metric_fn = len

    scores_before = np.array([metric_fn(o) for o in outputs_before])
    scores_after = np.array([metric_fn(o) for o in outputs_after])

    mean_before = float(scores_before.mean())
    mean_after = float(scores_after.mean())
    std_pooled = np.sqrt((scores_before.var() + scores_after.var()) / 2)

    if std_pooled < 1e-10:
        effect_size = 0.0
    else:
        effect_size = (mean_after - mean_before) / std_pooled

    # Flip sign if we wanted decrease
    if target_direction == "decrease":
        effect_size = -effect_size

    # Significance: |d| > 0.5 is medium effect
    significant = abs(effect_size) > 0.5

    return {
        "effect_size_cohens_d": float(effect_size),
        "mean_before": mean_before,
        "mean_after": mean_after,
        "std_pooled": float(std_pooled),
        "direction_correct": effect_size > 0 if target_direction == "increase" else effect_size < 0,
        "significant": significant,
    }


def validate_steering_effectiveness(
    model,
    tokenizer,
    steering_vector: torch.Tensor,
    test_prompts: List[str],
    layer: int,
    steering_strength: float = 1.0,
    metric_fn: Optional[callable] = None,
    target_direction: str = "increase",
    max_new_tokens: int = 50,
) -> Dict[str, Any]:
    """
    Validate steering by comparing outputs before and after.

    Args:
        model: The model to steer
        tokenizer: Tokenizer for the model
        steering_vector: The steering vector to apply
        test_prompts: Prompts to test on
        layer: Layer to apply steering
        steering_strength: Multiplier for steering vector
        metric_fn: Function to score outputs
        target_direction: Expected direction of change
        max_new_tokens: Max tokens to generate

    Returns:
        Dict with effect size, outputs, and validation status.
    """
    outputs_before = []
    outputs_after = []

    # Generate without steering
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        outputs_before.append(text[len(prompt):])

    # Generate with steering (using hooks)
    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        hidden[:, -1, :] += steering_strength * steering_vector.to(hidden.device)
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    # Register hook on appropriate layer
    layers = model.model.layers if hasattr(model, 'model') else model.transformer.h
    handle = layers[layer].register_forward_hook(steering_hook)

    try:
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            outputs_after.append(text[len(prompt):])
    finally:
        handle.remove()

    # Compute effect size
    effect = compute_steering_effect_size(
        outputs_before, outputs_after, target_direction, metric_fn
    )

    return {
        "effect_size": effect,
        "outputs_before": outputs_before,
        "outputs_after": outputs_after,
        "n_prompts": len(test_prompts),
        "steering_strength": steering_strength,
        "layer": layer,
        "validation_passed": effect["significant"] and effect["direction_correct"],
    }


def run_full_validation(
    pos: torch.Tensor,
    neg: torch.Tensor,
    model=None,
    tokenizer=None,
    test_prompts: Optional[List[str]] = None,
    layer: int = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run full protocol with validation: test -> recommend -> validate.

    Returns complete results including steering validation if model provided.
    """
    from .signal_null_tests import compute_signal_vs_null, compute_aggregate_signal
    from .effective_dim_null import compute_effective_dimensions_vs_null
    from .geometry_null import compute_geometry_vs_null
    from .is_linear import test_linearity
    from .decomposition_metrics import find_optimal_clustering
    from .intervention_selection import rigorous_select_intervention

    # Step 1: Signal test
    signal_metrics = compute_signal_vs_null(pos, neg, ["knn_accuracy", "mlp_probe_accuracy"])
    signal_z, signal_p, _ = compute_aggregate_signal(signal_metrics)

    # Step 2: Geometry test
    linearity = test_linearity(pos, neg)
    geometry_diagnosis = "LINEAR" if linearity.is_linear else "NONLINEAR"

    # Step 3: Effective dimension
    eff_dim = compute_effective_dimensions_vs_null(pos, neg)
    eff_dim_z = eff_dim["z_scores"].get("effective_rank_z", 0)

    # Step 4: Geometry type
    geo_type = compute_geometry_vs_null(pos, neg)
    geo_type_z = geo_type.get("z_scores", {})

    # Step 5: Decomposition
    diff = pos - neg
    n_concepts, labels, sil = find_optimal_clustering(diff)

    # Make recommendation
    intervention = rigorous_select_intervention(
        signal_z, signal_p, geometry_diagnosis, linearity.confidence,
        eff_dim_z, n_concepts, sil, geo_type_z
    )

    result = {
        "signal": {"z_score": signal_z, "p_value": signal_p},
        "geometry": {"diagnosis": geometry_diagnosis, "confidence": linearity.confidence},
        "effective_dimension": eff_dim,
        "geometry_type": geo_type,
        "decomposition": {"n_concepts": n_concepts, "silhouette": sil},
        "intervention": {
            "method": intervention.recommended_method,
            "confidence": intervention.confidence,
            "confidence_bounds": [intervention.confidence_lower, intervention.confidence_upper],
            "reasoning": intervention.reasoning,
            "warnings": intervention.warnings,
        },
    }

    # Validate if model provided
    if model is not None and test_prompts and intervention.recommended_method != "NONE":
        steering_vector = (pos.mean(dim=0) - neg.mean(dim=0)).to(device)
        validation = validate_steering_effectiveness(
            model, tokenizer, steering_vector, test_prompts, layer
        )
        result["validation"] = validation

    return result
