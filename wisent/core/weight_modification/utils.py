"""
Utility functions for weight modification.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from torch.nn import Module
    from torch import Tensor

__all__ = [
    "get_modifiable_components",
    "verify_modification",
    "compute_modification_metrics",
]

_LOG = setup_logger(__name__)


def get_modifiable_components(model: Module) -> list[tuple[str, list[str]]]:
    """
    Get list of modifiable components in model.

    Returns:
        List of (component_path, subcomponents) tuples

    Example:
        >>> components = get_modifiable_components(model)
        >>> for path, subcomps in components:
        ...     print(f"{path}: {subcomps}")
        self_attn: ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        mlp: ['gate_proj', 'up_proj', 'down_proj']
    """
    log = bind(_LOG)

    # Get layers
    if hasattr(model, "model"):
        sample_layer = model.model.layers[0]
    elif hasattr(model, "transformer"):
        sample_layer = model.transformer.h[0]
    else:
        sample_layer = model.layers[0]

    components = []

    # Check attention
    if hasattr(sample_layer, "self_attn"):
        attn = sample_layer.self_attn
        attn_subcomps = []
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if hasattr(attn, attr):
                attn_subcomps.append(attr)
        components.append(("self_attn", attn_subcomps))

    # Check MLP
    if hasattr(sample_layer, "mlp"):
        mlp = sample_layer.mlp
        mlp_subcomps = []
        for attr in ["gate_proj", "up_proj", "down_proj", "fc1", "fc2"]:
            if hasattr(mlp, attr):
                mlp_subcomps.append(attr)
        components.append(("mlp", mlp_subcomps))

    log.info("Found modifiable components", extra={"components": components})

    return components


def verify_modification(
    model: Module,
    original_state_dict: dict | None = None,
    expected_changes: int | None = None,
) -> dict[str, any]:
    """
    Verify that model was modified correctly.

    Args:
        model: Modified model
        original_state_dict: Optional original state dict to compare against
        expected_changes: Expected number of parameter changes

    Returns:
        Dictionary with verification results:
        - "verified": bool
        - "params_changed": int
        - "issues": list of issues found

    Example:
        >>> original_state = copy.deepcopy(model.state_dict())
        >>> project_weights(model, steering_vectors)
        >>> result = verify_modification(model, original_state, expected_changes=64)
        >>> assert result["verified"]
    """
    log = bind(_LOG)

    issues = []
    params_changed = 0

    if original_state_dict is not None:
        current_state = model.state_dict()

        for key in original_state_dict.keys():
            if key not in current_state:
                issues.append(f"Missing key in current state: {key}")
                continue

            orig = original_state_dict[key]
            curr = current_state[key]

            if orig.shape != curr.shape:
                issues.append(f"Shape mismatch for {key}: {orig.shape} vs {curr.shape}")
                continue

            diff = (curr - orig).abs()
            changed = (diff > 1e-6).sum().item()

            if changed > 0:
                params_changed += changed

        if expected_changes is not None:
            if params_changed != expected_changes:
                issues.append(
                    f"Expected {expected_changes} params changed, got {params_changed}"
                )

    # Check for NaN or Inf
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            issues.append(f"NaN detected in {name}")
        if torch.isinf(param).any():
            issues.append(f"Inf detected in {name}")

    verified = len(issues) == 0

    result = {
        "verified": verified,
        "params_changed": params_changed,
        "issues": issues,
    }

    if verified:
        log.info("Modification verified successfully", extra=result)
    else:
        log.warning("Verification failed", extra=result)

    return result


def compute_modification_metrics(
    original_model: Module,
    modified_model: Module,
    sample_inputs: Tensor | None = None,
) -> dict[str, float]:
    """
    Compute metrics quantifying the modification.

    Args:
        original_model: Original model
        modified_model: Modified model
        sample_inputs: Optional sample inputs to test output differences

    Returns:
        Dictionary with metrics:
        - "weight_l2_distance": L2 distance between weight matrices
        - "weight_cosine_similarity": Cosine similarity of weights
        - "output_difference": If sample_inputs provided, difference in outputs

    Example:
        >>> metrics = compute_modification_metrics(original, modified, sample_inputs)
        >>> print(f"Weight distance: {metrics['weight_l2_distance']:.4f}")
    """
    log = bind(_LOG)

    # Compare weights
    orig_state = original_model.state_dict()
    mod_state = modified_model.state_dict()

    weight_diff_sum = 0.0
    weight_norm_sum = 0.0
    weight_dot_product = 0.0
    weight_orig_norm = 0.0
    weight_mod_norm = 0.0

    for key in orig_state.keys():
        if key not in mod_state:
            continue

        orig = orig_state[key].float()
        mod = mod_state[key].float()

        if orig.shape != mod.shape:
            continue

        diff = (mod - orig)
        weight_diff_sum += diff.norm().item() ** 2
        weight_norm_sum += orig.norm().item() ** 2

        # Flatten for dot product
        orig_flat = orig.flatten()
        mod_flat = mod.flatten()

        weight_dot_product += (orig_flat @ mod_flat).item()
        weight_orig_norm += (orig_flat @ orig_flat).item()
        weight_mod_norm += (mod_flat @ mod_flat).item()

    l2_distance = weight_diff_sum ** 0.5
    cosine_sim = weight_dot_product / ((weight_orig_norm * weight_mod_norm) ** 0.5)

    metrics = {
        "weight_l2_distance": l2_distance,
        "weight_cosine_similarity": cosine_sim,
    }

    # Compare outputs if sample inputs provided
    if sample_inputs is not None:
        with torch.no_grad():
            orig_outputs = original_model(sample_inputs)
            mod_outputs = modified_model(sample_inputs)

            if hasattr(orig_outputs, "logits"):
                orig_logits = orig_outputs.logits
                mod_logits = mod_outputs.logits
            else:
                orig_logits = orig_outputs
                mod_logits = mod_outputs

            output_diff = (mod_logits - orig_logits).abs().mean().item()
            metrics["output_difference"] = output_diff

    log.info("Computed modification metrics", extra=metrics)

    return metrics
