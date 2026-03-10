"""
CLI factory for creating WicherSteeringObject from enriched pairs.

Bridges the argument parser to the WICHER training pipeline:
extract args -> compute concept directions + SVD subspace per layer
-> wrap in steering object.
"""

from __future__ import annotations

import torch

from wisent.core.control.steering_methods.steering_object import SteeringObjectMetadata
from .wicher_steering_object import WicherSteeringObject
from wisent.core.utils.config_tools.constants import (
    LOG_EPS, WICHER_CONCEPT_DIM_MIN, WICHER_CONCEPT_DIM_MAX,
    WICHER_DEFAULT_SOLVER,
)


def _require_arg(args, attr_name):
    val = getattr(args, attr_name, None)
    if val is None:
        raise ValueError(
            f"Parameter '{attr_name}' is required. "
            f"Run 'wisent optimize-steering auto' first, or pass it explicitly."
        )
    return val


def _select_concept_dim(
    s_squared: torch.Tensor, explicit_dim: int, var_threshold: float,
    min_concept_dim: int = None, max_concept_dim: int = None,
) -> int:
    """Select concept subspace dimensionality k."""
    if min_concept_dim is None:
        raise ValueError("min_concept_dim is required")
    if max_concept_dim is None:
        raise ValueError("max_concept_dim is required")
    if explicit_dim > min_concept_dim:
        return max(min_concept_dim, min(explicit_dim, max_concept_dim, len(s_squared)))
    total = s_squared.sum().item()
    if total < LOG_EPS:
        return min_concept_dim
    cumvar = torch.cumsum(s_squared, dim=0) / total
    k = int((cumvar < var_threshold).sum().item()) + 1
    return max(min_concept_dim, min(k, max_concept_dim, len(s_squared)))


def _create_wicher_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> WicherSteeringObject:
    """Create WICHER object with per-layer concept directions + subspace."""

    concept_dim = _require_arg(args, "wicher_concept_dim")
    variance_threshold = _require_arg(args, "wicher_variance_threshold")
    num_steps = _require_arg(args, "wicher_num_steps")
    alpha = _require_arg(args, "wicher_alpha")
    eta = _require_arg(args, "wicher_eta")
    beta = _require_arg(args, "wicher_beta")
    alpha_decay = _require_arg(args, "wicher_alpha_decay")
    solver = getattr(args, "wicher_solver", WICHER_DEFAULT_SOLVER)

    concept_dirs = {}
    concept_bases = {}
    comp_variances = {}
    layer_variances = {}

    for layer_str in available_layers:
        pos_list = layer_activations[layer_str]["positive"]
        neg_list = layer_activations[layer_str]["negative"]
        if not pos_list or not neg_list:
            continue

        pos = torch.stack(
            [t.detach().float().reshape(-1) for t in pos_list], dim=0
        )
        neg = torch.stack(
            [t.detach().float().reshape(-1) for t in neg_list], dim=0
        )

        direction = pos.mean(dim=0) - neg.mean(dim=0)
        layer_int = int(layer_str)
        concept_dirs[layer_int] = direction.detach()

        diff = pos - neg
        _, S, Vh = torch.linalg.svd(diff, full_matrices=False)
        s_squared = S ** 2
        total_var = s_squared.sum().item()
        layer_variances[layer_int] = total_var

        k = _select_concept_dim(
            s_squared, concept_dim, variance_threshold,
            min_concept_dim=WICHER_CONCEPT_DIM_MIN,
            max_concept_dim=WICHER_CONCEPT_DIM_MAX,
        )
        concept_bases[layer_int] = Vh[:k].detach()
        comp_variances[layer_int] = s_squared[:k].detach()

        explained = s_squared[:k].sum().item() / max(total_var, LOG_EPS)
        dir_norm = direction.norm().item()
        print(
            f"   Layer {layer_str}: k={k}, var_explained={explained:.3f}, "
            f"n_pos={pos.shape[0]}, n_neg={neg.shape[0]}, "
            f"dir_norm={dir_norm:.4f}"
        )

    return WicherSteeringObject(
        metadata=metadata,
        concept_directions=concept_dirs,
        concept_bases=concept_bases,
        component_variances=comp_variances,
        num_steps=num_steps,
        alpha=alpha,
        eta=eta,
        beta=beta,
        alpha_decay=alpha_decay,
        layer_variance=layer_variances,
        solver=solver,
    )
