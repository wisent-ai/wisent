"""
CLI factory for creating WicherSteeringObject from enriched pairs.

Bridges the argument parser to the WICHER training pipeline:
extract args -> compute concept directions + SVD subspace per layer
-> wrap in steering object.
"""

from __future__ import annotations

import torch

from wisent.core.steering_methods.steering_object import SteeringObjectMetadata
from .wicher_steering_object import WicherSteeringObject


def _select_concept_dim(
    s_squared: torch.Tensor, explicit_dim: int, var_threshold: float
) -> int:
    """Select concept subspace dimensionality k."""
    if explicit_dim > 0:
        return max(2, min(explicit_dim, 32, len(s_squared)))
    total = s_squared.sum().item()
    if total < 1e-12:
        return 2
    cumvar = torch.cumsum(s_squared, dim=0) / total
    k = int((cumvar < var_threshold).sum().item()) + 1
    return max(2, min(k, 32, len(s_squared)))


def _create_wicher_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> WicherSteeringObject:
    """Create WICHER object with per-layer concept directions + subspace."""

    concept_dim = getattr(args, "wicher_concept_dim", 0)
    variance_threshold = getattr(args, "wicher_variance_threshold", 0.80)
    num_steps = getattr(args, "wicher_num_steps", 3)
    alpha = getattr(args, "wicher_alpha", 5e-3)
    eta = getattr(args, "wicher_eta", 0.5)
    beta = getattr(args, "wicher_beta", 0.0)
    alpha_decay = getattr(args, "wicher_alpha_decay", 1.0)

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
        diff_centered = diff - diff.mean(dim=0, keepdim=True)
        _, S, Vh = torch.linalg.svd(diff_centered, full_matrices=False)
        s_squared = S ** 2
        total_var = s_squared.sum().item()
        layer_variances[layer_int] = total_var

        k = _select_concept_dim(s_squared, concept_dim, variance_threshold)
        concept_bases[layer_int] = Vh[:k].detach()
        comp_variances[layer_int] = s_squared[:k].detach()

        explained = s_squared[:k].sum().item() / max(total_var, 1e-12)
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
    )
