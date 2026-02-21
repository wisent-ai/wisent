"""
CLI factory for creating PrzelomSteeringObject from enriched pairs.

Bridges the argument parser to the attention-transport training pipeline:
extract args -> compute EOT cost inversion per layer -> wrap in steering object.
"""

from __future__ import annotations

import math
import torch

from wisent.core.steering_methods.steering_object import SteeringObjectMetadata
from .przelom_steering_object import PrzelomSteeringObject
from wisent.core.steering_methods.methods.szlak.transport import (
    compute_attention_affinity_cost,
    sinkhorn_one_sided,
)


def _regularized_pinv(M: torch.Tensor, reg: float = 1e-4) -> torch.Tensor:
    """Tikhonov-regularized pseudoinverse: (M^T M + reg I)^-1 M^T."""
    MtM = M.T @ M
    I = torch.eye(MtM.shape[0], device=M.device, dtype=M.dtype)
    return torch.linalg.solve(MtM + reg * I, M.T)


def _compute_target_transport(neg: torch.Tensor, pos: torch.Tensor, mode: str) -> torch.Tensor:
    """Compute target transport plan T_target [N_neg, N_pos]."""
    N_neg, N_pos = neg.shape[0], pos.shape[0]
    if mode == "nearest":
        dists = torch.cdist(neg, pos)
        nearest_idx = dists.argmin(dim=1)
        T = torch.zeros(N_neg, N_pos, device=neg.device, dtype=neg.dtype)
        T[torch.arange(N_neg, device=neg.device), nearest_idx] = 1.0 / N_neg
    else:
        T = torch.ones(N_neg, N_pos, device=neg.device, dtype=neg.dtype) / (N_neg * N_pos)
    return T


def _create_przelom_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> PrzelomSteeringObject:
    """Create attention-transport steering object with per-layer displacements."""
    epsilon = getattr(args, "przelom_epsilon", 1.0)
    target_mode = getattr(args, "przelom_target_mode", "uniform")
    regularization = getattr(args, "przelom_regularization", 1e-4)
    inference_k = getattr(args, "przelom_inference_k", 5)

    num_heads = metadata.extra.get('num_attention_heads')
    num_kv_heads = metadata.extra.get('num_key_value_heads')
    source_points = {}
    disps = {}
    for layer_str in available_layers:
        pos_list = layer_activations[layer_str]["positive"]
        neg_list = layer_activations[layer_str]["negative"]
        if not pos_list or not neg_list:
            continue
        pos = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
        q_data = layer_activations[layer_str].get("q_proj_activations")
        k_data = layer_activations[layer_str].get("k_proj_activations")
        if q_data is None or k_data is None:
            raise ValueError(f"Layer {layer_str}: Q/K projections required for PRZELOM")
        q_neg = torch.stack([t.detach().float().reshape(-1) for t in q_data], dim=0)
        k_pos = torch.stack([t.detach().float().reshape(-1) for t in k_data], dim=0)
        C = compute_attention_affinity_cost(q_neg, k_pos, num_heads=num_heads, num_kv_heads=num_kv_heads)
        T_current = torch.softmax(-C / epsilon, dim=1)
        T_target = _compute_target_transport(neg, pos, target_mode)
        log_target = torch.log(T_target.clamp(min=1e-12))
        log_current = torch.log(T_current.clamp(min=1e-12))
        delta_C = epsilon * (log_target - log_current)
        k_pos_pinv = _regularized_pinv(k_pos, regularization)
        delta_q = -math.sqrt(q_neg.shape[-1]) * (delta_C @ k_pos_pinv.T)
        delta_h = delta_q
        layer_int = int(layer_str)
        source_points[layer_int] = neg.detach()
        disps[layer_int] = delta_h.detach()
        disp_norm = delta_h.mean(dim=0).norm().item()
        print(f"   Layer {layer_str}: n_neg={neg.shape[0]}, n_pos={pos.shape[0]}, "
              f"mean_disp_norm={disp_norm:.4f}")

    return PrzelomSteeringObject(
        metadata=metadata,
        source_points=source_points,
        displacements=disps,
        inference_k=inference_k,
    )
