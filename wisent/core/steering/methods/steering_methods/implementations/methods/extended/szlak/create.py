"""
CLI factory for creating SzlakSteeringObject from enriched pairs.

Bridges the argument parser to the attention-transport training pipeline:
extract args -> compute EOT cost inversion per layer -> wrap in steering object.
"""

from __future__ import annotations

import torch

from wisent.core.steering_methods.steering_object import SteeringObjectMetadata
from .szlak_steering_object import SzlakSteeringObject
from .transport import compute_attention_affinity_cost, sinkhorn_one_sided
from wisent.core.constants import LOG_EPS, SZLAK_SINKHORN_REG, SZLAK_INFERENCE_K


def _create_szlak_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> SzlakSteeringObject:
    """Create attention-transport steering object with per-layer displacements."""
    sinkhorn_reg = getattr(args, "szlak_sinkhorn_reg", SZLAK_SINKHORN_REG)
    inference_k = getattr(args, "szlak_inference_k", SZLAK_INFERENCE_K)
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
            raise ValueError(f"Layer {layer_str}: Q/K projections required for SZLAK attention-transport")
        q_neg = torch.stack([t.detach().float().reshape(-1) for t in q_data], dim=0)
        k_pos = torch.stack([t.detach().float().reshape(-1) for t in k_data], dim=0)
        cost = compute_attention_affinity_cost(q_neg, k_pos, num_heads=num_heads, num_kv_heads=num_kv_heads)
        T = sinkhorn_one_sided(cost, reg=sinkhorn_reg)
        row_sums = T.sum(dim=1, keepdim=True).clamp(min=LOG_EPS)
        T_norm = T / row_sums
        targets = T_norm @ pos
        delta = targets - neg
        layer_int = int(layer_str)
        source_points[layer_int] = neg.detach()
        disps[layer_int] = delta.detach()
        disp_norm = delta.mean(dim=0).norm().item()
        print(f"   Layer {layer_str}: n_neg={neg.shape[0]}, n_pos={pos.shape[0]}, "
              f"mean_disp_norm={disp_norm:.4f}")
    return SzlakSteeringObject(
        metadata=metadata,
        source_points=source_points,
        displacements=disps,
        inference_k=inference_k,
    )
