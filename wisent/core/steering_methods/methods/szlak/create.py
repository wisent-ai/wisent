"""
CLI factory for creating SzlakSteeringObject from enriched pairs.

Bridges the argument parser to the Geodesic OT training pipeline:
extract args -> compute OT per layer -> wrap in steering object.
Supports geodesic (default) and attention_affinity cost modes.
"""

from __future__ import annotations

import torch

from wisent.core.steering_methods.steering_object import SteeringObjectMetadata
from .szlak import SzlakMethod
from .szlak_steering_object import SzlakSteeringObject
from .transport import (
    compute_geodesic_cost,
    sinkhorn,
    compute_attention_affinity_cost,
    sinkhorn_one_sided,
)


def _create_szlak_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> SzlakSteeringObject:
    """Create OT steering object with per-layer displacements."""

    k_neighbors = getattr(args, "szlak_k_neighbors", 10)
    sinkhorn_reg = getattr(args, "szlak_sinkhorn_reg", 0.1)
    sinkhorn_max_iter = getattr(args, "szlak_sinkhorn_max_iter", 100)
    inference_k = getattr(args, "szlak_inference_k", 5)
    cost_mode = getattr(args, "szlak_cost_mode", "geodesic")

    source_points = {}
    disps = {}

    for layer_str in available_layers:
        pos_list = layer_activations[layer_str]["positive"]
        neg_list = layer_activations[layer_str]["negative"]
        if not pos_list or not neg_list:
            continue

        pos = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)

        if cost_mode == "attention_affinity":
            q_data = layer_activations[layer_str].get("q_proj_activations")
            k_data = layer_activations[layer_str].get("k_proj_activations")
            if q_data is not None and k_data is not None:
                q_neg = torch.stack([t.detach().float().reshape(-1) for t in q_data], dim=0)
                k_pos = torch.stack([t.detach().float().reshape(-1) for t in k_data], dim=0)
                cost = compute_attention_affinity_cost(q_neg, k_pos)
                T = sinkhorn_one_sided(cost, reg=sinkhorn_reg)
            else:
                cost, _ = compute_geodesic_cost(neg, pos, k=k_neighbors)
                T = sinkhorn(cost, reg=sinkhorn_reg, max_iter=sinkhorn_max_iter)
        else:
            cost, _ = compute_geodesic_cost(neg, pos, k=k_neighbors)
            T = sinkhorn(cost, reg=sinkhorn_reg, max_iter=sinkhorn_max_iter)

        row_sums = T.sum(dim=1, keepdim=True).clamp(min=1e-12)
        T_norm = T / row_sums
        targets = T_norm @ pos
        delta = targets - neg

        layer_int = int(layer_str)
        source_points[layer_int] = neg.detach()
        disps[layer_int] = delta.detach()

        disp_norm = delta.mean(dim=0).norm().item()
        print(f"   Layer {layer_str}: n_neg={neg.shape[0]}, n_pos={pos.shape[0]}, "
              f"mean_disp_norm={disp_norm:.4f}, cost_mode={cost_mode}")

    return SzlakSteeringObject(
        metadata=metadata,
        source_points=source_points,
        displacements=disps,
        inference_k=inference_k,
    )
