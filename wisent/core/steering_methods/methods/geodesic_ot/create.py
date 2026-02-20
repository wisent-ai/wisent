"""
CLI factory for creating GeodesicOTSteeringObject from enriched pairs.

Bridges the argument parser to the Geodesic OT training pipeline:
extract args -> compute geodesic OT per layer -> wrap in steering object.
"""

from __future__ import annotations

import torch

from wisent.core.steering_methods.steering_object import SteeringObjectMetadata
from .geodesic_ot import GeodesicOTMethod
from .geodesic_ot_steering_object import GeodesicOTSteeringObject
from .transport import compute_geodesic_cost, sinkhorn


def _create_geodesic_ot_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> GeodesicOTSteeringObject:
    """Create Geodesic OT steering object with per-layer displacements."""

    k_neighbors = getattr(args, "geodesic_ot_k_neighbors", 10)
    sinkhorn_reg = getattr(args, "geodesic_ot_sinkhorn_reg", 0.1)
    sinkhorn_max_iter = getattr(args, "geodesic_ot_sinkhorn_max_iter", 100)
    inference_k = getattr(args, "geodesic_ot_inference_k", 5)

    source_points = {}
    disps = {}

    for layer_str in available_layers:
        pos_list = layer_activations[layer_str]["positive"]
        neg_list = layer_activations[layer_str]["negative"]
        if not pos_list or not neg_list:
            continue

        pos = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)

        # 1. Compute geodesic cost matrix
        cost, _ = compute_geodesic_cost(neg, pos, k=k_neighbors)

        # 2. Solve OT via Sinkhorn
        T = sinkhorn(cost, reg=sinkhorn_reg, max_iter=sinkhorn_max_iter)

        # 3. Row-normalize for per-source targets
        row_sums = T.sum(dim=1, keepdim=True).clamp(min=1e-12)
        T_norm = T / row_sums

        # 4. Per-source displacement
        targets = T_norm @ pos
        delta = targets - neg

        layer_int = int(layer_str)
        source_points[layer_int] = neg.detach()
        disps[layer_int] = delta.detach()

        disp_norm = delta.mean(dim=0).norm().item()
        print(f"   Layer {layer_str}: n_neg={neg.shape[0]}, n_pos={pos.shape[0]}, "
              f"mean_disp_norm={disp_norm:.4f}")

    return GeodesicOTSteeringObject(
        metadata=metadata,
        source_points=source_points,
        displacements=disps,
        inference_k=inference_k,
    )
