"""
Geodesic Optimal Transport steering method.

Builds a k-NN graph on activation space, computes geodesic distances,
and solves OT via Sinkhorn to produce per-source displacements that
respect the manifold geometry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn.functional as F

from wisent.core.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.errors import InsufficientDataError

from .transport import compute_geodesic_cost, sinkhorn

__all__ = [
    "GeodesicOTMethod",
    "GeodesicOTConfig",
    "GeodesicOTResult",
]


@dataclass
class GeodesicOTConfig:
    """Configuration for Geodesic OT steering method."""
    k_neighbors: int = 10
    """Number of nearest neighbors for k-NN graph construction."""
    sinkhorn_reg: float = 0.1
    """Entropic regularization for Sinkhorn solver."""
    sinkhorn_max_iter: int = 100
    """Maximum iterations for Sinkhorn convergence."""
    inference_k: int = 5
    """Number of nearest source points for inference interpolation."""


@dataclass
class GeodesicOTResult:
    """Result from Geodesic OT training with per-layer transport data."""
    source_points: Dict[LayerName, torch.Tensor]
    """Per-layer source (negative) activations [N_neg, D]."""
    displacements: Dict[LayerName, torch.Tensor]
    """Per-layer transport displacements [N_neg, D]."""
    mean_displacement: Dict[LayerName, torch.Tensor]
    """Per-layer mean displacement vector [D]."""
    config: GeodesicOTConfig
    layer_order: List[LayerName]
    metadata: Dict[str, Any] = field(default_factory=dict)


class GeodesicOTMethod(BaseSteeringMethod):
    """
    Geodesic Optimal Transport steering method.

    Builds a k-NN graph on concatenated neg/pos activations, computes
    geodesic distances via shortest paths, and solves entropic OT with
    Sinkhorn to derive manifold-respecting displacements.
    """

    name = "geodesic_ot"
    description = (
        "Manifold-aware optimal transport via geodesic distances - "
        "geometry-agnostic steering without shape assumptions"
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = GeodesicOTConfig(
            k_neighbors=kwargs.get("k_neighbors", 10),
            sinkhorn_reg=kwargs.get("sinkhorn_reg", 0.1),
            sinkhorn_max_iter=kwargs.get("sinkhorn_max_iter", 100),
            inference_k=kwargs.get("inference_k", 5),
        )

    def train(self, pair_set: ContrastivePairSet) -> LayerActivations:
        """Compatibility interface: returns mean transport vectors per layer."""
        result = self.train_geodesic_ot(pair_set)
        primary_map: RawActivationMap = {}
        for layer in result.layer_order:
            vec = result.mean_displacement[layer]
            vec = F.normalize(vec.float(), p=2, dim=-1)
            primary_map[layer] = vec
        dtype = self.kwargs.get("dtype", None)
        return LayerActivations(primary_map, dtype=dtype)

    def train_geodesic_ot(self, pair_set: ContrastivePairSet) -> GeodesicOTResult:
        """Full Geodesic OT training with per-layer source points + displacements."""
        buckets = self._collect_from_set(pair_set)
        if not buckets:
            raise InsufficientDataError(reason="No valid activation pairs found")

        source_points: Dict[LayerName, torch.Tensor] = {}
        displacements_dict: Dict[LayerName, torch.Tensor] = {}
        mean_disp: Dict[LayerName, torch.Tensor] = {}
        layer_order: List[LayerName] = []
        layer_meta: Dict[str, Any] = {}

        for layer_name in sorted(buckets.keys()):
            pos_list, neg_list = buckets[layer_name]
            if not pos_list or not neg_list:
                continue
            pos = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
            neg = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)

            # 1. Compute geodesic cost matrix
            cost, _ = compute_geodesic_cost(neg, pos, k=self.config.k_neighbors)

            # 2. Solve OT via Sinkhorn
            T = sinkhorn(cost, reg=self.config.sinkhorn_reg, max_iter=self.config.sinkhorn_max_iter)

            # 3. Row-normalize transport plan for per-source targets
            row_sums = T.sum(dim=1, keepdim=True).clamp(min=1e-12)
            T_norm = T / row_sums

            # 4. Per-source displacement: where OT says each neg point should go
            targets = T_norm @ pos  # [N_neg, D]
            delta = targets - neg   # [N_neg, D]

            source_points[layer_name] = neg.detach()
            displacements_dict[layer_name] = delta.detach()
            mean_disp[layer_name] = delta.mean(dim=0).detach()
            layer_order.append(layer_name)

            layer_meta[str(layer_name)] = {
                "n_neg": neg.shape[0],
                "n_pos": pos.shape[0],
                "mean_displacement_norm": delta.mean(dim=0).norm().item(),
                "max_cost": cost.max().item(),
            }

        if not layer_order:
            raise InsufficientDataError(reason="No layers produced valid transport")

        return GeodesicOTResult(
            source_points=source_points,
            displacements=displacements_dict,
            mean_displacement=mean_disp,
            config=self.config,
            layer_order=layer_order,
            metadata={"per_layer": layer_meta},
        )

    def _collect_from_set(
        self, pair_set: ContrastivePairSet
    ) -> Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Build {layer_name: ([pos tensors], [neg tensors])} from pairs."""
        buckets: Dict[
            LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]
        ] = defaultdict(lambda: ([], []))
        for pair in pair_set.pairs:
            pos_la = getattr(pair.positive_response, "layers_activations", None)
            neg_la = getattr(pair.negative_response, "layers_activations", None)
            if pos_la is None or neg_la is None:
                continue
            layer_names = set(pos_la.to_dict().keys()) | set(neg_la.to_dict().keys())
            for layer in layer_names:
                p = pos_la.to_dict().get(layer) if pos_la else None
                n = neg_la.to_dict().get(layer) if neg_la else None
                if isinstance(p, torch.Tensor) and isinstance(n, torch.Tensor):
                    buckets[layer][0].append(p)
                    buckets[layer][1].append(n)
        return buckets
