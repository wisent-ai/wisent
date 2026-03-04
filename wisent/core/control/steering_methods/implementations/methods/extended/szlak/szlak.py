"""
Attention-Transport steering method (SZLAK).

Uses attention-affinity cost matrix C_ij = -q_i·k_j/√d_k and one-sided
entropic optimal transport to compute per-source displacements.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn.functional as F

from wisent.core.control.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.utils.infra_tools.errors import InsufficientDataError
from wisent.core.utils.config_tools.constants import LOG_EPS

from .transport import (
    compute_attention_affinity_cost,
    sinkhorn_one_sided,
)

__all__ = [
    "SzlakMethod",
    "SzlakConfig",
    "SzlakResult",
]


def _require(name: str, kwargs: dict):
    """Raise ValueError if a required hyperparameter is missing."""
    if name not in kwargs:
        raise ValueError(
            f"Parameter '{name}' is required. "
            f"Run 'wisent optimize-steering auto' first, or pass it explicitly."
        )
    return kwargs[name]


@dataclass
class SzlakConfig:
    """Configuration for Attention-Transport steering method."""
    sinkhorn_reg: float
    """Entropic regularization for one-sided EOT solver."""
    max_iter: int
    """Maximum Sinkhorn iterations (unused for one-sided, kept for API consistency)."""
    inference_k: int
    """Number of nearest source points for inference interpolation."""


@dataclass
class SzlakResult:
    """Result from Geodesic OT training with per-layer transport data."""
    source_points: Dict[LayerName, torch.Tensor]
    """Per-layer source (negative) activations [N_neg, D]."""
    displacements: Dict[LayerName, torch.Tensor]
    """Per-layer transport displacements [N_neg, D]."""
    mean_displacement: Dict[LayerName, torch.Tensor]
    """Per-layer mean displacement vector [D]."""
    config: SzlakConfig
    layer_order: List[LayerName]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SzlakMethod(BaseSteeringMethod):
    """
    Attention-Transport steering method.

    Computes attention-affinity cost C = -(q@k.T)/√d_k and solves
    one-sided entropic OT to derive displacements.
    """

    name = "szlak"
    description = (
        "Attention-transport steering via EOT cost inversion"
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = SzlakConfig(
            sinkhorn_reg=_require("sinkhorn_reg", kwargs),
            max_iter=_require("max_iter", kwargs),
            inference_k=_require("inference_k", kwargs),
        )

    def train(self, pair_set: ContrastivePairSet) -> LayerActivations:
        """Compatibility interface: returns mean transport vectors per layer."""
        result = self.train_szlak(pair_set)
        primary_map: RawActivationMap = {}
        for layer in result.layer_order:
            vec = result.mean_displacement[layer]
            vec = F.normalize(vec.float(), p=2, dim=-1)
            primary_map[layer] = vec
        dtype = self.kwargs.get("dtype", None)
        return LayerActivations(primary_map, dtype=dtype)

    def train_szlak(
        self,
        pair_set: ContrastivePairSet,
        qk_activations: Dict[LayerName, Tuple[torch.Tensor, torch.Tensor]] | None = None,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
    ) -> SzlakResult:
        """Full attention-transport training with per-layer displacements.

        Args:
            pair_set: Contrastive pair set with activations.
            qk_activations: Q/K projections per layer (layer_name -> (q_neg, k_pos)).
            num_heads: Number of attention heads (for GQA handling).
            num_kv_heads: Number of key-value heads (for GQA handling).
        """
        buckets = self._collect_from_set(pair_set)
        if not buckets:
            raise InsufficientDataError(reason="No valid activation pairs found")
        if not qk_activations:
            raise InsufficientDataError(reason="Q/K projections required for attention-transport")
        source_points: Dict[LayerName, torch.Tensor] = {}
        displacements_dict: Dict[LayerName, torch.Tensor] = {}
        mean_disp: Dict[LayerName, torch.Tensor] = {}
        layer_order: List[LayerName] = []
        layer_meta: Dict[str, Any] = {}
        for layer_name in sorted(buckets.keys()):
            pos_list, neg_list = buckets[layer_name]
            if not pos_list or not neg_list:
                continue
            if layer_name not in qk_activations:
                continue
            pos = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
            neg = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
            q_neg, k_pos = qk_activations[layer_name]
            cost = compute_attention_affinity_cost(
                q_neg.float(), k_pos.float(), num_heads=num_heads, num_kv_heads=num_kv_heads)
            T = sinkhorn_one_sided(cost, reg=self.config.sinkhorn_reg, max_iter=self.config.max_iter)
            row_sums = T.sum(dim=1, keepdim=True).clamp(min=LOG_EPS)
            T_norm = T / row_sums
            targets = T_norm @ pos
            delta = targets - neg
            source_points[layer_name] = neg.detach()
            displacements_dict[layer_name] = delta.detach()
            mean_disp[layer_name] = delta.mean(dim=0).detach()
            layer_order.append(layer_name)
            layer_meta[str(layer_name)] = {
                "n_neg": neg.shape[0], "n_pos": pos.shape[0],
                "mean_displacement_norm": delta.mean(dim=0).norm().item(),
                "max_cost": cost.max().item(),
            }
        if not layer_order:
            raise InsufficientDataError(reason="No layers produced valid transport")
        return SzlakResult(
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
