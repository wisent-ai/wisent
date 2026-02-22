"""
PRZELOM — Attention-Transport Steering Method.

Operates directly on the attention EOT cost matrix: computes the desired
transport plan, then solves for the activation perturbation via Q-projection
pseudoinverse. Polish "przelom" means "breakthrough/passage".

Algorithm per layer:
  1. Compute attention-affinity cost: C = -(q_neg @ k_pos.T) / sqrt(d_k)
  2. Compute current transport: T_current = softmax(-C / epsilon)
  3. Compute target transport: T_target (uniform or nearest-neighbor)
  4. delta_C = epsilon * (log(T_target) - log(T_current))
  5. delta_q = -sqrt(d_k) * delta_C @ pinv(k_pos)
  6. delta_h = delta_q @ pinv(W_q)  (query perturbation -> residual stream)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn.functional as F

from wisent.core.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.errors import InsufficientDataError

__all__ = ["PrzelomMethod", "PrzelomConfig", "PrzelomResult"]


@dataclass
class PrzelomConfig:
    """Configuration for PRZELOM attention-transport steering."""
    epsilon: float = 1.0
    """Entropic regularization (temperature for softmax)."""
    target_mode: str = "uniform"
    """Target transport plan mode: 'uniform' or 'nearest'."""
    regularization: float = 1e-4
    """Tikhonov regularization for pseudoinverse."""
    inference_k: int = 5
    """Number of nearest source points for inference interpolation."""


@dataclass
class PrzelomResult:
    """Result from PRZELOM training with per-layer transport data."""
    source_points: Dict[LayerName, torch.Tensor]
    displacements: Dict[LayerName, torch.Tensor]
    mean_displacement: Dict[LayerName, torch.Tensor]
    config: PrzelomConfig
    layer_order: List[LayerName]
    metadata: Dict[str, Any] = field(default_factory=dict)


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


def _regularized_pinv(M: torch.Tensor, reg: float = 1e-4) -> torch.Tensor:
    """Compute Tikhonov-regularized pseudoinverse: (M^T M + reg I)^-1 M^T."""
    MtM = M.T @ M
    I = torch.eye(MtM.shape[0], device=M.device, dtype=M.dtype)
    return torch.linalg.solve(MtM + reg * I, M.T)


class PrzelomMethod(BaseSteeringMethod):
    """Attention-transport steering via EOT cost matrix inversion."""

    name = "przelom"
    description = "Attention-transport steering: computes desired transport plan and inverts through Q-projection"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = PrzelomConfig(
            epsilon=kwargs.get("epsilon", 1.0),
            target_mode=kwargs.get("target_mode", "uniform"),
            regularization=kwargs.get("regularization", 1e-4),
            inference_k=kwargs.get("inference_k", 5),
        )

    def train(self, pair_set: ContrastivePairSet) -> LayerActivations:
        """Compatibility interface: returns mean transport vectors per layer."""
        result = self.train_przelom(pair_set)
        primary_map: RawActivationMap = {}
        for layer in result.layer_order:
            vec = result.mean_displacement[layer]
            vec = F.normalize(vec.float(), p=2, dim=-1)
            primary_map[layer] = vec
        dtype = self.kwargs.get("dtype", None)
        return LayerActivations(primary_map, dtype=dtype)

    def train_przelom(
        self, pair_set: ContrastivePairSet,
        qk_activations: Optional[Dict[LayerName, Tuple[torch.Tensor, torch.Tensor]]] = None,
        w_q_weights: Optional[Dict[LayerName, torch.Tensor]] = None,
    ) -> PrzelomResult:
        """Full PRZELOM training: EOT cost inversion for per-layer displacements.

        Args:
            pair_set: Contrastive pairs with activations.
            qk_activations: Maps layer -> (q_neg [N, d_k], k_pos [M, d_k]).
            w_q_weights: Maps layer -> W_q [d_model, d_k] projection matrix.
        """
        buckets = self._collect_from_set(pair_set)
        if not buckets:
            raise InsufficientDataError(reason="No valid activation pairs found")
        source_points: Dict[LayerName, torch.Tensor] = {}
        displacements_dict: Dict[LayerName, torch.Tensor] = {}
        mean_disp: Dict[LayerName, torch.Tensor] = {}
        layer_order: List[LayerName] = []
        layer_meta: Dict[str, Any] = {}
        eps = self.config.epsilon
        reg = self.config.regularization
        for layer_name in sorted(buckets.keys()):
            pos_list, neg_list = buckets[layer_name]
            if not pos_list or not neg_list:
                continue
            pos = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
            neg = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
            has_qk = qk_activations and layer_name in qk_activations
            has_wq = w_q_weights and layer_name in w_q_weights
            if has_qk:
                q_neg, k_pos = qk_activations[layer_name]
                q_neg, k_pos = q_neg.float(), k_pos.float()
                d_k = q_neg.shape[-1]
                scale = math.sqrt(d_k)
                C = -(q_neg @ k_pos.T) / scale
                T_current = torch.softmax(-C / eps, dim=1)
                T_target = _compute_target_transport(neg, pos, self.config.target_mode)
                log_T_target = torch.log(T_target.clamp(min=1e-12))
                log_T_current = torch.log(T_current.clamp(min=1e-12))
                delta_C = eps * (log_T_target - log_T_current)
                k_pos_pinv = _regularized_pinv(k_pos, reg)
                delta_q = -scale * (delta_C @ k_pos_pinv.T)
                if has_wq:
                    W_q = w_q_weights[layer_name].float()
                    W_q_pinv = _regularized_pinv(W_q, reg)
                    delta_h = delta_q @ W_q_pinv.T
                else:
                    delta_h = delta_q
            else:
                T_target = _compute_target_transport(neg, pos, self.config.target_mode)
                row_sums = T_target.sum(dim=1, keepdim=True).clamp(min=1e-12)
                T_norm = T_target / row_sums
                targets = T_norm @ pos
                delta_h = targets - neg
            source_points[layer_name] = neg.detach()
            displacements_dict[layer_name] = delta_h.detach()
            mean_disp[layer_name] = delta_h.mean(dim=0).detach()
            layer_order.append(layer_name)
            layer_meta[str(layer_name)] = {
                "n_neg": neg.shape[0], "n_pos": pos.shape[0],
                "mean_displacement_norm": delta_h.mean(dim=0).norm().item(),
                "used_qk_inversion": bool(has_qk),
            }
        if not layer_order:
            raise InsufficientDataError(reason="No layers produced valid transport")
        return PrzelomResult(
            source_points=source_points, displacements=displacements_dict,
            mean_displacement=mean_disp, config=self.config,
            layer_order=layer_order, metadata={"per_layer": layer_meta})

    def _collect_from_set(
        self, pair_set: ContrastivePairSet
    ) -> Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        buckets: Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]] = defaultdict(lambda: ([], []))
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
