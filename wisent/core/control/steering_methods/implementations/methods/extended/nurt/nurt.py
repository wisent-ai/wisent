"""
Concept Flow - Flow Matching in Concept Subspace.

Learns on-manifold transport between contrastive activation distributions
using conditional flow matching in Zwiad's SVD-derived concept subspace.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn.functional as F

from wisent.core.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.contrastive_pairs.set import ContrastivePairSet
from wisent.core.errors import InsufficientDataError

from wisent.core.constants import (
    DEFAULT_VARIANCE_THRESHOLD,
    NURT_NUM_DIMS,
    NURT_TRAINING_EPOCHS,
    MLP_LEARNING_RATE,
    NURT_LR_MIN,
    NURT_NUM_INTEGRATION_STEPS,
    NURT_T_MAX,
    DEFAULT_WEIGHT_DECAY,
    GROM_MAX_GRAD_NORM,
)
from .flow_network import FlowVelocityNetwork
from .subspace import discover_concept_subspace, project_to_subspace

__all__ = [
    "NurtMethod",
    "NurtConfig",
    "NurtResult",
]


@dataclass
class NurtConfig:
    """Configuration for Concept Flow steering method."""
    num_dims: int = NURT_NUM_DIMS
    """Number of concept subspace dimensions. 0 = auto from variance threshold."""
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD
    """Cumulative variance threshold for auto dimension selection."""
    training_epochs: int = NURT_TRAINING_EPOCHS
    """Number of training epochs for flow matching."""
    lr: float = MLP_LEARNING_RATE
    """Learning rate for AdamW optimizer."""
    lr_min: float = NURT_LR_MIN
    """Minimum learning rate for cosine annealing."""
    num_integration_steps: int = NURT_NUM_INTEGRATION_STEPS
    """Number of Euler integration steps at inference."""
    t_max: float = NURT_T_MAX
    """Integration endpoint (t_max * strength = effective integration range)."""
    flow_hidden_dim: Optional[int] = None
    """Hidden dimension for velocity network. None = auto from concept_dim."""


@dataclass
class NurtResult:
    """Result from Concept Flow training containing all per-layer components."""
    flow_networks: Dict[LayerName, FlowVelocityNetwork]
    concept_bases: Dict[LayerName, torch.Tensor]
    mean_neg: Dict[LayerName, torch.Tensor]
    mean_pos: Dict[LayerName, torch.Tensor]
    singular_values: Dict[LayerName, torch.Tensor]
    config: NurtConfig
    layer_order: List[LayerName]
    metadata: Dict[str, Any] = field(default_factory=dict)


class NurtMethod(BaseSteeringMethod):
    """
    Concept Flow - Flow Matching in Concept Subspace.

    Learns per-layer velocity fields that transport activations from
    negative to positive distributions in a low-dimensional SVD subspace.
    """

    name = "nurt"
    description = (
        "Flow matching in SVD-derived concept subspace - "
        "learns on-manifold transport between contrastive distributions"
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = NurtConfig(
            num_dims=kwargs.get("num_dims", NURT_NUM_DIMS),
            variance_threshold=kwargs.get("variance_threshold", DEFAULT_VARIANCE_THRESHOLD),
            training_epochs=kwargs.get("training_epochs", NURT_TRAINING_EPOCHS),
            lr=kwargs.get("lr", MLP_LEARNING_RATE),
            lr_min=kwargs.get("lr_min", NURT_LR_MIN),
            num_integration_steps=kwargs.get("num_integration_steps", NURT_NUM_INTEGRATION_STEPS),
            t_max=kwargs.get("t_max", NURT_T_MAX),
            flow_hidden_dim=kwargs.get("flow_hidden_dim", None),
        )

    def train(self, pair_set: ContrastivePairSet) -> LayerActivations:
        """Compatibility interface: returns mean transport vectors per layer."""
        result = self.train_flow(pair_set)
        primary_map: RawActivationMap = {}
        for layer in result.layer_order:
            Vh = result.concept_bases[layer]
            diff = result.mean_pos[layer] - result.mean_neg[layer]
            transport_vec = diff.float() @ Vh.float()
            transport_vec = F.normalize(transport_vec, p=2, dim=-1)
            primary_map[layer] = transport_vec
        dtype = self.kwargs.get("dtype", None)
        return LayerActivations(primary_map, dtype=dtype)

    def train_flow(self, pair_set: ContrastivePairSet) -> NurtResult:
        """Full Concept Flow training with per-layer flow networks."""
        buckets = self._collect_from_set(pair_set)
        if not buckets:
            raise InsufficientDataError(reason="No valid activation pairs found")

        flow_networks: Dict[LayerName, FlowVelocityNetwork] = {}
        concept_bases: Dict[LayerName, torch.Tensor] = {}
        mean_neg_dict: Dict[LayerName, torch.Tensor] = {}
        mean_pos_dict: Dict[LayerName, torch.Tensor] = {}
        singular_values: Dict[LayerName, torch.Tensor] = {}
        layer_order: List[LayerName] = []
        layer_meta: Dict[str, Any] = {}

        for layer_name in sorted(buckets.keys()):
            pos_list, neg_list = buckets[layer_name]
            if not pos_list or not neg_list:
                continue
            pos = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
            neg = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)

            # 1. Discover concept subspace
            Vh, S, k = discover_concept_subspace(
                pos, neg,
                num_dims=self.config.num_dims,
                variance_threshold=self.config.variance_threshold,
            )
            # 2. Project to subspace
            z_pos = project_to_subspace(pos, Vh)
            z_neg = project_to_subspace(neg, Vh)
            # 3. Train velocity network
            network = self._train_flow_network(z_pos, z_neg, k)

            flow_networks[layer_name] = network
            concept_bases[layer_name] = Vh.detach()
            mean_neg_dict[layer_name] = z_neg.mean(dim=0).detach()
            mean_pos_dict[layer_name] = z_pos.mean(dim=0).detach()
            singular_values[layer_name] = S.detach()
            layer_order.append(layer_name)
            var_exp = ((S[:k] ** 2).sum() / (S ** 2).sum()).item() if S.sum() > 0 else 0.0
            layer_meta[str(layer_name)] = {
                "concept_dim": k, "n_samples": pos.shape[0],
                "variance_explained": var_exp,
            }

        if not layer_order:
            raise InsufficientDataError(reason="No layers produced valid flow networks")

        return NurtResult(
            flow_networks=flow_networks,
            concept_bases=concept_bases,
            mean_neg=mean_neg_dict,
            mean_pos=mean_pos_dict,
            singular_values=singular_values,
            config=self.config,
            layer_order=layer_order,
            metadata={"per_layer": layer_meta},
        )

    def _train_flow_network(
        self, z_pos: torch.Tensor, z_neg: torch.Tensor, concept_dim: int,
    ) -> FlowVelocityNetwork:
        """Train a single flow velocity network via conditional flow matching."""
        network = FlowVelocityNetwork(concept_dim, self.config.flow_hidden_dim)
        optimizer = torch.optim.AdamW(
            network.parameters(), lr=self.config.lr, weight_decay=DEFAULT_WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.training_epochs, eta_min=self.config.lr_min,
        )
        N = min(z_pos.shape[0], z_neg.shape[0])
        z_pos_train = z_pos[:N]
        z_neg_train = z_neg[:N]
        target = z_pos_train - z_neg_train

        network.train()
        for epoch in range(self.config.training_epochs):
            optimizer.zero_grad()
            t = torch.rand(N, 1, device=z_pos.device, dtype=z_pos.dtype)
            z_t = (1.0 - t) * z_neg_train + t * z_pos_train
            v_pred = network(z_t, t.squeeze(-1))
            loss = F.mse_loss(v_pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=GROM_MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

        network.eval()
        return network

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
