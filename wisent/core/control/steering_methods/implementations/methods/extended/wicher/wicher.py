"""
WICHER — Whirlwind Iterative Cholesky-free Hessian-projected Embedding Refinement.

Training computes concept directions AND SVD subspace per layer:
1. w = mean(pos) - mean(neg)
2. SVD on uncentered differences -> U, S, Vh
3. Auto-select k dims via cumulative variance threshold
4. Store: w [D], Vh[:k] [k,D], S[:k]^2 [k] per layer

The Broyden iteration in the subspace happens at inference time
inside WicherSteeringObject.apply_steering().
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn.functional as F

from wisent.core.control.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.primitives.model_interface.core.activations.core.atoms import (
    LayerActivations,
    RawActivationMap,
    LayerName,
)
from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.utils.infra_tools.errors import InsufficientDataError
from wisent.core.utils.config_tools.constants import LOG_EPS

__all__ = [
    "WicherMethod",
    "WicherConfig",
    "WicherResult",
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
class WicherConfig:
    """Configuration for WICHER method."""

    concept_dim: int
    """Concept subspace dimensionality (zero = auto from variance)."""
    num_steps: int
    """Number of Broyden iterations per forward pass."""
    alpha: float
    """Base Tikhonov regularisation."""
    eta: float
    """Step-size multiplier per Broyden iteration."""
    beta: float
    """EMA momentum coefficient (zero = disabled)."""
    alpha_decay: float
    """Per-step decay factor for alpha."""
    variance_threshold: float
    """Cumulative variance threshold for auto dim selection."""


@dataclass
class WicherResult:
    """Result from WICHER training with per-layer concept subspace."""

    concept_directions: Dict[LayerName, torch.Tensor]
    """Per-layer concept direction [D]."""
    concept_bases: Dict[LayerName, torch.Tensor]
    """Per-layer SVD basis [k, D]."""
    component_variances: Dict[LayerName, torch.Tensor]
    """Per-layer singular values squared [k]."""
    layer_variance: Dict[LayerName, float]
    """Per-layer total SVD variance."""
    config: WicherConfig
    layer_order: List[LayerName]
    metadata: Dict[str, Any] = field(default_factory=dict)


class WicherMethod(BaseSteeringMethod):
    """
    WICHER steering method.

    Training computes direction + SVD subspace per layer.
    Broyden iteration in the subspace is deferred to inference time.
    """

    name = "wicher"
    description = (
        "WICHER — subspace-projected Broyden steering via "
        "low-rank SVD concept basis with adaptive regularization"
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = WicherConfig(
            concept_dim=_require("concept_dim", kwargs),
            num_steps=_require("num_steps", kwargs),
            alpha=_require("alpha", kwargs),
            eta=_require("eta", kwargs),
            beta=_require("beta", kwargs),
            alpha_decay=_require("alpha_decay", kwargs),
            variance_threshold=_require("variance_threshold", kwargs),
        )

    def train(self, pair_set: ContrastivePairSet) -> LayerActivations:
        """Compatibility interface: returns normalised mean-diff vectors."""
        result = self.train_wicher(pair_set)
        primary_map: RawActivationMap = {}
        for layer in result.layer_order:
            vec = result.concept_directions[layer]
            vec = F.normalize(vec.float(), p=2, dim=-1)
            primary_map[layer] = vec
        dtype = self.kwargs.get("dtype", None)
        return LayerActivations(primary_map, dtype=dtype)

    def train_wicher(self, pair_set: ContrastivePairSet) -> WicherResult:
        """Full WICHER training: direction + SVD subspace per layer."""
        buckets = self._collect_from_set(pair_set)
        if not buckets:
            raise InsufficientDataError(reason="No valid activation pairs found")

        concept_dirs: Dict[LayerName, torch.Tensor] = {}
        concept_bases: Dict[LayerName, torch.Tensor] = {}
        comp_variances: Dict[LayerName, torch.Tensor] = {}
        layer_order: List[LayerName] = []
        layer_meta: Dict[str, Any] = {}
        layer_variance: Dict[LayerName, float] = {}

        for layer_name in sorted(buckets.keys()):
            pos_list, neg_list = buckets[layer_name]
            if not pos_list or not neg_list:
                continue
            pos = torch.stack(
                [t.detach().float().reshape(-1) for t in pos_list], dim=0
            )
            neg = torch.stack(
                [t.detach().float().reshape(-1) for t in neg_list], dim=0
            )

            direction = pos.mean(dim=0) - neg.mean(dim=0)
            concept_dirs[layer_name] = direction.detach()

            diff = pos - neg
            _, S, Vh = torch.linalg.svd(diff, full_matrices=False)
            s_squared = S ** 2
            total_var = s_squared.sum().item()
            layer_variance[layer_name] = total_var

            k = self._select_concept_dim(
                s_squared, self.config.concept_dim, self.config.variance_threshold
            )
            concept_bases[layer_name] = Vh[:k].detach()
            comp_variances[layer_name] = s_squared[:k].detach()
            layer_order.append(layer_name)

            explained = s_squared[:k].sum().item() / max(total_var, LOG_EPS)
            layer_meta[str(layer_name)] = {
                "n_pos": pos.shape[0],
                "n_neg": neg.shape[0],
                "direction_norm": direction.norm().item(),
                "concept_dim": k,
                "variance_explained": explained,
                "svd_variance": total_var,
            }

        if not layer_order:
            raise InsufficientDataError(
                reason="No layers produced valid concept directions"
            )

        return WicherResult(
            concept_directions=concept_dirs,
            concept_bases=concept_bases,
            component_variances=comp_variances,
            layer_variance=layer_variance,
            config=self.config,
            layer_order=layer_order,
            metadata={"per_layer": layer_meta},
        )

    @staticmethod
    def _select_concept_dim(
        s_squared: torch.Tensor, explicit_dim: int, var_threshold: float,
        min_concept_dim: int = None, max_concept_dim: int = None,
    ) -> int:
        """Select concept subspace dimensionality k."""
        if min_concept_dim is None:
            raise ValueError("min_concept_dim is required")
        if max_concept_dim is None:
            raise ValueError("max_concept_dim is required")
        if explicit_dim > 0:
            return max(min_concept_dim, min(explicit_dim, max_concept_dim, len(s_squared)))
        total = s_squared.sum().item()
        if total < LOG_EPS:
            return 2
        cumvar = torch.cumsum(s_squared, dim=0) / total
        k = int((cumvar < var_threshold).sum().item()) + 1
        return max(min_concept_dim, min(k, max_concept_dim, len(s_squared)))

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
            layer_names = set(pos_la.to_dict().keys()) | set(
                neg_la.to_dict().keys()
            )
            for layer in layer_names:
                p = pos_la.to_dict().get(layer) if pos_la else None
                n = neg_la.to_dict().get(layer) if neg_la else None
                if isinstance(p, torch.Tensor) and isinstance(n, torch.Tensor):
                    buckets[layer][0].append(p)
                    buckets[layer][1].append(n)
        return buckets
