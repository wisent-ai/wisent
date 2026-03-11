"""
ZAPIS — KV Cache Steering via one-shot cache injection.

Training computes per-layer mean-of-differences vectors from
contrastive pair activations (same as CAA). At inference time
the vectors are injected into the KV cache after prefill,
enabling native use_cache=True generation with no per-token hooks.
"""

from __future__ import annotations

from typing import Any, Dict, List
from dataclasses import dataclass, field
from collections import defaultdict

import torch

from wisent.core.control.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.primitives.model_interface.core.activations.core.atoms import (
    LayerActivations,
    RawActivationMap,
    LayerName,
)
from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.utils.infra_tools.errors import InsufficientDataError
from wisent.core.utils.config_tools.constants import (
    ZAPIS_OFFSET_TOKEN_DEFAULT,
    LOG_EPS,
    AXIS_ROWS,
    AXIS_COLS,
    INDEX_LAST,
    NDIM_VECTOR,
)

__all__ = [
    "ZapisMethod",
    "ZapisConfig",
    "ZapisResult",
]


def _require(name: str, kwargs: dict):
    """Raise ValueError if a required hyperparameter is missing."""
    if name not in kwargs:
        raise ValueError(
            f"Parameter '{name}' is required. "
            f"Run 'wisent optimize-steering auto' first, or pass it explicitly."
        )
    return kwargs[name]


def _safe_l2_normalize(v: torch.Tensor, eps: float = LOG_EPS) -> torch.Tensor:
    """L2-normalise a vector with epsilon guard."""
    if v.ndim != NDIM_VECTOR:
        v = v.reshape(INDEX_LAST)
    return v / (torch.linalg.norm(v) + eps)


@dataclass
class ZapisConfig:
    """Configuration for ZAPIS KV cache steering."""

    c_keys: float
    """Coefficient for key cache injection."""
    c_values: float
    """Coefficient for value cache injection."""
    offset_token: str = ZAPIS_OFFSET_TOKEN_DEFAULT
    """Alignment token appended for position matching."""


@dataclass
class ZapisResult:
    """Result from ZAPIS training with per-layer steering vectors."""

    vectors: Dict[LayerName, torch.Tensor]
    """Per-layer steering vector [hidden_dim]."""
    config: ZapisConfig
    layer_order: List[LayerName]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ZapisMethod(BaseSteeringMethod):
    """
    ZAPIS steering method.

    Training computes per-layer mean-of-differences vectors.
    Application injects them into the KV cache once after prefill.
    """

    name = "zapis"
    description = (
        "ZAPIS — KV cache steering via one-shot cache injection. "
        "Modify keys/values once after prefill, then generate at native speed."
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = ZapisConfig(
            c_keys=_require("c_keys", kwargs),
            c_values=_require("c_values", kwargs),
            offset_token=kwargs.get("offset_token", ZAPIS_OFFSET_TOKEN_DEFAULT),
        )

    def train(self, pair_set: ContrastivePairSet) -> LayerActivations:
        """Compatibility interface: returns normalised mean-diff vectors."""
        result = self.train_zapis(pair_set)
        primary_map: RawActivationMap = {}
        for layer in result.layer_order:
            vec = result.vectors[layer]
            vec = _safe_l2_normalize(vec.float())
            primary_map[layer] = vec
        dtype = self.kwargs.get("dtype", None)
        return LayerActivations(primary_map, dtype=dtype)

    def train_zapis(self, pair_set: ContrastivePairSet) -> ZapisResult:
        """Full ZAPIS training: mean-of-differences per layer."""
        buckets = self._collect_from_set(pair_set)
        if not buckets:
            raise InsufficientDataError(reason="No valid activation pairs found")

        vectors: Dict[LayerName, torch.Tensor] = {}
        layer_order: List[LayerName] = []
        layer_meta: Dict[str, Any] = {}

        for layer_name in sorted(buckets.keys()):
            pos_list, neg_list = buckets[layer_name]
            if not pos_list or not neg_list:
                continue
            pos = torch.stack(
                [t.detach().float().reshape(INDEX_LAST) for t in pos_list],
                dim=AXIS_ROWS,
            )
            neg = torch.stack(
                [t.detach().float().reshape(INDEX_LAST) for t in neg_list],
                dim=AXIS_ROWS,
            )

            direction = pos.mean(dim=AXIS_ROWS) - neg.mean(dim=AXIS_ROWS)
            vectors[layer_name] = direction.detach()
            layer_order.append(layer_name)

            layer_meta[str(layer_name)] = {
                "n_pos": pos.shape[AXIS_ROWS],
                "n_neg": neg.shape[AXIS_ROWS],
                "direction_norm": direction.norm().item(),
            }

        if not layer_order:
            raise InsufficientDataError(
                reason="No layers produced valid steering vectors"
            )

        return ZapisResult(
            vectors=vectors,
            config=self.config,
            layer_order=layer_order,
            metadata={"per_layer": layer_meta},
        )

    def _collect_from_set(
        self, pair_set: ContrastivePairSet
    ) -> Dict[LayerName, tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Build {layer_name: ([pos tensors], [neg tensors])} from pairs."""
        buckets: Dict[
            LayerName, tuple[List[torch.Tensor], List[torch.Tensor]]
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
                    buckets[layer][AXIS_ROWS].append(p)
                    buckets[layer][AXIS_COLS].append(n)
        return buckets
