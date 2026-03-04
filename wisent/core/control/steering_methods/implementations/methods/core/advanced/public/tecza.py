"""
TECZA - Projected Representations for Independent Steering Manifolds.

A gradient-optimized multi-directional steering method that discovers multiple
refusal directions per layer, forming a coherent steering manifold.

Based on insights from:
- "The Geometry of Refusal in Large Language Models" (Wollschlaeger et al., 2025)
- "SOM Directions are Better than One" (Piras et al., 2025)

Key innovations:
1. Gradient-based direction optimization (not just difference-in-means)
2. Multiple directions per layer that form a coherent manifold
3. Representational independence constraint (soft, not strict orthogonality)
4. Retain loss to minimize side effects on harmless queries
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from wisent.core.control.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.utils.infra_tools.errors import InsufficientDataError
from wisent.core.utils.infra_tools.errors import InsufficientDataError

__all__ = [
    "TECZAMethod",
    "TECZAConfig",
    "MultiDirectionResult",
]

from wisent.core.control.steering_methods.methods.advanced._tecza_types import (
    TECZAConfig, MultiDirectionResult,
)
from wisent.core.control.steering_methods.methods.advanced._tecza_training import TECZATrainingMixin
from wisent.core.control.steering_methods.methods.advanced._tecza_utils import TECZAUtilsMixin


def _require(name: str, kwargs: dict):
    """Raise ValueError if a required hyperparameter is missing."""
    if name not in kwargs:
        raise ValueError(
            f"Parameter '{name}' is required. "
            f"Run 'wisent optimize-steering auto' first, or pass it explicitly."
        )
    return kwargs[name]


class TECZAMethod(TECZATrainingMixin, TECZAUtilsMixin, BaseSteeringMethod):
    """
    TECZA - Projected Representations for Independent Steering Manifolds.

    Discovers multiple steering directions per layer using gradient-based
    optimization with representational independence constraints.

    Unlike CAA which computes a single direction via difference-in-means,
    TECZA finds k directions that:
    - Each mediate the target behavior (e.g., refusal)
    - Are related but not redundant (controlled cosine similarity)
    - Form a coherent manifold when ablated together
    - Minimize side effects via retain loss

    Usage:
        method = TECZAMethod(num_directions=3, optimization_steps=100)
        result = method.train(pair_set)
        # result.directions contains [num_directions, hidden_dim] per layer
    """

    name = "tecza"
    description = "Gradient-optimized multi-directional steering via projected representations"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        num_dirs = _require("num_directions", kwargs)
        auto_num = kwargs.get("auto_num_directions", False)
        if num_dirs == "auto":
            auto_num = True
            num_dirs = _require("num_directions", kwargs)
        self.log_interval: int = _require("log_interval", kwargs)
        self.config = TECZAConfig(
            num_directions=num_dirs,
            auto_num_directions=auto_num,
            variance_threshold=_require("variance_threshold", kwargs),
            marginal_threshold=_require("marginal_threshold", kwargs),
            max_directions=_require("max_directions", kwargs),
            optimization_steps=_require("optimization_steps", kwargs),
            learning_rate=_require("learning_rate", kwargs),
            retain_weight=_require("retain_weight", kwargs),
            independence_weight=_require("independence_weight", kwargs),
            ablation_weight=_require("ablation_weight", kwargs),
            addition_weight=_require("addition_weight", kwargs),
            separation_margin=_require("separation_margin", kwargs),
            perturbation_scale=_require("perturbation_scale", kwargs),
            universal_basis_noise=_require("universal_basis_noise", kwargs),
            normalize=kwargs.get("normalize", True),
            use_caa_init=kwargs.get("use_caa_init", True),
            use_universal_basis_init=kwargs.get("use_universal_basis_init", False),
            cone_constraint=kwargs.get("cone_constraint", True),
            min_cosine_similarity=_require("min_cosine_similarity", kwargs),
            max_cosine_similarity=_require("max_cosine_similarity", kwargs),
        )
        self._training_logs: List[Dict[str, float]] = []

    def train_for_layer(self, pos_list: List[torch.Tensor], neg_list: List[torch.Tensor]) -> torch.Tensor:
        """Train a TECZA steering vector for a single layer."""
        if not pos_list or not neg_list:
            raise InsufficientDataError(reason="Need both positive and negative activations")
        pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
        directions, _meta = self._train_layer_directions(pos_tensor, neg_tensor, "layer", log_interval=self.log_interval)
        primary = directions[0]
        if self.config.normalize:
            primary = F.normalize(primary, dim=-1)
        return primary

    def train(self, pair_set: ContrastivePairSet) -> LayerActivations:
        """Train TECZA directions, returning primary direction per layer."""
        multi_result = self.train_multi(pair_set)
        primary_map: RawActivationMap = multi_result.to_single_direction_map()
        dtype = self.kwargs.get("dtype", None)
        return LayerActivations(primary_map, dtype=dtype)

    def train_multi(self, pair_set: ContrastivePairSet) -> MultiDirectionResult:
        """Train multiple TECZA directions from contrastive pairs."""
        buckets = self._collect_from_set(pair_set)
        if not buckets:
            raise InsufficientDataError(reason="No valid activation pairs found in pair_set")
        all_directions: Dict[LayerName, torch.Tensor] = {}
        layer_metadata: Dict[LayerName, Dict[str, Any]] = {}
        for layer_name, (pos_list, neg_list) in sorted(buckets.items(), key=lambda kv: (len(kv[0]), kv[0])):
            if not pos_list or not neg_list:
                continue
            pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
            neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
            directions, meta = self._train_layer_directions(pos_tensor, neg_tensor, layer_name, log_interval=self.log_interval)
            all_directions[layer_name] = directions
            layer_metadata[layer_name] = meta
        return MultiDirectionResult(
            directions=all_directions,
            metadata={
                "config": self.config.__dict__,
                "num_layers": len(all_directions),
                "layer_metadata": layer_metadata,
                "training_logs": self._training_logs,
            }
        )
