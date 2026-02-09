"""Null-space projector computation and preserved key management (AlphaEdit-style).

Computes P_null = I - V diag(S^2/(S^2+eps)) V^T via SVD.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING

from wisent.core.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "PreservedKeyMatrix",
    "compute_null_space_projector",
    "project_delta_into_null_space",
]

_LOG = setup_logger(__name__)


@dataclass
class PreservedKeyMatrix:
    """Per-layer preserved key matrices with cached null-space projectors.

    Accumulates preserved keys and lazily computes P_null projectors.
    Cache is invalidated whenever new keys are accumulated.
    """

    keys: Dict[int, "Tensor"] = field(default_factory=dict)
    """Per-layer preserved key matrices [n_keys, hidden_dim]."""

    _cached_projectors: Dict[int, "Tensor"] = field(default_factory=dict, repr=False)
    """Cached P_null per layer, invalidated on accumulation."""

    epsilon: float = 1e-6
    """Tikhonov regularization for numerical stability."""

    max_rank: Optional[int] = None
    """Optional SVD rank truncation."""

    def accumulate(self, new_keys: Dict[int, "Tensor"]) -> None:
        """Append new preserved keys and invalidate projector cache.

        Args:
            new_keys: Per-layer key tensors. Each can be
                      1D [hidden_dim] or 2D [n_keys, hidden_dim].
        """
        for layer_idx, key_tensor in new_keys.items():
            key_tensor = key_tensor.float()
            if key_tensor.dim() == 1:
                key_tensor = key_tensor.unsqueeze(0)

            if layer_idx in self.keys:
                self.keys[layer_idx] = torch.cat(
                    [self.keys[layer_idx], key_tensor], dim=0
                )
            else:
                self.keys[layer_idx] = key_tensor

            self._cached_projectors.pop(layer_idx, None)

    def get_projector(self, layer_idx: int) -> Optional["Tensor"]:
        """Compute or return cached P_null for a layer.

        Returns None if no preserved keys exist for this layer.
        """
        if layer_idx not in self.keys:
            return None

        if layer_idx not in self._cached_projectors:
            self._cached_projectors[layer_idx] = compute_null_space_projector(
                self.keys[layer_idx],
                epsilon=self.epsilon,
                max_rank=self.max_rank,
            )

        return self._cached_projectors[layer_idx]

    def summary(self) -> Dict[int, int]:
        """Returns {layer_idx: n_keys} dict."""
        return {layer_idx: keys.shape[0] for layer_idx, keys in self.keys.items()}


def compute_null_space_projector(
    key_matrix: "Tensor",
    epsilon: float = 1e-6,
    max_rank: Optional[int] = None,
) -> "Tensor":
    """Compute null-space projector via SVD (numerically stable).

    Given preserved key matrix K0 [n_keys, hidden_dim], computes:
        K0 = U S V^T   (SVD)
        scale = S^2 / (S^2 + epsilon)
        P_null = I - V diag(scale) V^T

    Args:
        key_matrix: Preserved keys [n_keys, hidden_dim]
        epsilon: Tikhonov regularization parameter
        max_rank: Optional rank truncation for SVD

    Returns:
        P_null [hidden_dim, hidden_dim] in float32
    """
    log = bind(_LOG)
    K0 = key_matrix.float()
    hidden_dim = K0.shape[1]

    if max_rank is not None and max_rank < min(K0.shape):
        U, S, V = torch.svd_lowrank(K0, q=max_rank)
    else:
        U, S, Vt = torch.linalg.svd(K0, full_matrices=False)
        V = Vt.T  # [hidden_dim, n_components]

    S_sq = S ** 2
    scale = S_sq / (S_sq + epsilon)

    V_scaled = V * scale.unsqueeze(0)
    P_null = torch.eye(hidden_dim, device=K0.device, dtype=torch.float32) - V_scaled @ V.T

    log.debug(
        "Computed null-space projector",
        extra={
            "n_keys": K0.shape[0],
            "hidden_dim": hidden_dim,
            "n_components": S.shape[0],
            "max_singular_value": S[0].item(),
            "min_singular_value": S[-1].item(),
            "epsilon": epsilon,
        },
    )

    return P_null


def project_delta_into_null_space(delta_W: "Tensor", P_null: "Tensor") -> "Tensor":
    """Project a weight delta into the null space of preserved keys.

    Args:
        delta_W: Weight modification [out_dim, in_dim]
        P_null: Null-space projector [in_dim, in_dim]

    Returns:
        Constrained delta [out_dim, in_dim]
    """
    return delta_W @ P_null
