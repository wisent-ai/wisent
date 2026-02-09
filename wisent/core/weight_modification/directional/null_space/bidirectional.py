"""Bidirectional projection with null-space constraint."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "bidirectional_projection_null_space",
]


def bidirectional_projection_null_space(
    weight_matrix: "Tensor",
    suppress_directions: List["Tensor"],
    enhance_directions: List["Tensor"],
    suppress_strengths: List[float],
    enhance_strengths: List[float],
    P_null: "Tensor",
    norm_preserve: bool = True,
) -> "Tensor":
    """Bidirectional projection with null-space constraint.

    Each delta goes through delta @ P_null before application,
    ensuring modifications don't affect preserved activations.

    Mathematical operations:
        Suppress: delta = (vv^T)W, delta' = delta @ P_null, W -= s*delta'
        Enhance:  delta = (uu^T)W, delta' = delta @ P_null, W += s*delta'
    """
    original_dtype = weight_matrix.dtype
    device = weight_matrix.device
    W = weight_matrix.float()
    P = P_null.to(device).float()

    with torch.no_grad():
        if norm_preserve:
            original_norms = torch.norm(W, p=2, dim=1, keepdim=True)
            W_direction = F.normalize(W, p=2, dim=1)

            for direction, s in zip(suppress_directions, suppress_strengths):
                v = F.normalize(direction.to(device).float(), p=2, dim=0)
                v_row = v.unsqueeze(0)
                weighted_sum = v_row @ W_direction
                delta = v.unsqueeze(1) @ weighted_sum
                delta = delta @ P
                W_direction = W_direction - s * delta

            for direction, s in zip(enhance_directions, enhance_strengths):
                u = F.normalize(direction.to(device).float(), p=2, dim=0)
                u_row = u.unsqueeze(0)
                weighted_sum = u_row @ W_direction
                delta = u.unsqueeze(1) @ weighted_sum
                delta = delta @ P
                W_direction = W_direction + s * delta

            W_direction = F.normalize(W_direction, p=2, dim=1)
            W = original_norms * W_direction

        else:
            for direction, s in zip(suppress_directions, suppress_strengths):
                v = F.normalize(direction.to(device).float(), p=2, dim=0)
                projector = torch.outer(v, v)
                delta = (projector @ W) @ P
                W = W - s * delta

            for direction, s in zip(enhance_directions, enhance_strengths):
                u = F.normalize(direction.to(device).float(), p=2, dim=0)
                projector = torch.outer(u, u)
                delta = (projector @ W) @ P
                W = W + s * delta

    return W.to(original_dtype)
