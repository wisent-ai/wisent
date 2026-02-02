"""Component-level projection functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
from wisent.core.cli.cli_logger import setup_logger, bind
from wisent.core.errors import InvalidValueError

if TYPE_CHECKING:
    from torch import Tensor

_LOG = setup_logger(__name__)


def project_component_norm_preserved(
    weight_matrix: Tensor, steering_vector: Tensor, strength: float = 1.0, in_place: bool = True
) -> Tensor:
    """Project a weight matrix while PRESERVING ROW NORMS.

    This is the key innovation of the norm-preserving technique.
    Decomposes into direction and magnitude, projects only direction, then recombines.
    """
    log = bind(_LOG)
    original_dtype = weight_matrix.dtype
    device = weight_matrix.device

    with torch.no_grad():
        W = weight_matrix.float()
        v = steering_vector.to(device).float()
        if v.dim() > 1:
            v = v.view(-1)
        v = F.normalize(v, p=2, dim=0)
        out_dim, in_dim = W.shape
        if v.shape[0] != out_dim:
            raise InvalidValueError(
                param_name="steering_vector dimension",
                actual=v.shape[0],
                expected=f"weight matrix output dimension {out_dim}"
            )
        W_norm = torch.norm(W, p=2, dim=1, keepdim=True)
        W_direction = F.normalize(W, p=2, dim=1)
        v_row = v.unsqueeze(0)
        weighted_sum = v_row @ W_direction
        projection_term = v.unsqueeze(1) @ weighted_sum
        W_direction_new = W_direction - strength * projection_term
        W_direction_new = F.normalize(W_direction_new, p=2, dim=1)
        W_modified = W_norm * W_direction_new
        result = W_modified.to(original_dtype)
        if in_place:
            weight_matrix.copy_(result)
            result = weight_matrix
        projection_magnitude = (v.unsqueeze(1) * W_direction).sum(dim=1)
        log.debug("Norm-preserved projection", extra={
            "weight_shape": weight_matrix.shape, "strength": strength,
            "mean_projection": projection_magnitude.abs().mean().item()
        })
    return result


def project_component(
    weight_matrix: Tensor, projector: Tensor, strength: float, in_place: bool = True
) -> Tensor:
    """Standard directional projection (DOES NOT preserve norms - legacy method).

    WARNING: This method alters weight magnitudes, which can degrade model quality.
    Use project_component_norm_preserved() instead for better results.
    """
    log = bind(_LOG)
    device_projector = projector.to(weight_matrix.device)
    projected = device_projector @ weight_matrix
    if in_place:
        with torch.no_grad():
            weight_matrix.sub_(strength * projected)
        result = weight_matrix
    else:
        result = weight_matrix - strength * projected
    log.debug("Standard projection (norms NOT preserved)", extra={
        "weight_shape": weight_matrix.shape, "strength": strength
    })
    return result
