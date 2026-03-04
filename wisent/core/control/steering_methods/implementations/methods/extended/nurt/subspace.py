"""
Concept subspace discovery, projection, and reconstruction.

Provides the mathematical primitives for operating in the SVD-derived
concept subspace. The key property is that the orthogonal complement
is perfectly preserved during reconstruction, making Concept Flow
on-manifold by construction.
"""

from __future__ import annotations

from typing import Tuple

import torch
from wisent.core.utils.config_tools.constants import (
    LOG_EPS,
)


__all__ = [
    "discover_concept_subspace",
    "project_to_subspace",
    "reconstruct_from_subspace",
]


def discover_concept_subspace(
    pos: torch.Tensor,
    neg: torch.Tensor,
    variance_threshold: float,
    nurt_num_dims: int,
    nurt_max_concept_dim: int,
    min_concept_dim: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Discover the concept subspace via SVD on difference vectors.

    Computes D = pos - neg, then SVD(D) = U S Vh.
    Selects top-k rows of Vh where cumulative variance exceeds threshold.
    Uses nurt_num_dims for fixed dimensions when positive,
    otherwise auto-selects from eigenvalue spectrum.

    Args:
        pos: Positive activations [N, hidden_dim].
        neg: Negative activations [N, hidden_dim].
        variance_threshold: Cumulative variance threshold for auto-selection.
        nurt_num_dims: Number of concept subspace dimensions. Zero = auto.
        nurt_max_concept_dim: Maximum concept subspace dimensions.

    Returns:
        Vh: Concept basis matrix [k, hidden_dim].
        S: Singular values [min(N, hidden_dim)].
        k: Number of selected dimensions.
    """
    if min_concept_dim is None:
        raise ValueError("min_concept_dim is required")
    D = pos.float() - neg.float()

    # Center the difference vectors
    D_centered = D - D.mean(dim=0, keepdim=True)

    # SVD
    U, S, Vh = torch.linalg.svd(D_centered, full_matrices=False)

    if nurt_num_dims > 0:
        k = min(nurt_num_dims, S.shape[0], nurt_max_concept_dim)
        k = max(k, min_concept_dim)
    else:
        # Auto-select k from cumulative variance
        variance = S ** 2
        total_var = variance.sum()
        if total_var < LOG_EPS:
            k = min_concept_dim
        else:
            cumvar = torch.cumsum(variance, dim=0) / total_var
            # Find first index where cumulative variance exceeds threshold
            above = (cumvar >= variance_threshold).nonzero(as_tuple=False)
            if above.numel() > 0:
                k = above[0].item() + 1
            else:
                k = S.shape[0]
            # Clamp to [min_concept_dim, nurt_max_concept_dim]
            k = max(min_concept_dim, min(k, nurt_max_concept_dim))

    return Vh[:k], S, k


def project_to_subspace(
    activations: torch.Tensor,
    Vh: torch.Tensor,
) -> torch.Tensor:
    """
    Project activations into the concept subspace.

    Args:
        activations: Hidden states [N, hidden_dim] or [hidden_dim].
        Vh: Concept basis [k, hidden_dim].

    Returns:
        Projected activations [N, k] or [k].
    """
    squeeze = activations.dim() == 1
    if squeeze:
        activations = activations.unsqueeze(0)

    z = activations.float() @ Vh.T.float()

    if squeeze:
        z = z.squeeze(0)
    return z


def reconstruct_from_subspace(
    z_new: torch.Tensor,
    original: torch.Tensor,
    Vh: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct full hidden state from new subspace coordinates.

    Replaces the concept component while perfectly preserving
    the orthogonal complement:
        h' = z_new @ Vh + (original - z_old @ Vh)
           = z_new @ Vh + h_ortho

    Args:
        z_new: New coordinates in concept subspace [N, k] or [k].
        original: Original hidden states [N, hidden_dim] or [hidden_dim].
        Vh: Concept basis [k, hidden_dim].

    Returns:
        Reconstructed hidden states with same shape as original.
    """
    squeeze = z_new.dim() == 1
    if squeeze:
        z_new = z_new.unsqueeze(0)
        original = original.unsqueeze(0)

    Vh_f = Vh.float()
    orig_f = original.float()

    # Project original to get old subspace coordinates
    z_old = orig_f @ Vh_f.T

    # Orthogonal complement: everything not in the subspace
    h_ortho = orig_f - z_old @ Vh_f

    # Reconstruct: new concept component + preserved orthogonal complement
    h_new = z_new.float() @ Vh_f + h_ortho

    # Cast back to original dtype
    h_new = h_new.to(original.dtype)

    if squeeze:
        h_new = h_new.squeeze(0)
    return h_new
