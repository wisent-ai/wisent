"""Subspace alignment validation and geometry thresholds."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import torch
from wisent.core.utils.cli.cli_logger import setup_logger
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerName
from wisent.core import constants as _C


def _build_universal_subspace_thresholds(
    geo_diag_linear_variance=None,
    geo_diag_cone_threshold=None,
    geo_diag_manifold_threshold=None,
    geo_diag_cluster_silhouette=None,
    geo_diag_orthogonal_threshold=None,
):
    """Build thresholds dict - all params required."""
    if geo_diag_linear_variance is None:
        raise ValueError("geo_diag_linear_variance is required")
    if geo_diag_cone_threshold is None:
        raise ValueError("geo_diag_cone_threshold is required")
    if geo_diag_manifold_threshold is None:
        raise ValueError("geo_diag_manifold_threshold is required")
    if geo_diag_cluster_silhouette is None:
        raise ValueError("geo_diag_cluster_silhouette is required")
    if geo_diag_orthogonal_threshold is None:
        raise ValueError("geo_diag_orthogonal_threshold is required")
    return {
        "linear_variance_threshold": geo_diag_linear_variance,
        "cone_threshold": geo_diag_cone_threshold,
        "manifold_threshold": geo_diag_manifold_threshold,
        "cluster_silhouette_threshold": geo_diag_cluster_silhouette,
        "orthogonal_threshold": geo_diag_orthogonal_threshold,
    }

_LOG = setup_logger(__name__)

def compute_subspace_alignment(
    original_weights: torch.Tensor,
    modified_weights: torch.Tensor,
    n_components: Optional[int] = None,
) -> float:
    """
    Compute how well modified weights align with original's subspace.

    Args:
        original_weights: Original weight matrix
        modified_weights: Modified weight matrix
        n_components: Number of principal components to compare

    Returns:
        Alignment score (higher = better preservation)
    """
    if n_components is None:
        raise ValueError("n_components is required")
    orig = original_weights.detach().float()
    mod = modified_weights.detach().float()
    
    # Compute principal subspaces
    _, _, Vh_orig = torch.linalg.svd(orig, full_matrices=False)
    _, _, Vh_mod = torch.linalg.svd(mod, full_matrices=False)
    
    k = min(n_components, Vh_orig.shape[0], Vh_mod.shape[0])
    
    # Subspace alignment: how much do the principal directions overlap?
    # Compute principal angles via SVD of V1^T @ V2
    V1 = Vh_orig[:k]
    V2 = Vh_mod[:k]
    
    _, S, _ = torch.linalg.svd(V1 @ V2.T, full_matrices=False)
    
    # Singular values are cosines of principal angles
    # Perfect alignment = all 1s
    alignment = S.mean().item()
    
    return alignment


def verify_subspace_preservation(
    original_weights: torch.Tensor,
    modified_weights: torch.Tensor,
    threshold: Optional[float] = None,
    subspace_norm_tolerance: float = None,
    subspace_spectral_ratio_low: float = None,
    subspace_spectral_ratio_high: float = None,
) -> Tuple[bool, Dict[str, float]]:
    """
    Verify that weight modification preserved subspace membership.
    
    Based on the Universal Subspace Hypothesis, good modifications should
    keep weights within the same low-dimensional subspace.
    
    Args:
        original_weights: Original weight matrix
        modified_weights: Modified weight matrix  
        threshold: Minimum alignment score for preservation
        
    Returns:
        Tuple of (is_preserved, metrics_dict)
    """
    from wisent.core.utils.config_tools.constants import (
        ZERO_THRESHOLD, COMBO_OFFSET,
    )
    if subspace_norm_tolerance is None:
        raise ValueError("subspace_norm_tolerance is required")
    if subspace_spectral_ratio_low is None:
        raise ValueError("subspace_spectral_ratio_low is required")
    if subspace_spectral_ratio_high is None:
        raise ValueError("subspace_spectral_ratio_high is required")
    log = bind(_LOG)
    if threshold is None:
        raise ValueError("threshold is required")
    orig = original_weights.detach().float()
    mod = modified_weights.detach().float()
    # Compute various preservation metrics
    metrics = {}

    # Row norm preservation
    orig_norms = orig.norm(dim=COMBO_OFFSET)
    mod_norms = mod.norm(dim=COMBO_OFFSET)
    norm_ratio = (mod_norms / (orig_norms + ZERO_THRESHOLD)).mean().item()
    metrics["norm_ratio"] = norm_ratio
    metrics["norm_preserved"] = abs(norm_ratio - 1.0) < subspace_norm_tolerance
    
    # 2. Subspace alignment
    alignment = compute_subspace_alignment(orig, mod)
    metrics["subspace_alignment"] = alignment
    
    # 3. Frobenius norm of difference (relative)
    diff_norm = (orig - mod).norm().item()
    orig_norm = orig.norm().item()
    relative_change = diff_norm / (orig_norm + ZERO_THRESHOLD)
    metrics["relative_change"] = relative_change
    
    # 4. Spectral norm preservation
    orig_spectral = torch.linalg.svdvals(orig)[0].item()
    mod_spectral = torch.linalg.svdvals(mod)[0].item()
    spectral_ratio = mod_spectral / (orig_spectral + ZERO_THRESHOLD)
    metrics["spectral_ratio"] = spectral_ratio
    
    # Overall preservation check
    is_preserved = (
        alignment >= threshold and
        abs(norm_ratio - 1.0) < subspace_spectral_ratio_low and
        abs(spectral_ratio - 1.0) < subspace_spectral_ratio_high
    )
    metrics["is_preserved"] = is_preserved
    
    log.info(
        "Subspace preservation check",
        extra=metrics,
    )
    
    return is_preserved, metrics


# =============================================================================
# 6. GEOMETRY DETECTION THRESHOLD TUNING
# =============================================================================

def get_recommended_geometry_thresholds(
    n_samples: int,
    hidden_dim: int,
    subspace_sample_conservative: int = None,
    subspace_sample_relaxed: int = None,
    subspace_var_threshold_max: float = None,
    subspace_var_threshold_adjust: float = None,
    subspace_cone_min_dynamic: float = None,
    subspace_cone_adjust: float = None,
    subspace_var_threshold_min: float = None,
    subspace_var_fine_adjust: float = None,
    subspace_hidden_dim_large: int = None,
    subspace_silhouette_min_dynamic: float = None,
    subspace_silhouette_adjust: float = None,
    base_thresholds: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Get recommended geometry detection thresholds based on data characteristics.

    All threshold parameters are required.
    """
    if subspace_sample_conservative is None:
        raise ValueError("subspace_sample_conservative is required")
    if subspace_sample_relaxed is None:
        raise ValueError("subspace_sample_relaxed is required")
    if subspace_var_threshold_max is None:
        raise ValueError("subspace_var_threshold_max is required")
    if subspace_var_threshold_adjust is None:
        raise ValueError("subspace_var_threshold_adjust is required")
    if subspace_cone_min_dynamic is None:
        raise ValueError("subspace_cone_min_dynamic is required")
    if subspace_cone_adjust is None:
        raise ValueError("subspace_cone_adjust is required")
    if subspace_var_threshold_min is None:
        raise ValueError("subspace_var_threshold_min is required")
    if subspace_var_fine_adjust is None:
        raise ValueError("subspace_var_fine_adjust is required")
    if subspace_hidden_dim_large is None:
        raise ValueError("subspace_hidden_dim_large is required")
    if subspace_silhouette_min_dynamic is None:
        raise ValueError("subspace_silhouette_min_dynamic is required")
    if subspace_silhouette_adjust is None:
        raise ValueError("subspace_silhouette_adjust is required")
    if base_thresholds is None:
        raise ValueError("base_thresholds is required")
    thresholds = dict(base_thresholds)

    if n_samples < subspace_sample_conservative:
        thresholds["linear_variance_threshold"] = min(
            subspace_var_threshold_max,
            thresholds["linear_variance_threshold"] + subspace_var_threshold_adjust,
        )
        thresholds["cone_threshold"] = max(
            subspace_cone_min_dynamic,
            thresholds["cone_threshold"] - subspace_cone_adjust,
        )
    elif n_samples > subspace_sample_relaxed:
        thresholds["linear_variance_threshold"] = max(
            subspace_var_threshold_min,
            thresholds["linear_variance_threshold"] - subspace_var_fine_adjust,
        )

    if hidden_dim > subspace_hidden_dim_large:
        thresholds["cluster_silhouette_threshold"] = max(
            subspace_silhouette_min_dynamic,
            thresholds["cluster_silhouette_threshold"] - subspace_silhouette_adjust,
        )

    return thresholds
