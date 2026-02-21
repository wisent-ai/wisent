"""Subspace alignment validation and geometry thresholds."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import torch
from wisent.core.cli.cli_logger import setup_logger
from wisent.core.activations.core.atoms import LayerName

UNIVERSAL_SUBSPACE_THRESHOLDS = {
    "linear_variance_threshold": 0.85,
    "cone_threshold": 0.65,
    "manifold_threshold": 0.70,
    "cluster_silhouette_threshold": 0.55,
    "orthogonal_threshold": 0.12,
}

_LOG = setup_logger(__name__)

def compute_subspace_alignment(
    original_weights: torch.Tensor,
    modified_weights: torch.Tensor,
    n_components: int = UNIVERSAL_SUBSPACE_RANK,
) -> float:
    """
    Compute how well modified weights align with original's subspace.
    
    Args:
        original_weights: Original weight matrix
        modified_weights: Modified weight matrix
        n_components: Number of principal components to compare
        
    Returns:
        Alignment score (0-1, higher = better preservation)
    """
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
    threshold: float = 0.95,
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
    log = bind(_LOG)
    
    orig = original_weights.detach().float()
    mod = modified_weights.detach().float()
    
    # Compute various preservation metrics
    metrics = {}
    
    # 1. Row norm preservation
    orig_norms = orig.norm(dim=1)
    mod_norms = mod.norm(dim=1)
    norm_ratio = (mod_norms / (orig_norms + 1e-10)).mean().item()
    metrics["norm_ratio"] = norm_ratio
    metrics["norm_preserved"] = abs(norm_ratio - 1.0) < 0.05
    
    # 2. Subspace alignment
    alignment = compute_subspace_alignment(orig, mod)
    metrics["subspace_alignment"] = alignment
    
    # 3. Frobenius norm of difference (relative)
    diff_norm = (orig - mod).norm().item()
    orig_norm = orig.norm().item()
    relative_change = diff_norm / (orig_norm + 1e-10)
    metrics["relative_change"] = relative_change
    
    # 4. Spectral norm preservation
    orig_spectral = torch.linalg.svdvals(orig)[0].item()
    mod_spectral = torch.linalg.svdvals(mod)[0].item()
    spectral_ratio = mod_spectral / (orig_spectral + 1e-10)
    metrics["spectral_ratio"] = spectral_ratio
    
    # Overall preservation check
    is_preserved = (
        alignment >= threshold and
        abs(norm_ratio - 1.0) < 0.1 and
        abs(spectral_ratio - 1.0) < 0.2
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
) -> Dict[str, float]:
    """
    Get recommended geometry detection thresholds based on data characteristics.
    
    Based on Universal Subspace findings:
    - Linear structure is more common than previously assumed
    - True cone/manifold structures are rarer
    - Small sample sizes inflate apparent structure complexity
    
    Args:
        n_samples: Number of samples
        hidden_dim: Hidden dimension
        
    Returns:
        Dict of recommended thresholds
    """
    thresholds = UNIVERSAL_SUBSPACE_THRESHOLDS.copy()
    
    # Adjust for sample size
    # Small samples -> more conservative (raise thresholds)
    if n_samples < 20:
        thresholds["linear_variance_threshold"] = min(0.95, thresholds["linear_variance_threshold"] + 0.1)
        thresholds["cone_threshold"] = max(0.5, thresholds["cone_threshold"] - 0.1)
    elif n_samples > 100:
        thresholds["linear_variance_threshold"] = max(0.75, thresholds["linear_variance_threshold"] - 0.05)
    
    # Adjust for hidden dimension
    # Higher dim -> structure detection is harder
    if hidden_dim > 4096:
        thresholds["cluster_silhouette_threshold"] = max(0.4, thresholds["cluster_silhouette_threshold"] - 0.1)
    
    return thresholds
