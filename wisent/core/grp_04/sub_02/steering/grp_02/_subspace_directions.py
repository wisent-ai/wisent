"""Auto direction computation and universal basis initialization."""
from __future__ import annotations
import os
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import torch
from wisent.core.cli.cli_logger import setup_logger, bind
from wisent.core.activations.core.atoms import LayerName
from wisent.core.steering._subspace_compression import UniversalBasis, compute_universal_basis
from wisent.core.constants import ZERO_THRESHOLD, DEFAULT_VARIANCE_THRESHOLD, TECZA_MAX_DIRECTIONS, MARGINAL_VARIANCE_THRESHOLD, SUBSPACE_ROBUSTNESS_NOISE_SCALE, MAX_PCA_COMPONENTS_ANALYSIS

VARIANCE_EXPLAINED_THRESHOLD = DEFAULT_VARIANCE_THRESHOLD

_LOG = setup_logger(__name__)

def explained_variance_analysis(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    max_components: int = MAX_PCA_COMPONENTS_ANALYSIS,
) -> Tuple[List[float], List[float]]:
    """
    Analyze how many directions are needed to explain the behavioral difference.
    
    Args:
        pos_activations: Positive example activations [N_pos, hidden_dim]
        neg_activations: Negative example activations [N_neg, hidden_dim]
        max_components: Maximum components to analyze
        
    Returns:
        Tuple of (individual_variance, cumulative_variance) lists
    """
    # Compute difference vectors
    n_min = min(pos_activations.shape[0], neg_activations.shape[0])
    diff = pos_activations[:n_min] - neg_activations[:n_min]
    
    # Center
    centered = diff - diff.mean(dim=0, keepdim=True)
    
    # SVD
    try:
        _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    except Exception:
        return [1.0], [1.0]
    
    # Variance ratios
    total_var = (S ** 2).sum()
    if total_var < ZERO_THRESHOLD:
        return [1.0], [1.0]
    
    k = min(max_components, len(S))
    individual = ((S[:k] ** 2) / total_var).tolist()
    cumulative = ((S[:k] ** 2).cumsum(0) / total_var).tolist()
    
    return individual, cumulative


def compute_optimal_num_directions(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    variance_threshold: float = VARIANCE_EXPLAINED_THRESHOLD,
    marginal_threshold: float = MARGINAL_VARIANCE_THRESHOLD,
    max_directions: int = TECZA_MAX_DIRECTIONS,
    min_directions: int = 1,
) -> Tuple[int, Dict[str, Any]]:
    """
    Compute optimal number of steering directions based on explained variance.
    
    Stops adding directions when:
    1. Cumulative variance exceeds threshold, OR
    2. Marginal variance of next direction < marginal_threshold
    
    Args:
        pos_activations: Positive example activations
        neg_activations: Negative example activations
        variance_threshold: Target cumulative variance
        marginal_threshold: Minimum marginal variance for new direction
        max_directions: Maximum directions to return
        min_directions: Minimum directions to return
        
    Returns:
        Tuple of (optimal_num_directions, analysis_details)
    """
    log = bind(_LOG)
    
    individual, cumulative = explained_variance_analysis(
        pos_activations, neg_activations, max_directions + 5
    )
    
    # Find optimal k
    optimal_k = min_directions
    
    for k in range(min_directions, min(max_directions + 1, len(cumulative))):
        # Check cumulative threshold
        if cumulative[k - 1] >= variance_threshold:
            optimal_k = k
            break
        
        # Check marginal threshold
        if k > 1 and individual[k - 1] < marginal_threshold:
            optimal_k = k - 1
            break
        
        optimal_k = k
    
    # Ensure minimum
    optimal_k = max(min_directions, optimal_k)
    
    details = {
        "individual_variance": individual[:optimal_k + 2],
        "cumulative_variance": cumulative[:optimal_k + 2],
        "variance_at_optimal": cumulative[optimal_k - 1] if optimal_k <= len(cumulative) else 1.0,
        "variance_threshold": variance_threshold,
        "marginal_threshold": marginal_threshold,
        "reason": (
            "reached_threshold" if cumulative[optimal_k - 1] >= variance_threshold
            else "marginal_too_small" if optimal_k < max_directions
            else "max_reached"
        ),
    }
    
    log.info(
        f"Optimal num_directions: {optimal_k}",
        extra=details,
    )
    
    return optimal_k, details


# =============================================================================
# 4. UNIVERSAL BASIS INITIALIZATION FOR TECZA/GROM
# =============================================================================

_CACHED_UNIVERSAL_BASIS: Dict[str, UniversalBasis] = {}


def get_cached_universal_basis(
    model_name: str,
    hidden_dim: int,
    cache_dir: Optional[str] = None,
) -> Optional[UniversalBasis]:
    """
    Get cached universal basis for a model architecture.
    
    Args:
        model_name: Model name/architecture identifier
        hidden_dim: Hidden dimension to match
        cache_dir: Directory to search for cached bases
        
    Returns:
        UniversalBasis if found, None otherwise
    """
    # Check in-memory cache
    cache_key = f"{model_name}_{hidden_dim}"
    if cache_key in _CACHED_UNIVERSAL_BASIS:
        return _CACHED_UNIVERSAL_BASIS[cache_key]
    
    # Check file cache
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.wisent/universal_bases")
    
    cache_path = Path(cache_dir) / f"{cache_key}.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
            basis = UniversalBasis.from_dict(data)
            _CACHED_UNIVERSAL_BASIS[cache_key] = basis
            return basis
        except Exception:
            pass
    
    return None


def initialize_from_universal_basis(
    hidden_dim: int,
    num_directions: int,
    basis: Optional[UniversalBasis] = None,
    model_name: Optional[str] = None,
    noise_scale: float = SUBSPACE_ROBUSTNESS_NOISE_SCALE,
) -> torch.Tensor:
    """
    Initialize steering directions from universal basis.
    
    If no basis available, falls back to random initialization.
    
    Args:
        hidden_dim: Hidden dimension
        num_directions: Number of directions to initialize
        basis: Optional pre-computed universal basis
        model_name: Model name for cached basis lookup
        noise_scale: Scale of noise to add for diversity
        
    Returns:
        Initialized directions tensor [num_directions, hidden_dim]
    """
    log = bind(_LOG)
    
    # Try to get basis
    if basis is None and model_name is not None:
        basis = get_cached_universal_basis(model_name, hidden_dim)
    
    if basis is not None and basis.hidden_dim == hidden_dim:
        # Initialize from basis components
        k = min(num_directions, basis.n_components)
        
        # Use top-k principal directions
        directions = basis.components[:k].clone()
        
        # Add remaining directions with noise if needed
        if k < num_directions:
            additional = torch.randn(num_directions - k, hidden_dim)
            additional = F.normalize(additional, p=2, dim=1)
            directions = torch.cat([directions, additional], dim=0)
        
        # Add small noise for diversity
        if noise_scale > 0:
            noise = torch.randn_like(directions) * noise_scale
            directions = F.normalize(directions + noise, p=2, dim=1)
        
        log.info(f"Initialized {num_directions} directions from universal basis")
        return directions
    
    # Fallback to random initialization
    log.info("No universal basis available, using random initialization")
    directions = torch.randn(num_directions, hidden_dim)
    return F.normalize(directions, p=2, dim=1)


# =============================================================================
# 5. NORM PRESERVATION VALIDATION
# =============================================================================
