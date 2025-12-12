"""
Universal Subspace Analysis for Steering Vectors.

Based on "The Universal Weight Subspace Hypothesis" (Kaushik et al., 2025),
which shows that neural networks converge to shared low-dimensional subspaces.

This module provides:
1. Spectral analysis to check if steering vectors lie in a low-rank subspace
2. Steering vector compression via subspace coefficients
3. Auto num_directions based on explained variance
4. Universal basis initialization for PRISM/TITAN
5. Norm preservation validation for weight modifications
6. Geometry detection threshold tuning

Key insight: Steering vectors discovered by CAA/PRISM/TITAN are components
of the universal subspace that models naturally learn.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn.functional as F

from wisent.core.cli_logger import setup_logger, bind
from wisent.core.activations.core.atoms import LayerActivations, LayerName

__all__ = [
    # Spectral Analysis
    "SubspaceAnalysisConfig",
    "SubspaceAnalysisResult",
    "analyze_steering_vector_subspace",
    "check_vector_quality",
    # Compression
    "UniversalBasis",
    "compress_steering_vectors",
    "decompress_steering_vectors",
    "save_compressed_vectors",
    "load_compressed_vectors",
    # Auto Directions
    "compute_optimal_num_directions",
    "explained_variance_analysis",
    # Universal Basis
    "compute_universal_basis",
    "initialize_from_universal_basis",
    "get_cached_universal_basis",
    # Norm Preservation
    "verify_subspace_preservation",
    "compute_subspace_alignment",
    # Geometry Thresholds
    "UNIVERSAL_SUBSPACE_THRESHOLDS",
    "get_recommended_geometry_thresholds",
]

_LOG = setup_logger(__name__)


# =============================================================================
# CONSTANTS BASED ON PAPER FINDINGS
# =============================================================================

# Paper finding: ~16 principal directions capture majority variance across 500+ models
UNIVERSAL_SUBSPACE_RANK = 16

# Paper finding: 80%+ variance explained indicates good subspace membership
VARIANCE_EXPLAINED_THRESHOLD = 0.80

# Marginal variance threshold for auto num_directions
MARGINAL_VARIANCE_THRESHOLD = 0.05

# Updated geometry thresholds based on universal subspace theory
UNIVERSAL_SUBSPACE_THRESHOLDS = {
    "linear_variance_threshold": 0.85,  # Higher - linear is more common than expected
    "cone_threshold": 0.65,  # Lower - true cones are rarer
    "manifold_threshold": 0.70,
    "cluster_silhouette_threshold": 0.55,
    "orthogonal_threshold": 0.12,  # Stricter - orthogonal is rare in universal subspace
}


# =============================================================================
# 1. SPECTRAL ANALYSIS FOR STEERING VECTOR QUALITY
# =============================================================================

@dataclass
class SubspaceAnalysisConfig:
    """Configuration for subspace analysis."""
    
    n_components: int = UNIVERSAL_SUBSPACE_RANK
    """Number of principal components to analyze."""
    
    variance_threshold: float = VARIANCE_EXPLAINED_THRESHOLD
    """Variance explained threshold for quality check."""
    
    min_vectors: int = 3
    """Minimum vectors needed for meaningful analysis."""
    
    normalize_vectors: bool = True
    """Whether to normalize vectors before analysis."""


@dataclass
class SubspaceAnalysisResult:
    """Results from subspace analysis."""
    
    lies_in_subspace: bool
    """Whether vectors lie in a low-rank subspace."""
    
    variance_explained: float
    """Fraction of variance explained by top-k components."""
    
    effective_rank: int
    """Number of components needed for threshold variance."""
    
    singular_values: List[float]
    """Singular values from SVD."""
    
    principal_directions: Optional[torch.Tensor]
    """Top principal directions [n_components, hidden_dim]."""
    
    quality_score: float
    """Overall quality score (0-1) based on subspace membership."""
    
    details: Dict[str, Any] = field(default_factory=dict)
    """Additional analysis details."""
    
    def summary(self) -> str:
        """Return a summary string."""
        status = "✓ GOOD" if self.lies_in_subspace else "⚠ WARNING"
        return (
            f"Subspace Analysis: {status}\n"
            f"  Variance explained (top-{len(self.singular_values[:UNIVERSAL_SUBSPACE_RANK])}): {self.variance_explained:.1%}\n"
            f"  Effective rank: {self.effective_rank}\n"
            f"  Quality score: {self.quality_score:.2f}"
        )


def analyze_steering_vector_subspace(
    vectors: List[torch.Tensor] | Dict[LayerName, torch.Tensor] | torch.Tensor,
    config: SubspaceAnalysisConfig | None = None,
) -> SubspaceAnalysisResult:
    """
    Analyze whether steering vectors lie in a low-rank subspace.
    
    Based on the Universal Subspace Hypothesis, high-quality steering vectors
    should lie within the same low-dimensional subspace that models learn.
    
    Args:
        vectors: Steering vectors to analyze. Can be:
            - List of tensors [N, hidden_dim]
            - Dict mapping layer names to tensors
            - Single tensor [N, hidden_dim]
        config: Analysis configuration
        
    Returns:
        SubspaceAnalysisResult with quality metrics
    """
    log = bind(_LOG)
    cfg = config or SubspaceAnalysisConfig()
    
    # Convert to tensor matrix
    if isinstance(vectors, dict):
        vector_list = [v.detach().float().reshape(-1) for v in vectors.values() if v is not None]
    elif isinstance(vectors, list):
        vector_list = [v.detach().float().reshape(-1) for v in vectors]
    else:
        vector_list = [vectors[i].detach().float().reshape(-1) for i in range(vectors.shape[0])]
    
    if len(vector_list) < cfg.min_vectors:
        log.warning(f"Only {len(vector_list)} vectors provided, need at least {cfg.min_vectors}")
        return SubspaceAnalysisResult(
            lies_in_subspace=False,
            variance_explained=0.0,
            effective_rank=len(vector_list),
            singular_values=[],
            principal_directions=None,
            quality_score=0.0,
            details={"reason": "insufficient_vectors"}
        )
    
    # Stack and optionally normalize
    matrix = torch.stack(vector_list, dim=0)  # [N, hidden_dim]
    
    if cfg.normalize_vectors:
        matrix = F.normalize(matrix, p=2, dim=1)
    
    # Center the data
    mean = matrix.mean(dim=0, keepdim=True)
    centered = matrix - mean
    
    # SVD for spectral analysis
    try:
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    except Exception as e:
        log.error(f"SVD failed: {e}")
        return SubspaceAnalysisResult(
            lies_in_subspace=False,
            variance_explained=0.0,
            effective_rank=len(vector_list),
            singular_values=[],
            principal_directions=None,
            quality_score=0.0,
            details={"error": str(e)}
        )
    
    # Compute variance explained
    total_var = (S ** 2).sum()
    if total_var < 1e-10:
        return SubspaceAnalysisResult(
            lies_in_subspace=True,
            variance_explained=1.0,
            effective_rank=1,
            singular_values=[0.0],
            principal_directions=None,
            quality_score=1.0,
            details={"reason": "zero_variance"}
        )
    
    cumulative_var = (S ** 2).cumsum(0) / total_var
    
    # Find effective rank (components needed for threshold)
    k = min(cfg.n_components, len(S))
    variance_explained_k = float(cumulative_var[k - 1]) if k > 0 else 0.0
    
    # Effective rank: smallest k where cumulative variance >= threshold
    effective_rank = int((cumulative_var >= cfg.variance_threshold).float().argmax().item()) + 1
    if cumulative_var[-1] < cfg.variance_threshold:
        effective_rank = len(S)
    
    # Quality score based on:
    # 1. How much variance top-k explains (should be high)
    # 2. How quickly variance concentrates (should be in few components)
    # 3. Effective rank relative to universal subspace rank
    
    concentration_score = variance_explained_k
    rank_score = max(0, 1 - (effective_rank - 1) / UNIVERSAL_SUBSPACE_RANK)
    
    # Spectral decay rate (faster decay = better)
    if len(S) > 1:
        decay_rate = (S[0] / (S[1] + 1e-10)).item()
        decay_score = min(decay_rate / 10, 1.0)  # Normalize
    else:
        decay_score = 1.0
    
    quality_score = 0.5 * concentration_score + 0.3 * rank_score + 0.2 * decay_score
    
    lies_in_subspace = variance_explained_k >= cfg.variance_threshold
    
    log.info(
        "Subspace analysis complete",
        extra={
            "n_vectors": len(vector_list),
            "variance_explained": variance_explained_k,
            "effective_rank": effective_rank,
            "quality_score": quality_score,
            "lies_in_subspace": lies_in_subspace,
        }
    )
    
    return SubspaceAnalysisResult(
        lies_in_subspace=lies_in_subspace,
        variance_explained=variance_explained_k,
        effective_rank=effective_rank,
        singular_values=S.tolist(),
        principal_directions=Vh[:k] if Vh is not None else None,
        quality_score=quality_score,
        details={
            "n_vectors": len(vector_list),
            "hidden_dim": matrix.shape[1],
            "concentration_score": concentration_score,
            "rank_score": rank_score,
            "decay_score": decay_score,
            "cumulative_variance": cumulative_var.tolist()[:min(20, len(cumulative_var))],
        }
    )


def check_vector_quality(
    vector: torch.Tensor,
    reference_vectors: Optional[List[torch.Tensor]] = None,
    threshold: float = VARIANCE_EXPLAINED_THRESHOLD,
) -> Tuple[bool, float, str]:
    """
    Quick check if a single steering vector is high quality.
    
    Args:
        vector: Single steering vector to check
        reference_vectors: Optional reference vectors to compare against
        threshold: Quality threshold
        
    Returns:
        Tuple of (is_good, quality_score, message)
    """
    # Basic sanity checks
    if vector is None or vector.numel() == 0:
        return False, 0.0, "Vector is empty or None"
    
    norm = vector.float().norm().item()
    if norm < 1e-8:
        return False, 0.0, "Vector has near-zero norm"
    
    # Check for NaN/Inf
    if not torch.isfinite(vector).all():
        return False, 0.0, "Vector contains NaN or Inf values"
    
    # If reference vectors provided, check alignment
    if reference_vectors and len(reference_vectors) >= 3:
        all_vectors = reference_vectors + [vector]
        result = analyze_steering_vector_subspace(all_vectors)
        
        if result.quality_score >= threshold:
            return True, result.quality_score, f"Vector aligns well with subspace (score={result.quality_score:.2f})"
        else:
            return False, result.quality_score, f"Vector may not lie in expected subspace (score={result.quality_score:.2f})"
    
    # Without reference, do basic quality checks
    # Check sparsity (too sparse = suspicious)
    sparsity = (vector.abs() < 1e-6).float().mean().item()
    if sparsity > 0.99:
        return False, 0.1, f"Vector is too sparse ({sparsity:.1%} zeros)"
    
    # Check concentration (variance should be spread reasonably)
    sorted_abs = vector.abs().sort(descending=True).values
    top_10_contribution = sorted_abs[:max(1, len(sorted_abs)//10)].sum() / (sorted_abs.sum() + 1e-10)
    
    if top_10_contribution > 0.99:
        return False, 0.3, f"Vector is too concentrated (top 10% = {top_10_contribution:.1%})"
    
    return True, 0.8, "Vector passes basic quality checks"


# =============================================================================
# 2. STEERING VECTOR COMPRESSION
# =============================================================================

@dataclass
class UniversalBasis:
    """Universal basis for steering vector compression."""
    
    components: torch.Tensor
    """Principal components [n_components, hidden_dim]."""
    
    mean: torch.Tensor
    """Mean vector used for centering [hidden_dim]."""
    
    singular_values: torch.Tensor
    """Singular values for each component."""
    
    explained_variance_ratio: torch.Tensor
    """Variance explained by each component."""
    
    n_components: int
    """Number of components."""
    
    hidden_dim: int
    """Hidden dimension of original vectors."""
    
    source_info: Dict[str, Any] = field(default_factory=dict)
    """Information about how basis was computed."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "components": self.components.cpu().tolist(),
            "mean": self.mean.cpu().tolist(),
            "singular_values": self.singular_values.cpu().tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.cpu().tolist(),
            "n_components": self.n_components,
            "hidden_dim": self.hidden_dim,
            "source_info": self.source_info,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniversalBasis":
        """Load from dictionary."""
        return cls(
            components=torch.tensor(data["components"]),
            mean=torch.tensor(data["mean"]),
            singular_values=torch.tensor(data["singular_values"]),
            explained_variance_ratio=torch.tensor(data["explained_variance_ratio"]),
            n_components=data["n_components"],
            hidden_dim=data["hidden_dim"],
            source_info=data.get("source_info", {}),
        )


def compute_universal_basis(
    vectors: List[torch.Tensor] | Dict[LayerName, torch.Tensor],
    n_components: int = UNIVERSAL_SUBSPACE_RANK,
    normalize: bool = True,
) -> UniversalBasis:
    """
    Compute a universal basis from a collection of steering vectors.
    
    Args:
        vectors: Collection of steering vectors
        n_components: Number of basis components
        normalize: Whether to normalize vectors before PCA
        
    Returns:
        UniversalBasis that can be used for compression/initialization
    """
    log = bind(_LOG)
    
    # Convert to matrix
    if isinstance(vectors, dict):
        vector_list = [v.detach().float().reshape(-1) for v in vectors.values() if v is not None]
    else:
        vector_list = [v.detach().float().reshape(-1) for v in vectors]
    
    matrix = torch.stack(vector_list, dim=0)
    
    if normalize:
        matrix = F.normalize(matrix, p=2, dim=1)
    
    # Compute mean
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    
    # SVD
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    
    # Keep top k components
    k = min(n_components, len(S))
    components = Vh[:k]
    singular_values = S[:k]
    
    # Explained variance
    total_var = (S ** 2).sum()
    explained_var_ratio = (S[:k] ** 2) / total_var
    
    log.info(
        "Computed universal basis",
        extra={
            "n_vectors": len(vector_list),
            "n_components": k,
            "total_variance_explained": explained_var_ratio.sum().item(),
        }
    )
    
    return UniversalBasis(
        components=components,
        mean=mean,
        singular_values=singular_values,
        explained_variance_ratio=explained_var_ratio,
        n_components=k,
        hidden_dim=matrix.shape[1],
        source_info={
            "n_source_vectors": len(vector_list),
            "total_variance_explained": explained_var_ratio.sum().item(),
        }
    )


def compress_steering_vectors(
    vectors: Dict[LayerName, torch.Tensor],
    basis: UniversalBasis,
) -> Dict[LayerName, torch.Tensor]:
    """
    Compress steering vectors to coefficients in universal basis.
    
    Instead of storing [hidden_dim] floats per layer, stores [n_components] coefficients.
    Memory reduction: hidden_dim / n_components (e.g., 4096/16 = 256x).
    
    Args:
        vectors: Dict mapping layer names to steering vectors
        basis: Universal basis for compression
        
    Returns:
        Dict mapping layer names to coefficient vectors [n_components]
    """
    compressed = {}
    
    for layer, vec in vectors.items():
        if vec is None:
            continue
        
        vec_flat = vec.detach().float().reshape(-1)
        
        # Center using basis mean
        centered = vec_flat - basis.mean.to(vec_flat.device)
        
        # Project onto basis components
        coefficients = centered @ basis.components.T.to(vec_flat.device)
        
        compressed[layer] = coefficients
    
    return compressed


def decompress_steering_vectors(
    coefficients: Dict[LayerName, torch.Tensor],
    basis: UniversalBasis,
) -> Dict[LayerName, torch.Tensor]:
    """
    Decompress steering vectors from coefficients.
    
    Args:
        coefficients: Dict mapping layer names to coefficient vectors
        basis: Universal basis used for compression
        
    Returns:
        Dict mapping layer names to reconstructed steering vectors
    """
    decompressed = {}
    
    for layer, coeff in coefficients.items():
        if coeff is None:
            continue
        
        coeff = coeff.to(basis.components.device)
        
        # Reconstruct from basis
        reconstructed = coeff @ basis.components + basis.mean
        
        decompressed[layer] = reconstructed
    
    return decompressed


def save_compressed_vectors(
    coefficients: Dict[LayerName, torch.Tensor],
    basis: UniversalBasis,
    path: str,
) -> None:
    """Save compressed vectors and basis to file."""
    data = {
        "coefficients": {str(k): v.cpu().tolist() for k, v in coefficients.items()},
        "basis": basis.to_dict(),
        "format_version": "1.0",
    }
    
    with open(path, "w") as f:
        json.dump(data, f)


def load_compressed_vectors(path: str) -> Tuple[Dict[LayerName, torch.Tensor], UniversalBasis]:
    """Load compressed vectors and basis from file."""
    with open(path) as f:
        data = json.load(f)
    
    coefficients = {k: torch.tensor(v) for k, v in data["coefficients"].items()}
    basis = UniversalBasis.from_dict(data["basis"])
    
    return coefficients, basis


# =============================================================================
# 3. AUTO NUM_DIRECTIONS BASED ON EXPLAINED VARIANCE
# =============================================================================

def explained_variance_analysis(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    max_components: int = 20,
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
    if total_var < 1e-10:
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
    max_directions: int = 10,
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
# 4. UNIVERSAL BASIS INITIALIZATION FOR PRISM/TITAN
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
    noise_scale: float = 0.1,
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
