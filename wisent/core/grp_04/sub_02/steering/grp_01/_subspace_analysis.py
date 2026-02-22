"""Spectral analysis for steering vector quality."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn.functional as F
from wisent.core.cli.cli_logger import setup_logger, bind
from wisent.core.activations.core.atoms import LayerActivations, LayerName

_LOG = setup_logger(__name__)

UNIVERSAL_SUBSPACE_RANK = 16
VARIANCE_EXPLAINED_THRESHOLD = 0.80

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
