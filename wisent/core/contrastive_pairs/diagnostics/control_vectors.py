"""Diagnostics for steering/control vectors."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Mapping, List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F

from wisent.core.activations.core.atoms import LayerActivations, RawActivationMap

from .base import DiagnosticsIssue, DiagnosticsReport, MetricReport

__all__ = [
    "ControlVectorDiagnosticsConfig",
    "run_control_vector_diagnostics",
    "run_control_steering_diagnostics",
    "ConeAnalysisConfig",
    "ConeAnalysisResult",
    "check_cone_structure",
    "GeometryAnalysisConfig",
    "GeometryAnalysisResult",
    "StructureType",
    "detect_geometry_structure",
]


@dataclass(slots=True)
class ControlVectorDiagnosticsConfig:
    """Thresholds and options for control vector diagnostics."""

    min_norm: float = 1e-4
    max_norm: float | None = None
    zero_value_threshold: float = 1e-8
    max_zero_fraction: float = 0.999
    warn_on_missing: bool = True


def _to_layer_activations(vectors: LayerActivations | RawActivationMap | Mapping[str, object] | None) -> LayerActivations:
    if isinstance(vectors, LayerActivations):
        return vectors
    data: RawActivationMap = vectors or {}
    return LayerActivations(data)


def run_control_vector_diagnostics(
    vectors: LayerActivations | RawActivationMap | Mapping[str, object] | None,
    config: ControlVectorDiagnosticsConfig | None = None,
) -> DiagnosticsReport:
    """Evaluate steering/control vectors for basic health metrics."""

    cfg = config or ControlVectorDiagnosticsConfig()
    activations = _to_layer_activations(vectors)

    issues: list[DiagnosticsIssue] = []
    norms: list[float] = []
    zero_fractions: list[float] = []
    per_layer: dict[str, dict[str, float]] = {}

    for layer, tensor in activations.to_dict().items():
        if tensor is None:
            if cfg.warn_on_missing:
                issues.append(
                    DiagnosticsIssue(
                        metric="control_vectors",
                        severity="warning",
                        message=f"Layer {layer} has no control vector",
                        details={"layer": layer},
                    )
                )
            continue

        detached = tensor.detach()
        if detached.numel() == 0:
            issues.append(
                DiagnosticsIssue(
                    metric="control_vectors",
                    severity="critical",
                    message=f"Layer {layer} control vector is empty",
                    details={"layer": layer},
                )
            )
            continue

        flat = detached.to(dtype=torch.float32, device="cpu").reshape(-1)

        if not torch.isfinite(flat).all():
            non_finite = (~torch.isfinite(flat)).sum().item()
            issues.append(
                DiagnosticsIssue(
                    metric="control_vectors",
                    severity="critical",
                    message=f"Layer {layer} contains non-finite values",
                    details={"layer": layer, "non_finite_entries": int(non_finite)},
                )
            )
            continue

        norm_value = float(torch.linalg.vector_norm(flat).item())
        norms.append(norm_value)

        zero_fraction = float((flat.abs() <= cfg.zero_value_threshold).sum().item()) / float(flat.numel())
        zero_fractions.append(zero_fraction)

        per_layer[layer] = {
            "norm": norm_value,
            "zero_fraction": zero_fraction,
        }

        if norm_value < cfg.min_norm:
            issues.append(
                DiagnosticsIssue(
                    metric="control_vectors",
                    severity="critical",
                    message=f"Layer {layer} control vector norm {norm_value:.3e} below minimum {cfg.min_norm}",
                    details={"layer": layer, "norm": norm_value},
                )
            )

        if cfg.max_norm is not None and norm_value > cfg.max_norm:
            issues.append(
                DiagnosticsIssue(
                    metric="control_vectors",
                    severity="warning",
                    message=f"Layer {layer} control vector norm {norm_value:.3e} exceeds maximum {cfg.max_norm}",
                    details={"layer": layer, "norm": norm_value},
                )
            )

        if zero_fraction >= cfg.max_zero_fraction:
            severity = "critical" if zero_fraction >= 1.0 - 1e-9 else "warning"
            issues.append(
                DiagnosticsIssue(
                    metric="control_vectors",
                    severity=severity,
                    message=(
                        f"Layer {layer} control vector is {zero_fraction:.3%} zero-valued, exceeding allowed {cfg.max_zero_fraction:.3%}"
                    ),
                    details={"layer": layer, "zero_fraction": zero_fraction},
                )
            )

    summary: dict[str, object] = {
        "evaluated_layers": len(norms),
        "norm_min": min(norms) if norms else None,
        "norm_max": max(norms) if norms else None,
        "norm_mean": statistics.mean(norms) if norms else None,
        "norm_median": statistics.median(norms) if norms else None,
        "zero_fraction_max": max(zero_fractions) if zero_fractions else None,
        "per_layer": per_layer,
    }

    if not norms and not issues:
        issues.append(
            DiagnosticsIssue(
                metric="control_vectors",
                severity="critical",
                message="No control vectors were provided for diagnostics",
                details={},
            )
        )

    report = MetricReport(name="control_vectors", summary=summary, issues=issues)
    return DiagnosticsReport.from_metrics([report])

def run_control_steering_diagnostics(steering_vectors: list[RawActivationMap] | RawActivationMap | None) -> list[DiagnosticsReport]:
    if steering_vectors is None:
        return [DiagnosticsReport.from_metrics([])]

    if not isinstance(steering_vectors, list):
        steering_vectors = [steering_vectors]

    # Run diagnostics for each steering vector
    reports = [run_control_vector_diagnostics(vec) for vec in steering_vectors]
    return reports


@dataclass
class ConeAnalysisConfig:
    """Configuration for cone structure analysis."""
    
    num_directions: int = 5
    """Number of directions to discover in the cone."""
    
    optimization_steps: int = 100
    """Gradient steps for cone direction optimization."""
    
    learning_rate: float = 0.01
    """Learning rate for optimization."""
    
    min_cosine_similarity: float = 0.2
    """Minimum cosine similarity between cone directions (should be positive)."""
    
    max_cosine_similarity: float = 0.95
    """Maximum cosine similarity (avoid redundant directions)."""
    
    pca_components: int = 5
    """Number of PCA components to compare against."""
    
    cone_threshold: float = 0.7
    """Threshold for cone_score to declare cone structure exists."""


@dataclass
class ConeAnalysisResult:
    """Results from cone structure analysis."""
    
    has_cone_structure: bool
    """Whether a cone structure was detected."""
    
    cone_score: float
    """Score from 0-1 indicating cone-ness (1 = perfect cone)."""
    
    pca_explained_variance: float
    """Variance explained by PCA directions."""
    
    cone_explained_variance: float
    """Variance explained by cone directions."""
    
    num_directions_found: int
    """Number of valid cone directions discovered."""
    
    direction_cosine_similarities: List[List[float]]
    """Pairwise cosine similarities between discovered directions."""
    
    avg_cosine_similarity: float
    """Average pairwise cosine similarity (high = more cone-like)."""
    
    half_space_consistency: float
    """Fraction of directions in same half-space as primary (1.0 = perfect cone)."""
    
    separation_scores: List[float]
    """Per-direction separation between positive and negative activations."""
    
    positive_combination_score: float
    """How well positive activations can be represented as positive combinations."""
    
    details: Dict[str, Any] = field(default_factory=dict)
    """Additional diagnostic details."""


def check_cone_structure(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    config: ConeAnalysisConfig | None = None,
) -> ConeAnalysisResult:
    """
    Analyze whether activations form a cone structure vs linear subspace.
    
    A cone structure implies:
    1. Multiple directions mediate the behavior (not just one)
    2. These directions are positively correlated (same half-space)
    3. The behavior can be achieved by positive combinations of directions
    4. Cone explains variance better than or comparable to PCA
    
    Arguments:
        pos_activations: Positive example activations [N_pos, hidden_dim]
        neg_activations: Negative example activations [N_neg, hidden_dim]
        config: Analysis configuration
        
    Returns:
        ConeAnalysisResult with cone detection metrics
    """
    cfg = config or ConeAnalysisConfig()
    
    pos_tensor = pos_activations.detach().float()
    neg_tensor = neg_activations.detach().float()
    
    if pos_tensor.dim() == 1:
        pos_tensor = pos_tensor.unsqueeze(0)
    if neg_tensor.dim() == 1:
        neg_tensor = neg_tensor.unsqueeze(0)
    
    hidden_dim = pos_tensor.shape[1]
    
    # Compute difference vectors (the directions we want to analyze)
    diff_vectors = pos_tensor.mean(dim=0, keepdim=True) - neg_tensor.mean(dim=0, keepdim=True)
    
    # 1. PCA Analysis - find linear directions
    pca_directions, pca_explained = _compute_pca_directions(
        pos_tensor, neg_tensor, n_components=cfg.pca_components
    )
    
    # 2. Cone Direction Discovery - gradient-based optimization
    cone_directions, cone_metadata = _discover_cone_directions(
        pos_tensor, neg_tensor, 
        num_directions=cfg.num_directions,
        optimization_steps=cfg.optimization_steps,
        learning_rate=cfg.learning_rate,
        min_cos_sim=cfg.min_cosine_similarity,
        max_cos_sim=cfg.max_cosine_similarity,
    )
    
    # 3. Compute cone explained variance
    cone_explained = _compute_cone_explained_variance(
        pos_tensor, neg_tensor, cone_directions
    )
    
    # 4. Half-space consistency check
    half_space_score = _check_half_space_consistency(cone_directions)
    
    # 5. Positive combination test
    pos_combo_score = _test_positive_combinations(
        pos_tensor, neg_tensor, cone_directions
    )
    
    # 6. Compute cosine similarity matrix
    cos_sim_matrix = _compute_cosine_similarity_matrix(cone_directions)
    avg_cos_sim = _compute_avg_off_diagonal(cos_sim_matrix)
    
    # 7. Separation scores per direction
    separation_scores = _compute_separation_scores(
        pos_tensor, neg_tensor, cone_directions
    )
    
    # 8. Compute final cone score
    cone_score = _compute_cone_score(
        pca_explained=pca_explained,
        cone_explained=cone_explained,
        half_space_score=half_space_score,
        avg_cos_sim=avg_cos_sim,
        pos_combo_score=pos_combo_score,
        separation_scores=separation_scores,
    )
    
    has_cone = cone_score >= cfg.cone_threshold
    
    return ConeAnalysisResult(
        has_cone_structure=has_cone,
        cone_score=cone_score,
        pca_explained_variance=pca_explained,
        cone_explained_variance=cone_explained,
        num_directions_found=cone_directions.shape[0],
        direction_cosine_similarities=cos_sim_matrix.tolist(),
        avg_cosine_similarity=avg_cos_sim,
        half_space_consistency=half_space_score,
        separation_scores=separation_scores,
        positive_combination_score=pos_combo_score,
        details={
            "config": cfg.__dict__,
            "cone_metadata": cone_metadata,
            "pca_directions_shape": list(pca_directions.shape),
            "cone_directions_shape": list(cone_directions.shape),
        }
    )


def _compute_pca_directions(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    n_components: int,
) -> Tuple[torch.Tensor, float]:
    """Compute PCA directions and explained variance ratio."""
    # Combine all activations
    all_activations = torch.cat([pos_tensor, neg_tensor], dim=0)
    
    # Center the data
    mean = all_activations.mean(dim=0, keepdim=True)
    centered = all_activations - mean
    
    # SVD for PCA
    try:
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        
        # Get top k directions
        k = min(n_components, Vh.shape[0])
        pca_directions = Vh[:k]  # [k, hidden_dim]
        
        # Explained variance ratio
        total_var = (S ** 2).sum()
        explained_var = (S[:k] ** 2).sum() / total_var if total_var > 0 else 0.0
        
        return pca_directions, float(explained_var)
    except Exception:
        # Fallback if SVD fails
        return torch.zeros(n_components, pos_tensor.shape[1]), 0.0


def _discover_cone_directions(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    num_directions: int,
    optimization_steps: int,
    learning_rate: float,
    min_cos_sim: float,
    max_cos_sim: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Discover cone directions via gradient optimization.
    Similar to PRISM but focused on cone structure detection.
    """
    hidden_dim = pos_tensor.shape[1]
    
    # Initialize with CAA direction first, then random perturbations
    caa_dir = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
    caa_dir = F.normalize(caa_dir, p=2, dim=0)
    
    directions = torch.randn(num_directions, hidden_dim)
    directions[0] = caa_dir
    
    for i in range(1, num_directions):
        noise = torch.randn(hidden_dim) * 0.3
        directions[i] = F.normalize(caa_dir + noise, p=2, dim=0)
    
    directions = F.normalize(directions, p=2, dim=1)
    directions.requires_grad_(True)
    
    optimizer = torch.optim.Adam([directions], lr=learning_rate)
    
    training_losses = []
    
    for step in range(optimization_steps):
        optimizer.zero_grad()
        
        # Normalize for computation
        dirs_norm = F.normalize(directions, p=2, dim=1)
        
        # Loss 1: Separation - each direction should separate pos from neg
        pos_proj = pos_tensor @ dirs_norm.T  # [N_pos, K]
        neg_proj = neg_tensor @ dirs_norm.T  # [N_neg, K]
        separation_loss = -((pos_proj.mean(dim=0) - neg_proj.mean(dim=0)).abs().mean())
        
        # Loss 2: Cone constraint - directions should be positively correlated
        cos_sim = dirs_norm @ dirs_norm.T
        off_diag_mask = 1 - torch.eye(num_directions)
        off_diag = cos_sim * off_diag_mask
        
        # Penalize negative correlations (not a cone)
        negative_penalty = F.relu(-off_diag).sum()
        
        # Penalize too similar (redundant)
        too_similar = F.relu(off_diag - max_cos_sim).sum()
        
        # Penalize too dissimilar (not a cone)
        too_dissimilar = F.relu(min_cos_sim - off_diag).sum()
        
        cone_loss = negative_penalty + too_similar + 0.5 * too_dissimilar
        
        # Loss 3: Diversity - directions should capture different aspects
        diversity_loss = -off_diag.var()
        
        total_loss = separation_loss + 0.5 * cone_loss + 0.1 * diversity_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Project back to unit sphere
        with torch.no_grad():
            directions.data = F.normalize(directions.data, p=2, dim=1)
            
            # Ensure cone constraint: flip directions to same half-space as first
            if directions.shape[0] > 1:
                primary = directions[0:1]
                for i in range(1, directions.shape[0]):
                    if (directions[i:i+1] @ primary.T).item() < 0:
                        directions.data[i] = -directions.data[i]
        
        if step % 20 == 0:
            training_losses.append(float(total_loss.item()))
    
    final_directions = directions.detach()
    
    metadata = {
        "training_losses": training_losses,
        "final_loss": float(total_loss.item()),
    }
    
    return final_directions, metadata


def _compute_cone_explained_variance(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    cone_directions: torch.Tensor,
) -> float:
    """Compute how much variance the cone directions explain."""
    all_activations = torch.cat([pos_tensor, neg_tensor], dim=0)
    mean = all_activations.mean(dim=0, keepdim=True)
    centered = all_activations - mean
    
    total_var = (centered ** 2).sum()
    if total_var == 0:
        return 0.0
    
    # Project onto cone directions
    dirs_norm = F.normalize(cone_directions, p=2, dim=1)
    projections = centered @ dirs_norm.T  # [N, K]
    reconstructed = projections @ dirs_norm  # [N, hidden_dim]
    
    explained_var = (reconstructed ** 2).sum() / total_var
    return float(min(explained_var.item(), 1.0))


def _check_half_space_consistency(directions: torch.Tensor) -> float:
    """Check what fraction of directions are in the same half-space as the primary."""
    if directions.shape[0] <= 1:
        return 1.0
    
    dirs_norm = F.normalize(directions, p=2, dim=1)
    primary = dirs_norm[0:1]
    
    # Cosine similarity with primary
    cos_with_primary = (dirs_norm @ primary.T).squeeze()
    
    # Count positive correlations
    positive_count = (cos_with_primary > 0).sum().item()
    return positive_count / directions.shape[0]


def _test_positive_combinations(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    directions: torch.Tensor,
) -> float:
    """
    Test if difference vectors can be represented as positive combinations
    of cone directions (key property of polyhedral cones).
    """
    diff = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
    diff_norm = F.normalize(diff.unsqueeze(0), p=2, dim=1)
    dirs_norm = F.normalize(directions, p=2, dim=1)
    
    # Project difference onto each direction
    projections = (diff_norm @ dirs_norm.T).squeeze()
    
    # In a perfect cone, all projections should be non-negative
    positive_projections = (projections >= 0).sum().item()
    
    # Also check magnitude - projections should be substantial
    significant_projections = (projections > 0.1).sum().item()
    
    # Score combines both
    pos_ratio = positive_projections / directions.shape[0]
    sig_ratio = significant_projections / directions.shape[0]
    
    return 0.7 * pos_ratio + 0.3 * sig_ratio


def _compute_cosine_similarity_matrix(directions: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix."""
    dirs_norm = F.normalize(directions, p=2, dim=1)
    return dirs_norm @ dirs_norm.T


def _compute_avg_off_diagonal(matrix: torch.Tensor) -> float:
    """Compute average of off-diagonal elements."""
    n = matrix.shape[0]
    if n <= 1:
        return 1.0
    mask = 1 - torch.eye(n)
    return float((matrix * mask).sum() / (n * (n - 1)))


def _compute_separation_scores(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    directions: torch.Tensor,
) -> List[float]:
    """Compute separation score for each direction."""
    dirs_norm = F.normalize(directions, p=2, dim=1)
    
    pos_proj = pos_tensor @ dirs_norm.T
    neg_proj = neg_tensor @ dirs_norm.T
    
    separation = pos_proj.mean(dim=0) - neg_proj.mean(dim=0)
    return separation.tolist()


def _compute_cone_score(
    pca_explained: float,
    cone_explained: float,
    half_space_score: float,
    avg_cos_sim: float,
    pos_combo_score: float,
    separation_scores: List[float],
) -> float:
    """
    Compute overall cone score combining all metrics.
    
    High score indicates:
    - Cone explains variance well (comparable to or better than PCA)
    - Directions are in same half-space
    - Directions are positively correlated but not redundant
    - Difference can be expressed as positive combinations
    - Multiple directions contribute to separation
    """
    # 1. Variance explanation ratio (cone vs PCA)
    var_ratio = cone_explained / max(pca_explained, 1e-6)
    var_score = min(var_ratio, 1.0)  # Cap at 1.0
    
    # 2. Half-space consistency (critical for cone)
    half_space_component = half_space_score
    
    # 3. Cosine similarity should be moderate (0.3-0.7 is ideal for cone)
    # Too low = not a cone, too high = redundant
    if avg_cos_sim < 0:
        cos_score = 0.0  # Negative correlation = not a cone
    elif avg_cos_sim < 0.3:
        cos_score = avg_cos_sim / 0.3 * 0.5  # Below ideal range
    elif avg_cos_sim <= 0.7:
        cos_score = 1.0  # Ideal range
    else:
        cos_score = max(0.5, 1.0 - (avg_cos_sim - 0.7) / 0.3)  # Too similar
    
    # 4. Positive combination score
    combo_component = pos_combo_score
    
    # 5. Multi-direction contribution
    # Check if multiple directions have significant separation
    significant_directions = sum(1 for s in separation_scores if abs(s) > 0.1)
    multi_dir_score = min(significant_directions / max(len(separation_scores), 1), 1.0)
    
    # Weighted combination
    cone_score = (
        0.20 * var_score +
        0.25 * half_space_component +
        0.20 * cos_score +
        0.20 * combo_component +
        0.15 * multi_dir_score
    )
    
    return float(cone_score)


# =============================================================================
# Comprehensive Geometry Structure Detection
# =============================================================================

from enum import Enum


class StructureType(Enum):
    """Types of geometric structures that can be detected in activation space."""
    LINEAR = "linear"
    CONE = "cone"
    CLUSTER = "cluster"
    MANIFOLD = "manifold"
    SPARSE = "sparse"
    BIMODAL = "bimodal"
    ORTHOGONAL = "orthogonal"
    UNKNOWN = "unknown"


@dataclass
class GeometryAnalysisConfig:
    """Configuration for comprehensive geometry analysis.
    
    Default thresholds are tuned based on the Universal Subspace Hypothesis
    (Kaushik et al., 2025), which shows that neural networks converge to
    shared low-dimensional subspaces. Key implications:
    - Linear structure is more common than previously assumed
    - True cone/manifold structures are rarer
    - ~16 principal directions capture most variance
    """
    
    # General settings
    num_components: int = 5
    """Number of components/directions to analyze."""
    
    optimization_steps: int = 100
    """Steps for optimization-based methods."""
    
    # Linear detection - raised threshold per Universal Subspace findings
    linear_variance_threshold: float = 0.85
    """Variance explained threshold to declare linear structure."""
    
    # Cone detection - lowered threshold (true cones are rarer)
    cone_threshold: float = 0.65
    """Cone score threshold."""
    
    # Cluster detection
    max_clusters: int = 5
    """Maximum number of clusters to try."""
    
    cluster_silhouette_threshold: float = 0.55
    """Silhouette score threshold for cluster detection."""
    
    # Manifold detection
    manifold_neighbors: int = 10
    """Number of neighbors for manifold analysis."""
    
    manifold_threshold: float = 0.70
    """Score threshold for manifold structure."""
    
    # Sparse detection
    sparse_threshold: float = 0.1
    """Fraction of active dimensions threshold."""
    
    # Bimodal detection
    bimodal_dip_threshold: float = 0.05
    """P-value threshold for dip test."""
    
    # Orthogonal detection - stricter (orthogonal is rare in universal subspace)
    orthogonal_threshold: float = 0.12
    """Max correlation for orthogonal subspaces."""
    
    # Universal subspace integration
    use_universal_thresholds: bool = True
    """Whether to use thresholds tuned for universal subspace theory."""


@dataclass
class StructureScore:
    """Score for a single structure type."""
    structure_type: StructureType
    score: float
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeometryAnalysisResult:
    """Results from comprehensive geometry analysis."""
    
    best_structure: StructureType
    """The structure type that best fits the data."""
    
    best_score: float
    """Score of the best-fitting structure."""
    
    all_scores: Dict[str, StructureScore]
    """Scores for all analyzed structure types."""
    
    recommendation: str
    """Recommended steering method based on geometry."""
    
    details: Dict[str, Any] = field(default_factory=dict)
    """Additional analysis details."""
    
    def get_ranking(self) -> List[Tuple[StructureType, float]]:
        """Get structures ranked by score."""
        return sorted(
            [(s.structure_type, s.score) for s in self.all_scores.values()],
            key=lambda x: x[1],
            reverse=True
        )


def detect_geometry_structure(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    config: GeometryAnalysisConfig | None = None,
) -> GeometryAnalysisResult:
    """
    Detect the geometric structure of activation differences.
    
    Uses HIERARCHICAL detection - structures are mutually exclusive:
    - Linear: Single direction explains the data (simplest)
    - Cone: Multiple correlated directions needed (more complex than linear)
    - Cluster: Discrete groups (different from continuous structures)
    - Orthogonal: Independent subspaces (different from cone)
    - Sparse: Few neurons encode the behavior
    - Bimodal: Two distinct modes
    - Manifold: Non-linear curved structure (most general, fallback)
    
    The key insight: Linear ⊂ Cone ⊂ Manifold, so we check simpler
    structures first and only report more complex structures if simpler
    ones don't fit well.
    
    Arguments:
        pos_activations: Positive example activations [N_pos, hidden_dim]
        neg_activations: Negative example activations [N_neg, hidden_dim]
        config: Analysis configuration
        
    Returns:
        GeometryAnalysisResult with scores for each structure type
    """
    cfg = config or GeometryAnalysisConfig()
    
    pos_tensor = pos_activations.detach().float()
    neg_tensor = neg_activations.detach().float()
    
    if pos_tensor.dim() == 1:
        pos_tensor = pos_tensor.unsqueeze(0)
    if neg_tensor.dim() == 1:
        neg_tensor = neg_tensor.unsqueeze(0)
    
    # Compute difference vectors (primary analysis target)
    diff_vectors = pos_tensor - neg_tensor[:pos_tensor.shape[0]] if neg_tensor.shape[0] >= pos_tensor.shape[0] else pos_tensor[:neg_tensor.shape[0]] - neg_tensor
    
    # Compute raw scores for each structure type
    raw_scores: Dict[str, StructureScore] = {}
    
    # 1. Linear structure detection
    linear_score = _detect_linear_structure(pos_tensor, neg_tensor, diff_vectors, cfg)
    raw_scores["linear"] = linear_score
    
    # 2. Cone structure detection
    cone_score = _detect_cone_structure_score(pos_tensor, neg_tensor, cfg)
    raw_scores["cone"] = cone_score
    
    # 3. Cluster structure detection
    cluster_score = _detect_cluster_structure(pos_tensor, neg_tensor, diff_vectors, cfg)
    raw_scores["cluster"] = cluster_score
    
    # 4. Manifold structure detection
    manifold_score = _detect_manifold_structure(pos_tensor, neg_tensor, diff_vectors, cfg)
    raw_scores["manifold"] = manifold_score
    
    # 5. Sparse structure detection
    sparse_score = _detect_sparse_structure(pos_tensor, neg_tensor, diff_vectors, cfg)
    raw_scores["sparse"] = sparse_score
    
    # 6. Bimodal structure detection
    bimodal_score = _detect_bimodal_structure(pos_tensor, neg_tensor, diff_vectors, cfg)
    raw_scores["bimodal"] = bimodal_score
    
    # 7. Orthogonal subspaces detection
    orthogonal_score = _detect_orthogonal_structure(pos_tensor, neg_tensor, diff_vectors, cfg)
    raw_scores["orthogonal"] = orthogonal_score
    
    # HIERARCHICAL ADJUSTMENT: Make scores mutually exclusive
    # The principle: only credit a more complex structure if simpler ones fail
    all_scores = _apply_hierarchical_scoring(raw_scores)
    
    # Find best structure based on adjusted scores
    best_key = max(all_scores.keys(), key=lambda k: all_scores[k].score)
    best_structure = all_scores[best_key].structure_type
    best_score = all_scores[best_key].score
    
    # Generate recommendation
    recommendation = _generate_recommendation(best_structure, all_scores)
    
    return GeometryAnalysisResult(
        best_structure=best_structure,
        best_score=best_score,
        all_scores=all_scores,
        recommendation=recommendation,
        details={
            "config": cfg.__dict__,
            "n_positive": pos_tensor.shape[0],
            "n_negative": neg_tensor.shape[0],
            "hidden_dim": pos_tensor.shape[1],
            "raw_scores": {k: v.score for k, v in raw_scores.items()},
        }
    )


def _apply_hierarchical_scoring(raw_scores: Dict[str, StructureScore]) -> Dict[str, StructureScore]:
    """
    Apply hierarchical scoring to make structure types mutually exclusive.
    
    Hierarchy (simpler to more complex):
    1. Linear - if high, don't credit cone/manifold
    2. Cone - if high (and linear is low), don't credit manifold
    3. Cluster - independent axis (discrete vs continuous)
    4. Sparse - independent axis (encoding style)
    5. Bimodal - independent axis
    6. Orthogonal - alternative to cone (uncorrelated vs correlated directions)
    7. Manifold - fallback (only if nothing else fits)
    
    The adjusted score represents: "How well does THIS structure explain
    what simpler structures cannot?"
    """
    adjusted: Dict[str, StructureScore] = {}
    
    linear_raw = raw_scores.get("linear", StructureScore(StructureType.LINEAR, 0, 0)).score
    cone_raw = raw_scores.get("cone", StructureScore(StructureType.CONE, 0, 0)).score
    cluster_raw = raw_scores.get("cluster", StructureScore(StructureType.CLUSTER, 0, 0)).score
    manifold_raw = raw_scores.get("manifold", StructureScore(StructureType.MANIFOLD, 0, 0)).score
    sparse_raw = raw_scores.get("sparse", StructureScore(StructureType.SPARSE, 0, 0)).score
    bimodal_raw = raw_scores.get("bimodal", StructureScore(StructureType.BIMODAL, 0, 0)).score
    orthogonal_raw = raw_scores.get("orthogonal", StructureScore(StructureType.ORTHOGONAL, 0, 0)).score
    
    # Thresholds for "structure is sufficient"
    LINEAR_THRESHOLD = 0.6  # If linear > 0.6, linear structure is sufficient
    CONE_THRESHOLD = 0.5    # If cone > 0.5 (after adjustment), cone is sufficient
    
    # 1. LINEAR: No adjustment needed - it's the simplest
    adjusted["linear"] = StructureScore(
        StructureType.LINEAR,
        score=linear_raw,
        confidence=raw_scores["linear"].confidence,
        details={**raw_scores["linear"].details, "adjustment": "none (baseline)"}
    )
    
    # 2. CONE: Only credit if linear is insufficient
    # Cone score = raw_cone * (1 - linear_sufficiency)
    linear_sufficiency = min(1.0, linear_raw / LINEAR_THRESHOLD) if linear_raw > 0 else 0
    cone_adjusted = cone_raw * (1 - linear_sufficiency * 0.8)  # Reduce cone if linear is good
    adjusted["cone"] = StructureScore(
        StructureType.CONE,
        score=cone_adjusted,
        confidence=raw_scores["cone"].confidence,
        details={**raw_scores["cone"].details, "adjustment": f"reduced by linear_sufficiency={linear_sufficiency:.2f}"}
    )
    
    # 3. MANIFOLD: Only credit if both linear AND cone are insufficient
    # This is the "fallback" - only use if simpler structures don't work
    cone_sufficiency = min(1.0, max(linear_raw, cone_raw) / CONE_THRESHOLD)
    manifold_adjusted = manifold_raw * (1 - cone_sufficiency * 0.9)  # Heavily penalize if simpler works
    adjusted["manifold"] = StructureScore(
        StructureType.MANIFOLD,
        score=manifold_adjusted,
        confidence=raw_scores["manifold"].confidence,
        details={**raw_scores["manifold"].details, "adjustment": f"reduced by cone_sufficiency={cone_sufficiency:.2f}"}
    )
    
    # 4. CLUSTER: Independent axis - but penalize if continuous structures work
    # Cluster is meaningful only if data is truly discrete, not continuous
    continuous_score = max(linear_raw, cone_raw)
    cluster_adjusted = cluster_raw * (1 - continuous_score * 0.5)
    adjusted["cluster"] = StructureScore(
        StructureType.CLUSTER,
        score=cluster_adjusted,
        confidence=raw_scores["cluster"].confidence,
        details={**raw_scores["cluster"].details, "adjustment": f"reduced by continuous_score={continuous_score:.2f}"}
    )
    
    # 5. SPARSE: Independent axis - about encoding style, not geometry
    # Keep mostly unchanged, slight penalty if linear is very high (sparse + linear = still linear)
    sparse_adjusted = sparse_raw * (1 - linear_raw * 0.3)
    adjusted["sparse"] = StructureScore(
        StructureType.SPARSE,
        score=sparse_adjusted,
        confidence=raw_scores["sparse"].confidence,
        details={**raw_scores["sparse"].details, "adjustment": f"slight reduction for linear={linear_raw:.2f}"}
    )
    
    # 6. BIMODAL: Independent axis - about distribution shape
    # No adjustment needed
    adjusted["bimodal"] = StructureScore(
        StructureType.BIMODAL,
        score=bimodal_raw,
        confidence=raw_scores["bimodal"].confidence,
        details={**raw_scores["bimodal"].details, "adjustment": "none (independent axis)"}
    )
    
    # 7. ORTHOGONAL: Alternative to cone (mutually exclusive)
    # If directions are correlated (cone), they're not orthogonal
    # Only credit orthogonal if cone is low
    orthogonal_adjusted = orthogonal_raw * (1 - cone_raw * 0.7)
    adjusted["orthogonal"] = StructureScore(
        StructureType.ORTHOGONAL,
        score=orthogonal_adjusted,
        confidence=raw_scores["orthogonal"].confidence,
        details={**raw_scores["orthogonal"].details, "adjustment": f"reduced by cone={cone_raw:.2f}"}
    )
    
    return adjusted


def _detect_linear_structure(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    diff_vectors: torch.Tensor,
    cfg: GeometryAnalysisConfig,
) -> StructureScore:
    """Detect if a single linear direction captures the behavior."""
    if pos_tensor.shape[0] < 2 or neg_tensor.shape[0] < 2:
        return StructureScore(StructureType.LINEAR, 0.0, 0.0, {"reason": "insufficient_data"})
    
    try:
        # Compute mean difference direction
        mean_diff = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
        mean_diff_norm = mean_diff.norm()
        if mean_diff_norm < 1e-8:
            return StructureScore(StructureType.LINEAR, 0.0, 0.0, {"reason": "no_separation"})
        
        primary_dir = mean_diff / mean_diff_norm
        
        # Project all samples onto primary direction
        pos_proj = pos_tensor @ primary_dir
        neg_proj = neg_tensor @ primary_dir
        
        # Measure separation quality (Cohen's d)
        pos_mean, pos_std = pos_proj.mean(), pos_proj.std()
        neg_mean, neg_std = neg_proj.mean(), neg_proj.std()
        pooled_std = ((pos_std**2 + neg_std**2) / 2).sqrt()
        cohens_d = abs(pos_mean - neg_mean) / (pooled_std + 1e-8)
        
        # Measure variance explained by single direction
        # Compute residual variance after projecting out primary direction
        pos_residual = pos_tensor - (pos_proj.unsqueeze(1) * primary_dir.unsqueeze(0))
        neg_residual = neg_tensor - (neg_proj.unsqueeze(1) * primary_dir.unsqueeze(0))
        
        total_var = pos_tensor.var() + neg_tensor.var()
        residual_var = pos_residual.var() + neg_residual.var()
        variance_explained = 1 - (residual_var / (total_var + 1e-8))
        variance_explained = max(0, min(1, float(variance_explained)))
        
        # Measure within-class consistency (low spread along primary direction)
        within_class_spread = (pos_std + neg_std) / 2
        between_class_dist = abs(pos_mean - neg_mean)
        spread_ratio = within_class_spread / (between_class_dist + 1e-8)
        consistency = max(0, 1 - spread_ratio)  # High when spread is low relative to separation
        
        # Linear score: high cohens_d + high variance explained + high consistency
        linear_score = (
            0.35 * min(float(cohens_d) / 5, 1.0) +  # Separation quality
            0.35 * variance_explained +              # Single direction captures variance
            0.30 * consistency                       # Low within-class variance
        )
        
        confidence = min(1.0, (pos_tensor.shape[0] + neg_tensor.shape[0]) / 50)
        
        return StructureScore(
            StructureType.LINEAR,
            score=float(linear_score),
            confidence=float(confidence),
            details={
                "cohens_d": float(cohens_d),
                "variance_explained": float(variance_explained),
                "within_class_consistency": float(consistency),
                "pos_std": float(pos_std),
                "neg_std": float(neg_std),
                "separation": float(between_class_dist),
            }
        )
    except Exception as e:
        return StructureScore(StructureType.LINEAR, 0.0, 0.0, {"error": str(e)})


def _detect_cone_structure_score(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    cfg: GeometryAnalysisConfig,
) -> StructureScore:
    """Detect cone structure and return as StructureScore."""
    cone_config = ConeAnalysisConfig(
        num_directions=cfg.num_components,
        optimization_steps=cfg.optimization_steps,
        cone_threshold=cfg.cone_threshold,
    )
    
    try:
        result = check_cone_structure(pos_tensor, neg_tensor, cone_config)
        
        # Cone is meaningful when:
        # 1. Multiple directions are needed (PCA doesn't capture everything)
        # 2. But directions are correlated (same half-space)
        # 3. Cosine similarity is moderate (0.3-0.7 range ideal)
        
        # Penalize if PCA already explains most variance (that's linear, not cone)
        pca_penalty = result.pca_explained_variance  # High PCA = linear is enough
        
        # Reward if cone explains more than PCA
        cone_advantage = max(0, result.cone_explained_variance - result.pca_explained_variance)
        
        # Cone needs moderate cosine similarity - not too high (= linear) not too low (= orthogonal)
        cos_sim = result.avg_cosine_similarity
        if cos_sim > 0.85:
            # Very high similarity means directions are basically the same = linear
            cosine_score = 0.3
        elif cos_sim > 0.7:
            cosine_score = 0.7
        elif cos_sim > 0.3:
            # Ideal range for cone
            cosine_score = 1.0
        else:
            # Too different = not a cone
            cosine_score = max(0, cos_sim / 0.3)
        
        # Multiple significant directions needed
        significant_dirs = sum(1 for s in result.separation_scores if abs(s) > 0.1)
        multi_dir_score = min(significant_dirs / cfg.num_components, 1.0)
        
        # Adjusted cone score
        cone_score = (
            0.25 * result.half_space_consistency +
            0.25 * cosine_score +
            0.20 * cone_advantage +
            0.15 * multi_dir_score +
            0.15 * (1 - pca_penalty)  # Penalize when PCA is sufficient
        )
        
        return StructureScore(
            StructureType.CONE,
            score=float(cone_score),
            confidence=result.half_space_consistency,
            details={
                "pca_explained": result.pca_explained_variance,
                "cone_explained": result.cone_explained_variance,
                "cone_advantage": float(cone_advantage),
                "avg_cosine_similarity": result.avg_cosine_similarity,
                "half_space_consistency": result.half_space_consistency,
                "num_directions": result.num_directions_found,
                "significant_directions": significant_dirs,
            }
        )
    except Exception as e:
        return StructureScore(StructureType.CONE, 0.0, 0.0, {"error": str(e)})


def _detect_cluster_structure(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    diff_vectors: torch.Tensor,
    cfg: GeometryAnalysisConfig,
) -> StructureScore:
    """Detect if activations form discrete clusters."""
    all_activations = torch.cat([pos_tensor, neg_tensor], dim=0)
    n_samples = all_activations.shape[0]
    
    if n_samples < 6:
        return StructureScore(StructureType.CLUSTER, 0.0, 0.0, {"reason": "insufficient_data"})
    
    best_silhouette = -1.0
    best_k = 2
    silhouette_scores = {}
    
    for k in range(2, min(cfg.max_clusters + 1, n_samples // 2)):
        try:
            # Simple k-means implementation
            labels, centroids, silhouette = _kmeans_with_silhouette(all_activations, k, max_iters=50)
            silhouette_scores[k] = silhouette
            
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
        except Exception:
            continue
    
    if best_silhouette < 0:
        return StructureScore(StructureType.CLUSTER, 0.0, 0.0, {"reason": "clustering_failed"})
    
    # Check if clusters separate pos/neg
    labels, _, _ = _kmeans_with_silhouette(all_activations, best_k, max_iters=50)
    pos_labels = labels[:pos_tensor.shape[0]]
    neg_labels = labels[pos_tensor.shape[0]:]
    
    # Cluster purity: do pos and neg end up in different clusters?
    pos_majority = pos_labels.mode().values.item() if len(pos_labels) > 0 else -1
    neg_majority = neg_labels.mode().values.item() if len(neg_labels) > 0 else -1
    cluster_separation = 1.0 if pos_majority != neg_majority else 0.5
    
    # Silhouette score ranges from -1 to 1, where:
    # > 0.7 = strong structure
    # 0.5-0.7 = reasonable structure
    # 0.25-0.5 = weak structure
    # < 0.25 = no substantial structure
    
    # Only consider cluster structure if silhouette is reasonably high
    if best_silhouette < cfg.cluster_silhouette_threshold:
        # Low silhouette means no clear cluster structure
        cluster_score = best_silhouette * 0.5  # Scale down significantly
    else:
        # Good silhouette - this is truly clustered data
        # Normalize silhouette from [threshold, 1] to [0.5, 1]
        normalized_silhouette = (best_silhouette - cfg.cluster_silhouette_threshold) / (1 - cfg.cluster_silhouette_threshold)
        cluster_score = 0.5 + 0.4 * normalized_silhouette + 0.1 * cluster_separation
    
    return StructureScore(
        StructureType.CLUSTER,
        score=float(cluster_score),
        confidence=float(max(0, best_silhouette)),
        details={
            "best_k": best_k,
            "best_silhouette": float(best_silhouette),
            "all_silhouettes": {str(k): float(v) for k, v in silhouette_scores.items()},
            "cluster_separation": float(cluster_separation),
            "silhouette_threshold": cfg.cluster_silhouette_threshold,
        }
    )


def _kmeans_with_silhouette(
    data: torch.Tensor,
    k: int,
    max_iters: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Simple k-means with silhouette score computation."""
    n_samples, n_features = data.shape
    
    # Initialize centroids randomly
    indices = torch.randperm(n_samples)[:k]
    centroids = data[indices].clone()
    
    for _ in range(max_iters):
        # Assign labels
        distances = torch.cdist(data, centroids)
        labels = distances.argmin(dim=1)
        
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for i in range(k):
            mask = labels == i
            if mask.sum() > 0:
                new_centroids[i] = data[mask].mean(dim=0)
            else:
                new_centroids[i] = centroids[i]
        
        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids
    
    # Compute silhouette score
    silhouette = _compute_silhouette(data, labels, k)
    
    return labels, centroids, silhouette


def _compute_silhouette(data: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Compute silhouette score."""
    n_samples = data.shape[0]
    if n_samples < 2 or k < 2:
        return 0.0
    
    silhouette_samples = []
    
    for i in range(n_samples):
        label_i = labels[i].item()
        
        # a(i): mean distance to same cluster
        same_cluster = data[labels == label_i]
        if same_cluster.shape[0] > 1:
            a_i = (data[i] - same_cluster).norm(dim=1).sum() / (same_cluster.shape[0] - 1)
        else:
            a_i = 0.0
        
        # b(i): min mean distance to other clusters
        b_i = float('inf')
        for j in range(k):
            if j != label_i:
                other_cluster = data[labels == j]
                if other_cluster.shape[0] > 0:
                    mean_dist = (data[i] - other_cluster).norm(dim=1).mean()
                    b_i = min(b_i, mean_dist.item())
        
        if b_i == float('inf'):
            b_i = 0.0
        
        # Silhouette for sample i
        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0.0
        
        silhouette_samples.append(s_i)
    
    return float(sum(silhouette_samples) / len(silhouette_samples)) if silhouette_samples else 0.0


def _detect_manifold_structure(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    diff_vectors: torch.Tensor,
    cfg: GeometryAnalysisConfig,
) -> StructureScore:
    """Detect non-linear manifold structure via intrinsic dimensionality."""
    all_activations = torch.cat([pos_tensor, neg_tensor], dim=0)
    n_samples = all_activations.shape[0]
    
    if n_samples < cfg.manifold_neighbors + 1:
        return StructureScore(StructureType.MANIFOLD, 0.0, 0.0, {"reason": "insufficient_data"})
    
    try:
        # First check if there's meaningful separation
        mean_diff = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
        separation_strength = mean_diff.norm() / (pos_tensor.std() + neg_tensor.std() + 1e-8)
        has_structure = min(float(separation_strength) / 2, 1.0)
        
        if has_structure < 0.2:
            # No meaningful separation - can't determine manifold structure
            return StructureScore(StructureType.MANIFOLD, 0.1, 0.0, {"reason": "no_separation"})
        
        # Estimate intrinsic dimensionality using correlation dimension
        intrinsic_dim = _estimate_intrinsic_dimensionality(all_activations, cfg.manifold_neighbors)
        
        # Compare to ambient dimension
        ambient_dim = all_activations.shape[1]
        dim_ratio = intrinsic_dim / ambient_dim
        
        # Also compute local linearity deviation
        local_nonlinearity = _compute_local_nonlinearity(all_activations, cfg.manifold_neighbors)
        
        # Manifold score: high if low intrinsic dim AND non-linear AND has structure
        # Low intrinsic dim alone could be linear, so we need nonlinearity
        # But random noise also has "nonlinearity" - need to distinguish
        
        # Manifold is meaningful only with significant dimension reduction
        if dim_ratio > 0.5:
            # Not much dimension reduction = not a clear manifold
            manifold_score = 0.3 * has_structure
        else:
            manifold_score = (
                0.30 * (1 - dim_ratio) +
                0.25 * local_nonlinearity +
                0.45 * has_structure  # Weight structure heavily
            )
        
        # Confidence based on sample size
        confidence = min(1.0, n_samples / 100)
        
        return StructureScore(
            StructureType.MANIFOLD,
            score=float(manifold_score),
            confidence=float(confidence),
            details={
                "intrinsic_dimensionality": float(intrinsic_dim),
                "ambient_dimensionality": ambient_dim,
                "dim_ratio": float(dim_ratio),
                "local_nonlinearity": float(local_nonlinearity),
            }
        )
    except Exception as e:
        return StructureScore(StructureType.MANIFOLD, 0.0, 0.0, {"error": str(e)})


def _estimate_intrinsic_dimensionality(data: torch.Tensor, k: int) -> float:
    """Estimate intrinsic dimensionality using MLE method."""
    n_samples = data.shape[0]
    
    # Compute pairwise distances
    distances = torch.cdist(data, data)
    
    # For each point, get k nearest neighbors (excluding self)
    intrinsic_dims = []
    
    for i in range(n_samples):
        dists_i = distances[i]
        dists_i[i] = float('inf')  # Exclude self
        
        # Get k smallest distances
        knn_dists, _ = torch.topk(dists_i, k, largest=False)
        knn_dists = knn_dists[knn_dists > 1e-10]  # Filter zeros
        
        if len(knn_dists) < 2:
            continue
        
        # MLE estimator for intrinsic dimensionality
        # d = 1 / (mean(log(r_k / r_j)) for j < k)
        r_k = knn_dists[-1]
        log_ratios = torch.log(r_k / knn_dists[:-1])
        
        if log_ratios.mean() > 0:
            d_i = 1.0 / log_ratios.mean()
            intrinsic_dims.append(min(float(d_i), data.shape[1]))  # Cap at ambient dim
    
    if not intrinsic_dims:
        return float(data.shape[1])
    
    return float(sum(intrinsic_dims) / len(intrinsic_dims))


def _compute_local_nonlinearity(data: torch.Tensor, k: int) -> float:
    """Compute how much local neighborhoods deviate from linear."""
    n_samples = data.shape[0]
    distances = torch.cdist(data, data)
    
    nonlinearity_scores = []
    
    for i in range(min(n_samples, 50)):  # Sample for efficiency
        # Get k nearest neighbors
        dists_i = distances[i].clone()
        dists_i[i] = float('inf')
        _, knn_indices = torch.topk(dists_i, k, largest=False)
        
        # Get local neighborhood
        neighborhood = data[knn_indices]
        center = neighborhood.mean(dim=0, keepdim=True)
        centered = neighborhood - center
        
        # PCA on local neighborhood
        try:
            _, S, _ = torch.linalg.svd(centered, full_matrices=False)
            
            # Nonlinearity: how spread are singular values?
            # Linear would have first few dominating
            total_var = (S ** 2).sum()
            if total_var > 0:
                # Entropy-like measure of variance distribution
                var_dist = (S ** 2) / total_var
                var_dist = var_dist[var_dist > 1e-10]
                entropy = -(var_dist * torch.log(var_dist + 1e-10)).sum()
                max_entropy = torch.log(torch.tensor(float(len(var_dist))))
                nonlinearity = float(entropy / max_entropy) if max_entropy > 0 else 0.0
                nonlinearity_scores.append(nonlinearity)
        except Exception:
            continue
    
    return float(sum(nonlinearity_scores) / len(nonlinearity_scores)) if nonlinearity_scores else 0.0


def _detect_sparse_structure(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    diff_vectors: torch.Tensor,
    cfg: GeometryAnalysisConfig,
) -> StructureScore:
    """Detect if behavior is encoded in sparse neuron activations."""
    # Mean difference vector
    mean_diff = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
    
    # Compute sparsity metrics
    abs_diff = mean_diff.abs()
    
    # L1/L2 ratio (lower = sparser)
    l1_norm = abs_diff.sum()
    l2_norm = abs_diff.norm()
    
    if l2_norm > 0:
        l1_l2_ratio = l1_norm / (l2_norm * (len(mean_diff) ** 0.5))
    else:
        l1_l2_ratio = 1.0
    
    # Fraction of "active" dimensions (above threshold)
    threshold = abs_diff.max() * cfg.sparse_threshold
    active_fraction = (abs_diff > threshold).float().mean()
    
    # Gini coefficient (measures inequality)
    sorted_abs = abs_diff.sort().values
    n = len(sorted_abs)
    cumsum = sorted_abs.cumsum(0)
    gini = (2 * torch.arange(1, n + 1, dtype=torch.float32) @ sorted_abs - (n + 1) * sorted_abs.sum()) / (n * sorted_abs.sum() + 1e-10)
    
    # Sparse score: high if few dimensions are active
    sparse_score = 0.4 * (1 - float(l1_l2_ratio)) + 0.3 * (1 - float(active_fraction)) + 0.3 * float(gini)
    sparse_score = max(0, min(1, sparse_score))
    
    # Top contributing dimensions
    top_k = min(10, len(mean_diff))
    top_values, top_indices = torch.topk(abs_diff, top_k)
    top_contribution = top_values.sum() / (abs_diff.sum() + 1e-10)
    
    return StructureScore(
        StructureType.SPARSE,
        score=float(sparse_score),
        confidence=min(1.0, (pos_tensor.shape[0] + neg_tensor.shape[0]) / 30),
        details={
            "l1_l2_ratio": float(l1_l2_ratio),
            "active_fraction": float(active_fraction),
            "gini_coefficient": float(gini),
            "top_10_contribution": float(top_contribution),
            "top_indices": top_indices.tolist(),
        }
    )


def _detect_bimodal_structure(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    diff_vectors: torch.Tensor,
    cfg: GeometryAnalysisConfig,
) -> StructureScore:
    """Detect if activations have bimodal/multimodal distribution."""
    # Project onto principal direction
    mean_diff = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
    direction = F.normalize(mean_diff, p=2, dim=0)
    
    all_activations = torch.cat([pos_tensor, neg_tensor], dim=0)
    projections = (all_activations @ direction).cpu()
    
    # Hartigan's dip test approximation
    dip_statistic = _compute_dip_statistic(projections)
    
    # Check separation between pos and neg projections
    pos_proj = (pos_tensor @ direction)
    neg_proj = (neg_tensor @ direction)
    
    # Overlap between distributions
    pos_mean, pos_std = pos_proj.mean(), pos_proj.std()
    neg_mean, neg_std = neg_proj.mean(), neg_proj.std()
    
    # Bhattacharyya distance approximation
    if pos_std > 0 and neg_std > 0:
        mean_diff_normalized = abs(pos_mean - neg_mean) / ((pos_std + neg_std) / 2)
    else:
        mean_diff_normalized = 0.0
    
    # Bimodal score: high dip + clear separation
    bimodal_score = 0.5 * min(float(dip_statistic) * 10, 1.0) + 0.5 * min(float(mean_diff_normalized) / 3, 1.0)
    
    return StructureScore(
        StructureType.BIMODAL,
        score=float(bimodal_score),
        confidence=min(1.0, len(projections) / 50),
        details={
            "dip_statistic": float(dip_statistic),
            "mean_separation": float(mean_diff_normalized),
            "pos_mean": float(pos_mean),
            "neg_mean": float(neg_mean),
            "pos_std": float(pos_std),
            "neg_std": float(neg_std),
        }
    )


def _compute_dip_statistic(data: torch.Tensor) -> float:
    """Compute Hartigan's dip statistic (simplified)."""
    sorted_data = data.sort().values
    n = len(sorted_data)
    
    if n < 4:
        return 0.0
    
    # Empirical CDF
    ecdf = torch.arange(1, n + 1, dtype=torch.float32) / n
    
    # Greatest convex minorant and least concave majorant
    # Simplified: measure deviation from uniform
    uniform = torch.linspace(0, 1, n)
    
    # Kolmogorov-Smirnov like statistic
    ks_stat = (ecdf - uniform).abs().max()
    
    return float(ks_stat)


def _detect_orthogonal_structure(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    diff_vectors: torch.Tensor,
    cfg: GeometryAnalysisConfig,
) -> StructureScore:
    """Detect if behavior is encoded in multiple orthogonal/independent subspaces.
    
    Orthogonal structure means the data requires MULTIPLE independent directions
    that are NOT correlated with each other. This is different from cone (where
    directions are correlated) and linear (where one direction suffices).
    """
    if diff_vectors.shape[0] < cfg.num_components:
        return StructureScore(StructureType.ORTHOGONAL, 0.0, 0.0, {"reason": "insufficient_data"})
    
    try:
        # PCA to understand variance distribution
        centered = diff_vectors - diff_vectors.mean(dim=0, keepdim=True)
        _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        
        total_var = (S ** 2).sum()
        if total_var < 1e-8:
            return StructureScore(StructureType.ORTHOGONAL, 0.0, 0.0, {"reason": "no_variance"})
        
        # For orthogonal structure:
        # 1. Multiple components should have significant variance (not just one = linear)
        # 2. Variance should be spread across multiple dimensions (not concentrated)
        
        var_explained = (S ** 2) / total_var
        k = min(cfg.num_components, len(S))
        
        # First component dominance (low = more orthogonal/spread)
        first_var = float(var_explained[0])
        
        # Effective dimensionality (entropy-based)
        var_explained_clipped = var_explained[var_explained > 1e-10]
        entropy = -(var_explained_clipped * torch.log(var_explained_clipped + 1e-10)).sum()
        max_entropy = torch.log(torch.tensor(float(len(var_explained_clipped))))
        effective_dim_ratio = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        
        # Count significant dimensions (>5% variance each)
        significant_dims = (var_explained > 0.05).sum().item()
        multi_dim_score = min(significant_dims / 3, 1.0)  # 3+ significant dims is fully orthogonal
        
        # Orthogonal structure is RARE and specific:
        # It requires MULTIPLE INDEPENDENT directions with separation on EACH
        # High spread alone is not orthogonal - it could be noise or cone
        
        # Check separation strength
        mean_diff = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
        separation_strength = mean_diff.norm() / (pos_tensor.std() + neg_tensor.std() + 1e-8)
        has_separation = min(float(separation_strength) / 3, 1.0)
        
        # For true orthogonal structure, we need:
        # 1. Strong separation (otherwise no structure)
        # 2. Multiple significant dimensions (otherwise linear)
        # 3. But NOT too spread (otherwise just noise)
        
        # Sweet spot: 2-4 significant dimensions with clear separation
        if significant_dims < 2:
            # Too few dimensions = linear
            orthogonal_score = 0.2
        elif significant_dims > 10:
            # Too many = likely noise, not structure
            orthogonal_score = 0.3 * has_separation
        else:
            # Reasonable number of dimensions
            # Check if it's not dominated by first (would be linear)
            # and not too spread (would be noise)
            structure_score = (
                0.3 * (1 - first_var) +  # Not dominated by one direction
                0.3 * min(significant_dims / 4, 1.0) +  # 2-4 directions is ideal
                0.4 * has_separation  # Must have separation
            )
            orthogonal_score = structure_score * 0.8  # Scale down - orthogonal is rare
        
        return StructureScore(
            StructureType.ORTHOGONAL,
            score=float(orthogonal_score),
            confidence=min(1.0, diff_vectors.shape[0] / 30),
            details={
                "first_component_variance": float(first_var),
                "effective_dim_ratio": float(effective_dim_ratio),
                "significant_dimensions": int(significant_dims),
                "top_5_variances": var_explained[:min(5, len(var_explained))].tolist(),
            }
        )
    except Exception as e:
        return StructureScore(StructureType.ORTHOGONAL, 0.0, 0.0, {"error": str(e)})


def _generate_recommendation(best_structure: StructureType, all_scores: Dict[str, StructureScore]) -> str:
    """Generate steering method recommendation based on detected geometry."""
    recommendations = {
        StructureType.LINEAR: "Use CAA (Contrastive Activation Addition) - single direction steering is optimal.",
        StructureType.CONE: "Use PRISM - multi-directional steering will capture the full behavior cone.",
        StructureType.CLUSTER: "Consider cluster-based steering or multiple separate vectors for each cluster.",
        StructureType.MANIFOLD: "Use TITAN with learned gating - non-linear structure requires adaptive steering.",
        StructureType.SPARSE: "Use SAE-based steering targeting the specific active neurons.",
        StructureType.BIMODAL: "Use PULSE with conditional gating - behavior has two distinct modes.",
        StructureType.ORTHOGONAL: "Use multiple independent CAA vectors or ICA-based steering.",
        StructureType.UNKNOWN: "Structure unclear - start with CAA and evaluate effectiveness.",
    }
    
    base_rec = recommendations.get(best_structure, recommendations[StructureType.UNKNOWN])
    
    # Add context from other scores
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1].score, reverse=True)
    if len(sorted_scores) >= 2:
        second_best = sorted_scores[1]
        if second_best[1].score > 0.6:
            base_rec += f" (Also consider {second_best[0]}: score {second_best[1].score:.2f})"
    
    return base_rec