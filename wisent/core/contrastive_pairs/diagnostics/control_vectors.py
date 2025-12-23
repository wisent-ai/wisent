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
    "MultiLayerGeometryConfig",
    "MultiLayerGeometryResult",
    "LayerGeometryResult",
    "detect_geometry_multi_layer",
    "detect_geometry_all_layers",
    "ExhaustiveCombinationResult",
    "ExhaustiveGeometryAnalysisResult",
    "detect_geometry_exhaustive",
    "detect_geometry_limited",
    "detect_geometry_contiguous",
    "detect_geometry_smart",
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

        flat = detached.to(device="cpu").reshape(-1)

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
    
    # Use raw scores directly - no penalization
    # The recommendation logic will handle specificity (prefer simpler structures when they fit)
    all_scores = raw_scores
    
    # Find MOST SPECIFIC structure that fits well
    # Specificity order: linear > cone > orthogonal > cluster > sparse > bimodal > manifold
    best_structure, best_score = _find_most_specific_structure(all_scores)
    
    # Generate recommendation based on specificity
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
        }
    )


def _find_most_specific_structure(scores: Dict[str, StructureScore]) -> Tuple[StructureType, float]:
    """
    Find the most specific structure that fits the data well.
    
    Specificity order (most to least specific):
    1. Linear - single direction (most specific)
    2. Cone - correlated directions 
    3. Orthogonal - uncorrelated directions
    4. Cluster - discrete groups
    5. Sparse - few active neurons
    6. Bimodal - two modes
    7. Manifold - any continuous structure (least specific, always fits)
    
    We pick the MOST SPECIFIC structure that exceeds its threshold.
    More specific = more constrained = more informative about the data.
    """
    # Thresholds for "this structure fits well enough"
    THRESHOLDS = {
        "linear": 0.5,      # 1D separable
        "cone": 0.5,        # correlated directions
        "orthogonal": 0.5,  # independent directions
        "cluster": 0.6,     # discrete groups
        "sparse": 0.7,      # few active neurons
        "bimodal": 0.5,     # two-mode distribution
        "manifold": 0.3,    # fallback (always fits)
    }
    
    # Check in specificity order
    specificity_order = ["linear", "cone", "orthogonal", "cluster", "sparse", "bimodal", "manifold"]
    
    for struct_name in specificity_order:
        if struct_name in scores:
            score = scores[struct_name].score
            threshold = THRESHOLDS.get(struct_name, 0.5)
            if score >= threshold:
                return scores[struct_name].structure_type, score
    
    # Fallback: return highest scoring structure
    best_key = max(scores.keys(), key=lambda k: scores[k].score)
    return scores[best_key].structure_type, scores[best_key].score


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
    """Detect cone structure using RAW cosine similarity of difference vectors.
    
    A cone structure means:
    - Multiple difference vectors (pos_i - neg_i) point in SIMILAR directions
    - High cosine similarity between raw difference vectors
    - NOT using gradient-optimized directions (which inflate the score)
    
    This matches what the visualization computes.
    """
    try:
        # Compute raw difference vectors (what visualization uses)
        n_pairs = min(pos_tensor.shape[0], neg_tensor.shape[0])
        if n_pairs < 3:
            return StructureScore(StructureType.CONE, 0.0, 0.0, {"reason": "insufficient_pairs"})
        
        diff_vectors = pos_tensor[:n_pairs] - neg_tensor[:n_pairs]
        
        # Normalize difference vectors
        norms = diff_vectors.norm(dim=1, keepdim=True)
        valid_mask = (norms.squeeze() > 1e-8)
        if valid_mask.sum() < 3:
            return StructureScore(StructureType.CONE, 0.0, 0.0, {"reason": "zero_differences"})
        
        diff_normalized = diff_vectors[valid_mask] / norms[valid_mask]
        
        # Compute pairwise cosine similarity matrix
        cos_sim_matrix = diff_normalized @ diff_normalized.T
        
        # Get off-diagonal elements (exclude self-similarity of 1.0)
        n = cos_sim_matrix.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=cos_sim_matrix.device)
        off_diagonal = cos_sim_matrix[mask]
        
        # Raw cosine similarity statistics
        mean_cos_sim = float(off_diagonal.mean())
        std_cos_sim = float(off_diagonal.std())
        min_cos_sim = float(off_diagonal.min())
        max_cos_sim = float(off_diagonal.max())
        
        # Fraction of pairs with positive correlation (same half-space)
        positive_fraction = float((off_diagonal > 0).float().mean())
        
        # Fraction with strong correlation (>0.3)
        strong_fraction = float((off_diagonal > 0.3).float().mean())
        
        # Cone score based on raw cosine similarity:
        # - High mean cosine = directions are aligned = cone
        # - Low mean cosine = directions are independent = NOT cone
        # - Negative mean cosine = directions are opposing = NOT cone
        
        if mean_cos_sim < 0:
            # Negative correlation = definitely not a cone
            cone_score = 0.0
        elif mean_cos_sim < 0.1:
            # Near zero = orthogonal/independent, not cone
            cone_score = mean_cos_sim  # 0.0 - 0.1
        elif mean_cos_sim < 0.3:
            # Weak correlation = weak cone
            cone_score = 0.1 + 0.2 * ((mean_cos_sim - 0.1) / 0.2)  # 0.1 - 0.3
        elif mean_cos_sim < 0.7:
            # Moderate correlation = good cone (ideal range)
            cone_score = 0.3 + 0.5 * ((mean_cos_sim - 0.3) / 0.4)  # 0.3 - 0.8
        else:
            # Very high correlation = almost linear, still cone-like
            cone_score = 0.8 + 0.2 * ((mean_cos_sim - 0.7) / 0.3)  # 0.8 - 1.0
        
        # Confidence based on consistency (low std = more consistent = higher confidence)
        consistency = max(0, 1 - std_cos_sim)
        confidence = consistency * min(1.0, n_pairs / 20)
        
        return StructureScore(
            StructureType.CONE,
            score=float(cone_score),
            confidence=float(confidence),
            details={
                "raw_mean_cosine_similarity": mean_cos_sim,
                "raw_std_cosine_similarity": std_cos_sim,
                "raw_min_cosine_similarity": min_cos_sim,
                "raw_max_cosine_similarity": max_cos_sim,
                "positive_correlation_fraction": positive_fraction,
                "strong_correlation_fraction": strong_fraction,
                "n_valid_pairs": int(valid_mask.sum()),
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
    """Detect if activations form discrete clusters.
    
    Cluster structure means:
    - Data forms DISCRETE, SEPARATED groups
    - Not just "pos vs neg" (that's trivially 2 clusters)
    - Actual subgroups within the data
    
    Key insight: k-means will ALWAYS find clusters.
    We need high silhouette AND clear separation to claim clusters.
    Also, if pos/neg perfectly separate, that's "linear", not "cluster".
    """
    all_activations = torch.cat([pos_tensor, neg_tensor], dim=0)
    n_samples = all_activations.shape[0]
    
    if n_samples < 6:
        return StructureScore(StructureType.CLUSTER, 0.0, 0.0, {"reason": "insufficient_data"})
    
    best_silhouette = -1.0
    best_k = 2
    silhouette_scores = {}
    
    for k in range(2, min(cfg.max_clusters + 1, n_samples // 2)):
        try:
            labels, centroids, silhouette = _kmeans_with_silhouette(all_activations, k, max_iters=50)
            silhouette_scores[k] = silhouette
            
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
        except Exception:
            continue
    
    if best_silhouette < 0:
        return StructureScore(StructureType.CLUSTER, 0.0, 0.0, {"reason": "clustering_failed"})
    
    # Check if clusters just separate pos/neg (that's linear, not cluster)
    labels, _, _ = _kmeans_with_silhouette(all_activations, best_k, max_iters=50)
    pos_labels = labels[:pos_tensor.shape[0]]
    neg_labels = labels[pos_tensor.shape[0]:]
    
    # If k=2 and it perfectly separates pos/neg, that's LINEAR not cluster
    if best_k == 2:
        pos_mode = pos_labels.mode().values.item() if len(pos_labels) > 0 else -1
        neg_mode = neg_labels.mode().values.item() if len(neg_labels) > 0 else -1
        pos_purity = (pos_labels == pos_mode).float().mean()
        neg_purity = (neg_labels == neg_mode).float().mean()
        
        if pos_mode != neg_mode and pos_purity > 0.8 and neg_purity > 0.8:
            # Perfect pos/neg separation - this is LINEAR, not cluster
            return StructureScore(
                StructureType.CLUSTER,
                score=0.1,  # Low score - it's actually linear
                confidence=0.8,
                details={
                    "reason": "pos_neg_separation_is_linear",
                    "best_k": 2,
                    "pos_purity": float(pos_purity),
                    "neg_purity": float(neg_purity),
                }
            )
    
    # For true cluster structure, we need:
    # 1. High silhouette (> 0.5 is good, > 0.7 is strong)
    # 2. k > 2 OR k=2 with mixed clusters
    
    # Silhouette thresholds - be strict
    if best_silhouette < 0.4:
        # Very low silhouette = no clear cluster structure
        cluster_score = best_silhouette * 0.3  # Very low score
    elif best_silhouette < cfg.cluster_silhouette_threshold:
        # Moderate silhouette = weak cluster structure
        cluster_score = 0.1 + 0.2 * (best_silhouette / cfg.cluster_silhouette_threshold)
    else:
        # High silhouette = good cluster structure
        # But only if it's not just pos/neg separation
        base_score = 0.3 + 0.5 * ((best_silhouette - cfg.cluster_silhouette_threshold) / (1 - cfg.cluster_silhouette_threshold))
        
        # Bonus for k > 2 (more interesting structure)
        if best_k > 2:
            cluster_score = base_score + 0.2
        else:
            cluster_score = base_score
    
    cluster_score = min(1.0, cluster_score)
    
    return StructureScore(
        StructureType.CLUSTER,
        score=float(cluster_score),
        confidence=float(max(0, best_silhouette)),
        details={
            "best_k": best_k,
            "best_silhouette": float(best_silhouette),
            "all_silhouettes": {str(k): float(v) for k, v in silhouette_scores.items()},
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
    """Detect non-linear manifold structure.
    
    Manifold structure means:
    - Data lies on a CURVED surface (not linear)
    - Linear methods (PCA, CAA) cannot capture the structure
    - Requires non-linear methods (TITAN, neural steering)
    
    Key insight: Manifold should be a FALLBACK, not default.
    Only report manifold if:
    1. Linear doesn't work (PCA explains little variance)
    2. There's actual curvature (local neighborhoods don't align)
    3. BUT there IS structure (not just noise)
    """
    all_activations = torch.cat([pos_tensor, neg_tensor], dim=0)
    n_samples = all_activations.shape[0]
    
    if n_samples < cfg.manifold_neighbors + 1:
        return StructureScore(StructureType.MANIFOLD, 0.0, 0.0, {"reason": "insufficient_data"})
    
    try:
        # 1. Check if linear works well (if yes, not manifold)
        centered = all_activations - all_activations.mean(dim=0, keepdim=True)
        try:
            _, S, _ = torch.linalg.svd(centered, full_matrices=False)
            total_var = (S ** 2).sum()
            if total_var > 0:
                # Top 2 PCs variance explained
                top2_var = (S[:2] ** 2).sum() / total_var
                linear_explains_well = float(top2_var) > 0.7
            else:
                linear_explains_well = True  # No variance = trivial
        except Exception:
            linear_explains_well = False
            top2_var = torch.tensor(0.0)
        
        if linear_explains_well:
            # Linear works well - not a manifold (it's linear)
            return StructureScore(
                StructureType.MANIFOLD, 
                score=0.1, 
                confidence=0.8,
                details={
                    "reason": "linear_sufficient",
                    "pca_top2_variance": float(top2_var),
                }
            )
        
        # 2. Check for actual curvature (local PCA directions vary)
        local_nonlinearity = _compute_local_nonlinearity(all_activations, cfg.manifold_neighbors)
        
        # 3. Check if there's meaningful structure (separation between pos/neg)
        mean_diff = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
        separation_strength = mean_diff.norm() / (pos_tensor.std() + neg_tensor.std() + 1e-8)
        has_structure = min(float(separation_strength) / 2, 1.0)
        
        if has_structure < 0.3:
            # No clear structure - likely noise, not manifold
            return StructureScore(
                StructureType.MANIFOLD,
                score=0.2,
                confidence=0.5,
                details={
                    "reason": "weak_structure",
                    "separation_strength": float(separation_strength),
                }
            )
        
        # 4. Manifold requires BOTH:
        #    - Linear doesn't work (already checked)
        #    - AND there's curvature
        #    - AND there's structure
        
        # If nonlinearity is low, it might be orthogonal/independent, not curved
        if local_nonlinearity < 0.3:
            manifold_score = 0.3 * has_structure  # Low score
        else:
            # High nonlinearity + structure = manifold candidate
            manifold_score = (
                0.30 * local_nonlinearity +
                0.30 * (1 - float(top2_var)) +  # Reward when linear fails
                0.40 * has_structure
            )
        
        # Confidence based on sample size and consistency
        confidence = min(1.0, n_samples / 100) * has_structure
        
        return StructureScore(
            StructureType.MANIFOLD,
            score=float(manifold_score),
            confidence=float(confidence),
            details={
                "pca_top2_variance": float(top2_var),
                "local_nonlinearity": float(local_nonlinearity),
                "separation_strength": float(separation_strength),
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
    gini = (2 * torch.arange(1, n + 1, dtype=sorted_abs.dtype, device=sorted_abs.device) @ sorted_abs - (n + 1) * sorted_abs.sum()) / (n * sorted_abs.sum() + 1e-10)
    
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
    ecdf = torch.arange(1, n + 1, dtype=sorted_data.dtype, device=sorted_data.device) / n
    
    # Greatest convex minorant and least concave majorant
    # Simplified: measure deviation from uniform
    uniform = torch.linspace(0, 1, n, dtype=sorted_data.dtype, device=sorted_data.device)
    
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
    
    Orthogonal structure means:
    - Multiple difference vectors point in INDEPENDENT directions
    - Low cosine similarity between difference vectors (near 0)
    - NOT correlated (that's cone) and NOT single direction (that's linear)
    
    This is the OPPOSITE of cone - if cosine sim is low, it's orthogonal.
    Uses raw cosine similarity like the cone detector for consistency.
    """
    try:
        # Compute raw difference vectors (same as cone detector)
        n_pairs = min(pos_tensor.shape[0], neg_tensor.shape[0])
        if n_pairs < 3:
            return StructureScore(StructureType.ORTHOGONAL, 0.0, 0.0, {"reason": "insufficient_pairs"})
        
        diff_vectors_raw = pos_tensor[:n_pairs] - neg_tensor[:n_pairs]
        
        # Normalize difference vectors
        norms = diff_vectors_raw.norm(dim=1, keepdim=True)
        valid_mask = (norms.squeeze() > 1e-8)
        if valid_mask.sum() < 3:
            return StructureScore(StructureType.ORTHOGONAL, 0.0, 0.0, {"reason": "zero_differences"})
        
        diff_normalized = diff_vectors_raw[valid_mask] / norms[valid_mask]
        
        # Compute pairwise cosine similarity matrix
        cos_sim_matrix = diff_normalized @ diff_normalized.T
        
        # Get off-diagonal elements
        n = cos_sim_matrix.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=cos_sim_matrix.device)
        off_diagonal = cos_sim_matrix[mask]
        
        # Raw cosine similarity statistics
        mean_cos_sim = float(off_diagonal.mean())
        std_cos_sim = float(off_diagonal.std())
        abs_mean_cos_sim = float(off_diagonal.abs().mean())
        
        # Fraction near zero (truly orthogonal)
        near_zero_fraction = float((off_diagonal.abs() < 0.2).float().mean())
        
        # Orthogonal = LOW cosine similarity (opposite of cone)
        # Ideal orthogonal: mean cosine sim near 0, low absolute mean
        
        if abs_mean_cos_sim < 0.1:
            # Very low correlation = strong orthogonal
            orthogonal_score = 0.8 + 0.2 * (1 - abs_mean_cos_sim / 0.1)
        elif abs_mean_cos_sim < 0.2:
            # Low correlation = moderate orthogonal
            orthogonal_score = 0.5 + 0.3 * (1 - (abs_mean_cos_sim - 0.1) / 0.1)
        elif abs_mean_cos_sim < 0.4:
            # Moderate correlation = weak orthogonal
            orthogonal_score = 0.2 + 0.3 * (1 - (abs_mean_cos_sim - 0.2) / 0.2)
        else:
            # High correlation = not orthogonal (probably cone or linear)
            orthogonal_score = max(0, 0.2 * (1 - (abs_mean_cos_sim - 0.4) / 0.6))
        
        # Check if there's meaningful separation (not just noise)
        mean_diff = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
        separation_strength = mean_diff.norm() / (pos_tensor.std() + neg_tensor.std() + 1e-8)
        has_separation = min(float(separation_strength) / 2, 1.0)
        
        # Orthogonal without separation is just noise
        if has_separation < 0.3:
            orthogonal_score *= 0.3  # Heavy penalty
        
        # Confidence based on consistency and sample size
        confidence = near_zero_fraction * min(1.0, n_pairs / 20)
        
        return StructureScore(
            StructureType.ORTHOGONAL,
            score=float(orthogonal_score),
            confidence=float(confidence),
            details={
                "raw_mean_cosine_similarity": mean_cos_sim,
                "raw_abs_mean_cosine_similarity": abs_mean_cos_sim,
                "raw_std_cosine_similarity": std_cos_sim,
                "near_zero_fraction": near_zero_fraction,
                "separation_strength": float(separation_strength),
                "n_valid_pairs": int(valid_mask.sum()),
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


# =============================================================================
# Multi-Layer Geometry Analysis
# =============================================================================

@dataclass
class MultiLayerGeometryConfig:
    """Configuration for multi-layer geometry analysis."""
    
    num_components: int = 5
    optimization_steps: int = 50
    combination_method: str = "concat"  # "concat", "mean", "weighted"
    analyze_per_layer: bool = True
    analyze_combined: bool = True
    analyze_subsets: bool = True  # early/middle/late
    analyze_pairs: bool = True  # all pairs of layers
    analyze_adjacent: bool = True  # adjacent layer pairs
    analyze_skip: bool = True  # every other layer, every third, etc.
    analyze_custom: Optional[List[List[int]]] = None  # custom layer combinations
    max_pair_combinations: int = 50  # limit number of pair combinations to analyze


@dataclass
class LayerGeometryResult:
    """Geometry result for a single layer."""
    layer: int
    best_structure: StructureType
    best_score: float
    all_scores: Dict[str, float]


@dataclass
class MultiLayerGeometryResult:
    """Results from multi-layer geometry analysis."""
    
    per_layer_results: Dict[int, LayerGeometryResult]
    """Geometry analysis for each individual layer."""
    
    combined_result: Optional[GeometryAnalysisResult]
    """Geometry analysis for all layers combined."""
    
    layer_subset_results: Dict[str, GeometryAnalysisResult]
    """Geometry analysis for layer subsets (e.g., 'early', 'middle', 'late')."""
    
    layer_pair_results: Dict[str, GeometryAnalysisResult]
    """Geometry analysis for pairs of layers (e.g., 'L1+L5', 'L2+L8')."""
    
    adjacent_pair_results: Dict[str, GeometryAnalysisResult]
    """Geometry analysis for adjacent layer pairs (e.g., 'L1+L2', 'L2+L3')."""
    
    skip_results: Dict[str, GeometryAnalysisResult]
    """Geometry analysis for skip patterns (e.g., 'every_2nd', 'every_3rd')."""
    
    custom_results: Dict[str, GeometryAnalysisResult]
    """Geometry analysis for custom layer combinations."""
    
    best_single_layer: int
    """Layer with strongest structure detection."""
    
    best_single_layer_structure: StructureType
    """Structure type detected at best single layer."""
    
    best_single_layer_score: float
    """Score at best single layer."""
    
    best_combination: Optional[str]
    """Best performing layer combination (if better than single layer)."""
    
    best_combination_score: float
    """Score of best combination."""
    
    best_combination_structure: Optional[StructureType]
    """Structure type detected at best combination."""
    
    combined_vs_single: str
    """Whether combined layers improve over single layer."""
    
    layer_agreement: float
    """How much layers agree on structure type (0-1)."""
    
    structure_by_depth: Dict[str, List[float]]
    """How each structure score varies by layer depth."""
    
    all_combinations_ranked: List[Tuple[str, float, StructureType]]
    """All combinations ranked by score: (name, score, structure)."""
    
    recommendation: str
    """Recommendation based on multi-layer analysis."""


def detect_geometry_multi_layer(
    pos_activations_by_layer: Dict[int, torch.Tensor],
    neg_activations_by_layer: Dict[int, torch.Tensor],
    config: MultiLayerGeometryConfig | None = None,
) -> MultiLayerGeometryResult:
    """
    Detect geometric structure across multiple layers.
    
    Analyzes:
    1. Each layer individually
    2. All layers combined (concatenated or aggregated)
    3. Layer subsets (early, middle, late)
    4. Layer pairs (all combinations of 2 layers)
    5. Adjacent layer pairs (L1+L2, L2+L3, etc.)
    6. Skip patterns (every 2nd, every 3rd layer)
    7. Custom layer combinations
    8. How structure varies by depth
    
    Arguments:
        pos_activations_by_layer: Dict mapping layer index to positive activations [N, hidden_dim]
        neg_activations_by_layer: Dict mapping layer index to negative activations [N, hidden_dim]
        config: Analysis configuration
        
    Returns:
        MultiLayerGeometryResult with comprehensive multi-layer analysis
    """
    cfg = config or MultiLayerGeometryConfig()
    geo_cfg = GeometryAnalysisConfig(num_components=cfg.num_components, optimization_steps=cfg.optimization_steps)
    
    layers = sorted(pos_activations_by_layer.keys())
    if not layers:
        raise ValueError("No layers provided")
    
    # Track all combination results for ranking
    all_combo_results: Dict[str, GeometryAnalysisResult] = {}
    
    # 1. Analyze each layer individually
    per_layer_results: Dict[int, LayerGeometryResult] = {}
    structure_by_depth: Dict[str, List[float]] = {
        "linear": [], "cone": [], "cluster": [], "manifold": [],
        "sparse": [], "bimodal": [], "orthogonal": []
    }
    
    if cfg.analyze_per_layer:
        for layer in layers:
            pos_acts = pos_activations_by_layer[layer]
            neg_acts = neg_activations_by_layer[layer]
            
            result = detect_geometry_structure(pos_acts, neg_acts, geo_cfg)
            
            all_scores = {name: score.score for name, score in result.all_scores.items()}
            per_layer_results[layer] = LayerGeometryResult(
                layer=layer,
                best_structure=result.best_structure,
                best_score=result.best_score,
                all_scores=all_scores,
            )
            all_combo_results[f"L{layer}"] = result
            
            for struct_name, score in all_scores.items():
                if struct_name in structure_by_depth:
                    structure_by_depth[struct_name].append(score)
    
    # 2. Find best single layer
    if per_layer_results:
        best_layer = max(per_layer_results.keys(), key=lambda l: per_layer_results[l].best_score)
        best_single_layer = best_layer
        best_single_layer_structure = per_layer_results[best_layer].best_structure
        best_single_layer_score = per_layer_results[best_layer].best_score
    else:
        best_single_layer = layers[0]
        best_single_layer_structure = StructureType.UNKNOWN
        best_single_layer_score = 0.0
    
    # 3. Analyze all layers combined
    combined_result = None
    if cfg.analyze_combined and len(layers) > 1:
        combined_pos, combined_neg = _combine_layer_activations(
            pos_activations_by_layer, neg_activations_by_layer, layers, cfg.combination_method
        )
        combined_result = detect_geometry_structure(combined_pos, combined_neg, geo_cfg)
        all_combo_results["all_layers"] = combined_result
    
    # 4. Analyze layer subsets (early, middle, late)
    layer_subset_results: Dict[str, GeometryAnalysisResult] = {}
    if cfg.analyze_subsets and len(layers) >= 3:
        n_layers = len(layers)
        third = n_layers // 3
        
        early_layers = layers[:third] if third > 0 else layers[:1]
        middle_layers = layers[third:2*third] if third > 0 else layers[1:2]
        late_layers = layers[2*third:] if third > 0 else layers[-1:]
        
        # Also add first_half and second_half
        half = n_layers // 2
        first_half = layers[:half] if half > 0 else layers[:1]
        second_half = layers[half:] if half > 0 else layers[-1:]
        
        subsets = [
            ("early", early_layers),
            ("middle", middle_layers),
            ("late", late_layers),
            ("first_half", first_half),
            ("second_half", second_half),
        ]
        
        for subset_name, subset_layers in subsets:
            if len(subset_layers) >= 1:
                subset_pos, subset_neg = _combine_layer_activations(
                    pos_activations_by_layer, neg_activations_by_layer, subset_layers, cfg.combination_method
                )
                result = detect_geometry_structure(subset_pos, subset_neg, geo_cfg)
                layer_subset_results[subset_name] = result
                all_combo_results[subset_name] = result
    
    # 5. Analyze layer pairs
    layer_pair_results: Dict[str, GeometryAnalysisResult] = {}
    if cfg.analyze_pairs and len(layers) >= 2:
        from itertools import combinations
        pair_count = 0
        for l1, l2 in combinations(layers, 2):
            if pair_count >= cfg.max_pair_combinations:
                break
            pair_name = f"L{l1}+L{l2}"
            pair_pos, pair_neg = _combine_layer_activations(
                pos_activations_by_layer, neg_activations_by_layer, [l1, l2], cfg.combination_method
            )
            result = detect_geometry_structure(pair_pos, pair_neg, geo_cfg)
            layer_pair_results[pair_name] = result
            all_combo_results[pair_name] = result
            pair_count += 1
    
    # 6. Analyze adjacent layer pairs
    adjacent_pair_results: Dict[str, GeometryAnalysisResult] = {}
    if cfg.analyze_adjacent and len(layers) >= 2:
        for i in range(len(layers) - 1):
            l1, l2 = layers[i], layers[i + 1]
            pair_name = f"adj_L{l1}+L{l2}"
            pair_pos, pair_neg = _combine_layer_activations(
                pos_activations_by_layer, neg_activations_by_layer, [l1, l2], cfg.combination_method
            )
            result = detect_geometry_structure(pair_pos, pair_neg, geo_cfg)
            adjacent_pair_results[pair_name] = result
            all_combo_results[pair_name] = result
    
    # 7. Analyze skip patterns
    skip_results: Dict[str, GeometryAnalysisResult] = {}
    if cfg.analyze_skip and len(layers) >= 4:
        # Every 2nd layer
        every_2nd = layers[::2]
        if len(every_2nd) >= 2:
            skip_pos, skip_neg = _combine_layer_activations(
                pos_activations_by_layer, neg_activations_by_layer, every_2nd, cfg.combination_method
            )
            result = detect_geometry_structure(skip_pos, skip_neg, geo_cfg)
            skip_results["every_2nd"] = result
            all_combo_results["every_2nd"] = result
        
        # Every 3rd layer
        if len(layers) >= 6:
            every_3rd = layers[::3]
            if len(every_3rd) >= 2:
                skip_pos, skip_neg = _combine_layer_activations(
                    pos_activations_by_layer, neg_activations_by_layer, every_3rd, cfg.combination_method
                )
                result = detect_geometry_structure(skip_pos, skip_neg, geo_cfg)
                skip_results["every_3rd"] = result
                all_combo_results["every_3rd"] = result
        
        # First and last layer only
        first_last = [layers[0], layers[-1]]
        skip_pos, skip_neg = _combine_layer_activations(
            pos_activations_by_layer, neg_activations_by_layer, first_last, cfg.combination_method
        )
        result = detect_geometry_structure(skip_pos, skip_neg, geo_cfg)
        skip_results["first_last"] = result
        all_combo_results["first_last"] = result
        
        # First, middle, last
        if len(layers) >= 3:
            mid_idx = len(layers) // 2
            first_mid_last = [layers[0], layers[mid_idx], layers[-1]]
            skip_pos, skip_neg = _combine_layer_activations(
                pos_activations_by_layer, neg_activations_by_layer, first_mid_last, cfg.combination_method
            )
            result = detect_geometry_structure(skip_pos, skip_neg, geo_cfg)
            skip_results["first_mid_last"] = result
            all_combo_results["first_mid_last"] = result
    
    # 8. Analyze custom combinations
    custom_results: Dict[str, GeometryAnalysisResult] = {}
    if cfg.analyze_custom:
        for i, custom_layers in enumerate(cfg.analyze_custom):
            valid_layers = [l for l in custom_layers if l in layers]
            if len(valid_layers) >= 1:
                custom_name = f"custom_{i}_L" + "+L".join(map(str, valid_layers))
                custom_pos, custom_neg = _combine_layer_activations(
                    pos_activations_by_layer, neg_activations_by_layer, valid_layers, cfg.combination_method
                )
                result = detect_geometry_structure(custom_pos, custom_neg, geo_cfg)
                custom_results[custom_name] = result
                all_combo_results[custom_name] = result
    
    # 9. Compute layer agreement
    if per_layer_results:
        structures = [r.best_structure for r in per_layer_results.values()]
        most_common = max(set(structures), key=structures.count)
        layer_agreement = structures.count(most_common) / len(structures)
    else:
        layer_agreement = 0.0
    
    # 10. Rank all combinations and find best
    all_combinations_ranked = sorted(
        [(name, r.best_score, r.best_structure) for name, r in all_combo_results.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    if all_combinations_ranked:
        best_combo_name, best_combo_score, best_combo_structure = all_combinations_ranked[0]
        if best_combo_score > best_single_layer_score:
            best_combination = best_combo_name
            best_combination_score = best_combo_score
            best_combination_structure = best_combo_structure
        else:
            best_combination = None
            best_combination_score = best_single_layer_score
            best_combination_structure = best_single_layer_structure
    else:
        best_combination = None
        best_combination_score = best_single_layer_score
        best_combination_structure = best_single_layer_structure
    
    # 11. Compare combined vs single
    if combined_result and per_layer_results:
        if combined_result.best_score > best_single_layer_score + 0.1:
            combined_vs_single = f"Combined ({combined_result.best_score:.2f}) better than single layer ({best_single_layer_score:.2f})"
        elif best_single_layer_score > combined_result.best_score + 0.1:
            combined_vs_single = f"Single layer {best_single_layer} ({best_single_layer_score:.2f}) better than combined ({combined_result.best_score:.2f})"
        else:
            combined_vs_single = f"Similar performance: combined={combined_result.best_score:.2f}, single={best_single_layer_score:.2f}"
    else:
        combined_vs_single = "No comparison available"
    
    # 12. Generate recommendation
    recommendation = _generate_multi_layer_recommendation_v2(
        per_layer_results, combined_result, layer_subset_results,
        layer_pair_results, skip_results,
        best_single_layer, best_single_layer_structure, best_single_layer_score,
        best_combination, best_combination_score, best_combination_structure,
        layer_agreement, all_combinations_ranked
    )
    
    return MultiLayerGeometryResult(
        per_layer_results=per_layer_results,
        combined_result=combined_result,
        layer_subset_results=layer_subset_results,
        layer_pair_results=layer_pair_results,
        adjacent_pair_results=adjacent_pair_results,
        skip_results=skip_results,
        custom_results=custom_results,
        best_single_layer=best_single_layer,
        best_single_layer_structure=best_single_layer_structure,
        best_single_layer_score=best_single_layer_score,
        best_combination=best_combination,
        best_combination_score=best_combination_score,
        best_combination_structure=best_combination_structure,
        combined_vs_single=combined_vs_single,
        layer_agreement=layer_agreement,
        structure_by_depth=structure_by_depth,
        all_combinations_ranked=all_combinations_ranked,
        recommendation=recommendation,
    )


def _combine_layer_activations(
    pos_by_layer: Dict[int, torch.Tensor],
    neg_by_layer: Dict[int, torch.Tensor],
    layers: List[int],
    method: str = "concat",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine activations from multiple layers."""
    pos_acts = [pos_by_layer[l] for l in layers if l in pos_by_layer]
    neg_acts = [neg_by_layer[l] for l in layers if l in neg_by_layer]
    
    if not pos_acts or not neg_acts:
        raise ValueError("No activations found for specified layers")
    
    if method == "concat":
        combined_pos = torch.cat(pos_acts, dim=-1)
        combined_neg = torch.cat(neg_acts, dim=-1)
    elif method == "mean":
        combined_pos = torch.stack(pos_acts, dim=0).mean(dim=0)
        combined_neg = torch.stack(neg_acts, dim=0).mean(dim=0)
    elif method == "weighted":
        weights = torch.linspace(0.5, 1.5, len(pos_acts))
        weights = weights / weights.sum()
        combined_pos = sum(w * a for w, a in zip(weights, pos_acts))
        combined_neg = sum(w * a for w, a in zip(weights, neg_acts))
    else:
        raise ValueError(f"Unknown combination method: {method}")
    
    return combined_pos, combined_neg


def _generate_multi_layer_recommendation(
    per_layer_results: Dict[int, LayerGeometryResult],
    combined_result: Optional[GeometryAnalysisResult],
    layer_subset_results: Dict[str, GeometryAnalysisResult],
    best_single_layer: int,
    best_single_layer_structure: StructureType,
    best_single_layer_score: float,
    layer_agreement: float,
) -> str:
    """Generate recommendation based on multi-layer analysis."""
    parts = []
    
    # Layer agreement insight
    if layer_agreement > 0.8:
        parts.append(f"High layer agreement ({layer_agreement:.0%}): structure is consistent across depth.")
    elif layer_agreement < 0.4:
        parts.append(f"Low layer agreement ({layer_agreement:.0%}): different structures at different depths.")
    
    # Best layer recommendation
    parts.append(f"Best single layer: {best_single_layer} with {best_single_layer_structure.value} ({best_single_layer_score:.2f}).")
    
    # Combined vs single
    if combined_result:
        if combined_result.best_score > best_single_layer_score + 0.1:
            parts.append(f"Combined layers improve detection ({combined_result.best_score:.2f} vs {best_single_layer_score:.2f}). Use multi-layer steering.")
        else:
            parts.append(f"Single layer is sufficient. Target layer {best_single_layer}.")
    
    # Layer subset insights
    if layer_subset_results:
        subset_scores = {name: r.best_score for name, r in layer_subset_results.items()}
        best_subset = max(subset_scores.keys(), key=lambda k: subset_scores[k])
        if subset_scores[best_subset] > best_single_layer_score:
            parts.append(f"'{best_subset}' layers show strongest structure ({subset_scores[best_subset]:.2f}).")
    
    return " ".join(parts)


def _generate_multi_layer_recommendation_v2(
    per_layer_results: Dict[int, LayerGeometryResult],
    combined_result: Optional[GeometryAnalysisResult],
    layer_subset_results: Dict[str, GeometryAnalysisResult],
    layer_pair_results: Dict[str, GeometryAnalysisResult],
    skip_results: Dict[str, GeometryAnalysisResult],
    best_single_layer: int,
    best_single_layer_structure: StructureType,
    best_single_layer_score: float,
    best_combination: Optional[str],
    best_combination_score: float,
    best_combination_structure: Optional[StructureType],
    layer_agreement: float,
    all_combinations_ranked: List[Tuple[str, float, StructureType]],
) -> str:
    """Generate comprehensive recommendation based on multi-layer analysis."""
    parts = []
    
    # Layer agreement insight
    if layer_agreement > 0.8:
        parts.append(f"High layer agreement ({layer_agreement:.0%}): consistent structure across depth.")
    elif layer_agreement < 0.4:
        parts.append(f"Low layer agreement ({layer_agreement:.0%}): structure varies by depth.")
    else:
        parts.append(f"Moderate layer agreement ({layer_agreement:.0%}).")
    
    # Overall best recommendation
    if best_combination and best_combination_score > best_single_layer_score + 0.05:
        improvement = best_combination_score - best_single_layer_score
        parts.append(
            f"BEST: '{best_combination}' ({best_combination_structure.value}: {best_combination_score:.2f}) "
            f"outperforms single layer {best_single_layer} by {improvement:.2f}."
        )
    else:
        parts.append(
            f"BEST: Layer {best_single_layer} ({best_single_layer_structure.value}: {best_single_layer_score:.2f}). "
            f"Multi-layer combinations don't improve detection."
        )
    
    # Top 3 combinations summary
    if len(all_combinations_ranked) >= 3:
        top3 = all_combinations_ranked[:3]
        top3_str = ", ".join([f"{name}={score:.2f}" for name, score, _ in top3])
        parts.append(f"Top 3: {top3_str}.")
    
    # Specific pattern insights
    if skip_results:
        skip_scores = {name: r.best_score for name, r in skip_results.items()}
        best_skip = max(skip_scores.keys(), key=lambda k: skip_scores[k])
        if skip_scores[best_skip] > best_single_layer_score:
            parts.append(f"Skip pattern '{best_skip}' is effective ({skip_scores[best_skip]:.2f}).")
    
    if layer_pair_results:
        pair_scores = {name: r.best_score for name, r in layer_pair_results.items()}
        best_pair = max(pair_scores.keys(), key=lambda k: pair_scores[k])
        best_pair_score = pair_scores[best_pair]
        if best_pair_score > best_single_layer_score:
            parts.append(f"Layer pair '{best_pair}' shows synergy ({best_pair_score:.2f}).")
    
    # Depth pattern analysis
    if per_layer_results and len(per_layer_results) >= 3:
        layers_sorted = sorted(per_layer_results.keys())
        early_score = per_layer_results[layers_sorted[0]].best_score
        late_score = per_layer_results[layers_sorted[-1]].best_score
        if late_score > early_score + 0.2:
            parts.append("Later layers show stronger structure than early layers.")
        elif early_score > late_score + 0.2:
            parts.append("Early layers show stronger structure than later layers.")
    
    return " ".join(parts)


def detect_geometry_all_layers(
    pairs_with_activations: List,
    layers: Optional[List[int]] = None,
    config: MultiLayerGeometryConfig | None = None,
) -> MultiLayerGeometryResult:
    """
    Convenience function to detect geometry from pairs with pre-collected activations.
    
    Arguments:
        pairs_with_activations: List of ContrastivePair objects with layers_activations populated
        layers: Specific layers to analyze (None = all available)
        config: Analysis configuration
        
    Returns:
        MultiLayerGeometryResult
    """
    if not pairs_with_activations:
        raise ValueError("No pairs provided")
    
    # Extract activations by layer
    pos_by_layer: Dict[int, List[torch.Tensor]] = {}
    neg_by_layer: Dict[int, List[torch.Tensor]] = {}
    
    for pair in pairs_with_activations:
        pos_acts = pair.positive_response.layers_activations
        neg_acts = pair.negative_response.layers_activations
        
        for layer_key, act in pos_acts.items():
            layer = int(layer_key)
            if layers is None or layer in layers:
                if layer not in pos_by_layer:
                    pos_by_layer[layer] = []
                pos_by_layer[layer].append(act.float() if act is not None else None)
        
        for layer_key, act in neg_acts.items():
            layer = int(layer_key)
            if layers is None or layer in layers:
                if layer not in neg_by_layer:
                    neg_by_layer[layer] = []
                neg_by_layer[layer].append(act.float() if act is not None else None)
    
    # Stack into tensors
    pos_tensors = {}
    neg_tensors = {}
    for layer in pos_by_layer:
        valid_pos = [a for a in pos_by_layer[layer] if a is not None]
        valid_neg = [a for a in neg_by_layer.get(layer, []) if a is not None]
        if valid_pos and valid_neg:
            pos_tensors[layer] = torch.stack(valid_pos)
            neg_tensors[layer] = torch.stack(valid_neg)
    
    return detect_geometry_multi_layer(pos_tensors, neg_tensors, config)


@dataclass
class ExhaustiveCombinationResult:
    """Result for a single layer combination."""
    layers: Tuple[int, ...]
    best_structure: StructureType
    best_score: float
    all_scores: Dict[str, float]


@dataclass
class ExhaustiveGeometryAnalysisResult:
    """Results from exhaustive layer combination analysis."""
    
    total_combinations: int
    """Total number of combinations tested."""
    
    all_results: List[ExhaustiveCombinationResult]
    """All results, sorted by best_score descending."""
    
    best_combination: Tuple[int, ...]
    """Layer combination with highest score."""
    
    best_score: float
    """Highest score achieved."""
    
    best_structure: StructureType
    """Structure type at best combination."""
    
    top_10: List[ExhaustiveCombinationResult]
    """Top 10 combinations."""
    
    single_layer_best: int
    """Best single layer."""
    
    single_layer_best_score: float
    """Score of best single layer."""
    
    combination_beats_single: bool
    """Whether any multi-layer combination beats best single layer."""
    
    improvement_over_single: float
    """How much best combination improves over best single layer."""
    
    patterns: Dict[str, Any]
    """Discovered patterns (layer frequency in top combinations, etc.)."""
    
    recommendation: str
    """Final recommendation."""


def detect_geometry_exhaustive(
    pos_activations_by_layer: Dict[int, torch.Tensor],
    neg_activations_by_layer: Dict[int, torch.Tensor],
    max_layers: int = 16,
    combination_method: str = "concat",
    num_components: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    top_k: int = 100,
) -> ExhaustiveGeometryAnalysisResult:
    """
    Exhaustively test all 2^N - 1 layer combinations for geometric structure.
    
    Memory-efficient: uses generators and only keeps top_k results in memory.
    
    Arguments:
        pos_activations_by_layer: Dict mapping layer index to positive activations [N, hidden_dim]
        neg_activations_by_layer: Dict mapping layer index to negative activations [N, hidden_dim]
        max_layers: Maximum number of layers to consider (limits combinations)
        combination_method: How to combine layers ("concat", "mean", "weighted")
        num_components: Number of PCA components for analysis
        progress_callback: Optional callback(current, total) for progress reporting
        top_k: Number of top results to keep in memory (default 100)
        
    Returns:
        ExhaustiveGeometryAnalysisResult with top combinations ranked
    """
    import heapq
    from itertools import combinations as itertools_combinations
    
    layers = sorted(pos_activations_by_layer.keys())[:max_layers]
    n_layers = len(layers)
    
    if n_layers == 0:
        raise ValueError("No layers provided")
    
    geo_cfg = GeometryAnalysisConfig(num_components=num_components, optimization_steps=50)
    
    # Calculate total without building list (2^n - 1)
    total_combinations = (1 << n_layers) - 1
    
    # Use min-heap to keep top_k results (negate scores for max-heap behavior)
    top_results_heap: List[Tuple[float, ExhaustiveCombinationResult]] = []
    single_layer_results: List[ExhaustiveCombinationResult] = []
    
    # Generator for combinations - no upfront memory allocation
    def combo_generator():
        for r in range(1, n_layers + 1):
            for combo in itertools_combinations(layers, r):
                yield combo
    
    # Test each combination
    idx = 0
    for combo in combo_generator():
        idx += 1
        if progress_callback:
            progress_callback(idx, total_combinations)
        
        # Combine activations for this subset
        if len(combo) == 1:
            layer = combo[0]
            combined_pos = pos_activations_by_layer[layer]
            combined_neg = neg_activations_by_layer[layer]
        else:
            combined_pos, combined_neg = _combine_layer_activations(
                pos_activations_by_layer, neg_activations_by_layer, 
                list(combo), combination_method
            )
        
        # Run geometry detection
        result = detect_geometry_structure(combined_pos, combined_neg, geo_cfg)
        
        all_scores = {name: score.score for name, score in result.all_scores.items()}
        combo_result = ExhaustiveCombinationResult(
            layers=combo,
            best_structure=result.best_structure,
            best_score=result.best_score,
            all_scores=all_scores,
        )
        
        # Track single layer results separately
        if len(combo) == 1:
            single_layer_results.append(combo_result)
        
        # Maintain top_k using heap
        if len(top_results_heap) < top_k:
            heapq.heappush(top_results_heap, (combo_result.best_score, combo_result))
        elif combo_result.best_score > top_results_heap[0][0]:
            heapq.heapreplace(top_results_heap, (combo_result.best_score, combo_result))
    
    # Extract top results sorted by score descending
    all_results = [r for _, r in sorted(top_results_heap, key=lambda x: -x[0])]
    
    # Extract insights
    best_result = all_results[0] if all_results else None
    best_combination = best_result.layers if best_result else ()
    best_score = best_result.best_score if best_result else 0.0
    best_structure = best_result.best_structure if best_result else StructureType.UNKNOWN
    
    top_10 = all_results[:10]
    
    # Find best single layer
    if single_layer_results:
        single_layer_results.sort(key=lambda x: x.best_score, reverse=True)
        single_layer_best = single_layer_results[0].layers[0]
        single_layer_best_score = single_layer_results[0].best_score
    else:
        single_layer_best = layers[0]
        single_layer_best_score = 0.0
    
    combination_beats_single = best_score > single_layer_best_score
    improvement_over_single = best_score - single_layer_best_score
    
    # Analyze patterns from top results
    patterns = _analyze_combination_patterns(all_results, layers, top_k=min(50, len(all_results)))
    
    # Generate recommendation
    recommendation = _generate_exhaustive_recommendation(
        best_combination, best_score, best_structure,
        single_layer_best, single_layer_best_score,
        combination_beats_single, improvement_over_single,
        patterns, total_combinations
    )
    
    return ExhaustiveGeometryAnalysisResult(
        total_combinations=total_combinations,
        all_results=all_results,
        best_combination=best_combination,
        best_score=best_score,
        best_structure=best_structure,
        top_10=top_10,
        single_layer_best=single_layer_best,
        single_layer_best_score=single_layer_best_score,
        combination_beats_single=combination_beats_single,
        improvement_over_single=improvement_over_single,
        patterns=patterns,
        recommendation=recommendation,
    )


def detect_geometry_limited(
    pos_activations_by_layer: Dict[int, torch.Tensor],
    neg_activations_by_layer: Dict[int, torch.Tensor],
    max_combo_size: int = 3,
    combination_method: str = "concat",
    num_components: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    top_k: int = 100,
) -> ExhaustiveGeometryAnalysisResult:
    """
    Test limited layer combinations: 1-layer, 2-layer, ..., max_combo_size-layer, plus all layers.
    
    Much faster than exhaustive search while still finding good combinations.
    For N layers with max_combo_size=3:
    - 1-layer: N combinations
    - 2-layer: N*(N-1)/2 combinations  
    - 3-layer: N*(N-1)*(N-2)/6 combinations
    - all-layers: 1 combination
    
    Total: O(N^3) instead of O(2^N)
    
    Arguments:
        pos_activations_by_layer: Dict mapping layer index to positive activations [N, hidden_dim]
        neg_activations_by_layer: Dict mapping layer index to negative activations [N, hidden_dim]
        max_combo_size: Maximum combination size to test (1, 2, 3, etc.) before jumping to all
        combination_method: How to combine layers ("concat", "mean", "weighted")
        num_components: Number of PCA components for analysis
        progress_callback: Optional callback(current, total) for progress reporting
        top_k: Number of top results to keep in memory (default 100)
        
    Returns:
        ExhaustiveGeometryAnalysisResult with top combinations ranked
    """
    import heapq
    from itertools import combinations as itertools_combinations
    from math import comb
    
    layers = sorted(pos_activations_by_layer.keys())
    n_layers = len(layers)
    
    if n_layers == 0:
        raise ValueError("No layers provided")
    
    geo_cfg = GeometryAnalysisConfig(num_components=num_components, optimization_steps=50)
    
    # Calculate total combinations: sum of C(n,r) for r=1 to max_combo_size, plus 1 for all layers
    total_combinations = sum(comb(n_layers, r) for r in range(1, min(max_combo_size, n_layers) + 1))
    if max_combo_size < n_layers:
        total_combinations += 1  # Add all-layers combination
    
    # Use min-heap to keep top_k results
    top_results_heap: List[Tuple[float, ExhaustiveCombinationResult]] = []
    single_layer_results: List[ExhaustiveCombinationResult] = []
    
    # Generator for limited combinations
    def combo_generator():
        # 1-layer, 2-layer, ..., max_combo_size-layer
        for r in range(1, min(max_combo_size, n_layers) + 1):
            for combo in itertools_combinations(layers, r):
                yield combo
        # All layers (if not already included)
        if max_combo_size < n_layers:
            yield tuple(layers)
    
    # Test each combination
    idx = 0
    for combo in combo_generator():
        idx += 1
        if progress_callback:
            progress_callback(idx, total_combinations)
        
        # Combine activations for this subset
        if len(combo) == 1:
            layer = combo[0]
            combined_pos = pos_activations_by_layer[layer]
            combined_neg = neg_activations_by_layer[layer]
        else:
            combined_pos, combined_neg = _combine_layer_activations(
                pos_activations_by_layer, neg_activations_by_layer, 
                list(combo), combination_method
            )
        
        # Run geometry detection
        result = detect_geometry_structure(combined_pos, combined_neg, geo_cfg)
        
        all_scores = {name: score.score for name, score in result.all_scores.items()}
        combo_result = ExhaustiveCombinationResult(
            layers=combo,
            best_structure=result.best_structure,
            best_score=result.best_score,
            all_scores=all_scores,
        )
        
        # Track single layer results separately
        if len(combo) == 1:
            single_layer_results.append(combo_result)
        
        # Maintain top_k using heap
        if len(top_results_heap) < top_k:
            heapq.heappush(top_results_heap, (combo_result.best_score, combo_result))
        elif combo_result.best_score > top_results_heap[0][0]:
            heapq.heapreplace(top_results_heap, (combo_result.best_score, combo_result))
    
    # Extract top results sorted by score descending
    all_results = [r for _, r in sorted(top_results_heap, key=lambda x: -x[0])]
    
    # Extract insights
    best_result = all_results[0] if all_results else None
    best_combination = best_result.layers if best_result else ()
    best_score = best_result.best_score if best_result else 0.0
    best_structure = best_result.best_structure if best_result else StructureType.UNKNOWN
    
    top_10 = all_results[:10]
    
    # Find best single layer
    if single_layer_results:
        single_layer_results.sort(key=lambda x: x.best_score, reverse=True)
        single_layer_best = single_layer_results[0].layers[0]
        single_layer_best_score = single_layer_results[0].best_score
    else:
        single_layer_best = layers[0]
        single_layer_best_score = 0.0
    
    combination_beats_single = best_score > single_layer_best_score
    improvement_over_single = best_score - single_layer_best_score
    
    # Analyze patterns from top results
    patterns = _analyze_combination_patterns(all_results, layers, top_k=min(50, len(all_results)))
    
    # Generate recommendation
    recommendation = _generate_exhaustive_recommendation(
        best_combination, best_score, best_structure,
        single_layer_best, single_layer_best_score,
        combination_beats_single, improvement_over_single,
        patterns, total_combinations
    )
    
    return ExhaustiveGeometryAnalysisResult(
        total_combinations=total_combinations,
        all_results=all_results,
        best_combination=best_combination,
        best_score=best_score,
        best_structure=best_structure,
        top_10=top_10,
        single_layer_best=single_layer_best,
        single_layer_best_score=single_layer_best_score,
        combination_beats_single=combination_beats_single,
        improvement_over_single=improvement_over_single,
        patterns=patterns,
        recommendation=recommendation,
    )


def detect_geometry_contiguous(
    pos_activations_by_layer: Dict[int, torch.Tensor],
    neg_activations_by_layer: Dict[int, torch.Tensor],
    combination_method: str = "concat",
    num_components: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    top_k: int = 100,
) -> ExhaustiveGeometryAnalysisResult:
    """
    Test contiguous layer combinations only.
    
    Only tests combinations where layers are adjacent: 1-2, 2-3, 1-3, 5-8, etc.
    Much faster: O(N^2) combinations instead of O(2^N).
    
    For N layers: N*(N+1)/2 combinations
    - 36 layers: 666 combinations
    - 24 layers: 300 combinations
    
    Arguments:
        pos_activations_by_layer: Dict mapping layer index to positive activations [N, hidden_dim]
        neg_activations_by_layer: Dict mapping layer index to negative activations [N, hidden_dim]
        combination_method: How to combine layers ("concat", "mean", "weighted")
        num_components: Number of PCA components for analysis
        progress_callback: Optional callback(current, total) for progress reporting
        top_k: Number of top results to keep in memory (default 100)
        
    Returns:
        ExhaustiveGeometryAnalysisResult with top combinations ranked
    """
    import heapq
    
    layers = sorted(pos_activations_by_layer.keys())
    n_layers = len(layers)
    
    if n_layers == 0:
        raise ValueError("No layers provided")
    
    geo_cfg = GeometryAnalysisConfig(num_components=num_components, optimization_steps=50)
    
    # Total contiguous combinations: N*(N+1)/2
    total_combinations = n_layers * (n_layers + 1) // 2
    
    # Use min-heap to keep top_k results
    top_results_heap: List[Tuple[float, ExhaustiveCombinationResult]] = []
    single_layer_results: List[ExhaustiveCombinationResult] = []
    
    # Generator for contiguous combinations
    def combo_generator():
        # For each starting layer
        for start_idx in range(n_layers):
            # For each ending layer (inclusive)
            for end_idx in range(start_idx, n_layers):
                yield tuple(layers[start_idx:end_idx + 1])
    
    # Test each combination
    idx = 0
    for combo in combo_generator():
        idx += 1
        if progress_callback:
            progress_callback(idx, total_combinations)
        
        # Combine activations for this subset
        if len(combo) == 1:
            layer = combo[0]
            combined_pos = pos_activations_by_layer[layer]
            combined_neg = neg_activations_by_layer[layer]
        else:
            combined_pos, combined_neg = _combine_layer_activations(
                pos_activations_by_layer, neg_activations_by_layer, 
                list(combo), combination_method
            )
        
        # Run geometry detection
        result = detect_geometry_structure(combined_pos, combined_neg, geo_cfg)
        
        all_scores = {name: score.score for name, score in result.all_scores.items()}
        combo_result = ExhaustiveCombinationResult(
            layers=combo,
            best_structure=result.best_structure,
            best_score=result.best_score,
            all_scores=all_scores,
        )
        
        # Track single layer results separately
        if len(combo) == 1:
            single_layer_results.append(combo_result)
        
        # Maintain top_k using heap
        if len(top_results_heap) < top_k:
            heapq.heappush(top_results_heap, (combo_result.best_score, combo_result))
        elif combo_result.best_score > top_results_heap[0][0]:
            heapq.heapreplace(top_results_heap, (combo_result.best_score, combo_result))
    
    # Extract top results sorted by score descending
    all_results = [r for _, r in sorted(top_results_heap, key=lambda x: -x[0])]
    
    # Extract insights
    best_result = all_results[0] if all_results else None
    best_combination = best_result.layers if best_result else ()
    best_score = best_result.best_score if best_result else 0.0
    best_structure = best_result.best_structure if best_result else StructureType.UNKNOWN
    
    top_10 = all_results[:10]
    
    # Find best single layer
    if single_layer_results:
        single_layer_results.sort(key=lambda x: x.best_score, reverse=True)
        single_layer_best = single_layer_results[0].layers[0]
        single_layer_best_score = single_layer_results[0].best_score
    else:
        single_layer_best = layers[0]
        single_layer_best_score = 0.0
    
    combination_beats_single = best_score > single_layer_best_score
    improvement_over_single = best_score - single_layer_best_score
    
    # Analyze patterns from top results
    patterns = _analyze_combination_patterns(all_results, layers, top_k=min(50, len(all_results)))
    
    # Generate recommendation
    recommendation = _generate_exhaustive_recommendation(
        best_combination, best_score, best_structure,
        single_layer_best, single_layer_best_score,
        combination_beats_single, improvement_over_single,
        patterns, total_combinations
    )
    
    return ExhaustiveGeometryAnalysisResult(
        total_combinations=total_combinations,
        all_results=all_results,
        best_combination=best_combination,
        best_score=best_score,
        best_structure=best_structure,
        top_10=top_10,
        single_layer_best=single_layer_best,
        single_layer_best_score=single_layer_best_score,
        combination_beats_single=combination_beats_single,
        improvement_over_single=improvement_over_single,
        patterns=patterns,
        recommendation=recommendation,
    )


def detect_geometry_smart(
    pos_activations_by_layer: Dict[int, torch.Tensor],
    neg_activations_by_layer: Dict[int, torch.Tensor],
    max_combo_size: int = 3,
    combination_method: str = "concat",
    num_components: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    top_k: int = 100,
) -> ExhaustiveGeometryAnalysisResult:
    """
    Smart layer combination search: contiguous + limited (1,2,3-layer) combinations.
    
    Tests:
    1. All contiguous combinations (L1-L2, L1-L3, L5-L10, etc.)
    2. All 1,2,3-layer non-contiguous combinations
    
    Deduplicates overlapping combinations.
    
    For N=36 layers with max_combo_size=3:
    - Contiguous: 666 combinations
    - Limited non-contiguous: ~7,100 additional combinations
    - Total: ~7,800 unique combinations
    
    Arguments:
        pos_activations_by_layer: Dict mapping layer index to positive activations [N, hidden_dim]
        neg_activations_by_layer: Dict mapping layer index to negative activations [N, hidden_dim]
        max_combo_size: Maximum combination size for non-contiguous (default: 3)
        combination_method: How to combine layers ("concat", "mean", "weighted")
        num_components: Number of PCA components for analysis
        progress_callback: Optional callback(current, total) for progress reporting
        top_k: Number of top results to keep in memory (default 100)
        
    Returns:
        ExhaustiveGeometryAnalysisResult with top combinations ranked
    """
    import heapq
    from itertools import combinations as itertools_combinations
    
    layers = sorted(pos_activations_by_layer.keys())
    n_layers = len(layers)
    
    if n_layers == 0:
        raise ValueError("No layers provided")
    
    geo_cfg = GeometryAnalysisConfig(num_components=num_components, optimization_steps=50)
    
    # Generate all unique combinations: contiguous + limited
    all_combos_set: set = set()
    
    # Add contiguous combinations
    for start_idx in range(n_layers):
        for end_idx in range(start_idx, n_layers):
            all_combos_set.add(tuple(layers[start_idx:end_idx + 1]))
    
    # Add limited combinations (1,2,3-layer)
    for r in range(1, min(max_combo_size, n_layers) + 1):
        for combo in itertools_combinations(layers, r):
            all_combos_set.add(combo)
    
    # Convert to sorted list
    all_combos = sorted(all_combos_set, key=lambda x: (len(x), x))
    total_combinations = len(all_combos)
    
    # Use min-heap to keep top_k results
    top_results_heap: List[Tuple[float, ExhaustiveCombinationResult]] = []
    single_layer_results: List[ExhaustiveCombinationResult] = []
    
    # Test each combination
    for idx, combo in enumerate(all_combos):
        if progress_callback:
            progress_callback(idx + 1, total_combinations)
        
        # Combine activations for this subset
        if len(combo) == 1:
            layer = combo[0]
            combined_pos = pos_activations_by_layer[layer]
            combined_neg = neg_activations_by_layer[layer]
        else:
            combined_pos, combined_neg = _combine_layer_activations(
                pos_activations_by_layer, neg_activations_by_layer, 
                list(combo), combination_method
            )
        
        # Run geometry detection
        result = detect_geometry_structure(combined_pos, combined_neg, geo_cfg)
        
        all_scores = {name: score.score for name, score in result.all_scores.items()}
        combo_result = ExhaustiveCombinationResult(
            layers=combo,
            best_structure=result.best_structure,
            best_score=result.best_score,
            all_scores=all_scores,
        )
        
        # Track single layer results separately
        if len(combo) == 1:
            single_layer_results.append(combo_result)
        
        # Maintain top_k using heap
        if len(top_results_heap) < top_k:
            heapq.heappush(top_results_heap, (combo_result.best_score, combo_result))
        elif combo_result.best_score > top_results_heap[0][0]:
            heapq.heapreplace(top_results_heap, (combo_result.best_score, combo_result))
    
    # Extract top results sorted by score descending
    all_results = [r for _, r in sorted(top_results_heap, key=lambda x: -x[0])]
    
    # Extract insights
    best_result = all_results[0] if all_results else None
    best_combination = best_result.layers if best_result else ()
    best_score = best_result.best_score if best_result else 0.0
    best_structure = best_result.best_structure if best_result else StructureType.UNKNOWN
    
    top_10 = all_results[:10]
    
    # Find best single layer
    if single_layer_results:
        single_layer_results.sort(key=lambda x: x.best_score, reverse=True)
        single_layer_best = single_layer_results[0].layers[0]
        single_layer_best_score = single_layer_results[0].best_score
    else:
        single_layer_best = layers[0]
        single_layer_best_score = 0.0
    
    combination_beats_single = best_score > single_layer_best_score
    improvement_over_single = best_score - single_layer_best_score
    
    # Analyze patterns from top results
    patterns = _analyze_combination_patterns(all_results, layers, top_k=min(50, len(all_results)))
    
    # Generate recommendation
    recommendation = _generate_exhaustive_recommendation(
        best_combination, best_score, best_structure,
        single_layer_best, single_layer_best_score,
        combination_beats_single, improvement_over_single,
        patterns, total_combinations
    )
    
    return ExhaustiveGeometryAnalysisResult(
        total_combinations=total_combinations,
        all_results=all_results,
        best_combination=best_combination,
        best_score=best_score,
        best_structure=best_structure,
        top_10=top_10,
        single_layer_best=single_layer_best,
        single_layer_best_score=single_layer_best_score,
        combination_beats_single=combination_beats_single,
        improvement_over_single=improvement_over_single,
        patterns=patterns,
        recommendation=recommendation,
    )


def _analyze_combination_patterns(
    all_results: List[ExhaustiveCombinationResult],
    layers: List[int],
    top_k: int = 50,
) -> Dict[str, Any]:
    """Analyze patterns in top combinations."""
    from collections import Counter
    
    top_results = all_results[:top_k]
    
    # Layer frequency in top combinations
    layer_freq = Counter()
    for r in top_results:
        for layer in r.layers:
            layer_freq[layer] += 1
    
    # Combination size distribution in top results
    size_dist = Counter(len(r.layers) for r in top_results)
    
    # Best score by combination size
    size_to_best: Dict[int, float] = {}
    for r in all_results:
        size = len(r.layers)
        if size not in size_to_best or r.best_score > size_to_best[size]:
            size_to_best[size] = r.best_score
    
    # Structure frequency in top combinations
    structure_freq = Counter(r.best_structure for r in top_results)
    
    # Adjacent layer pairs in top combinations
    adjacent_count = 0
    for r in top_results:
        if len(r.layers) >= 2:
            sorted_layers = sorted(r.layers)
            for i in range(len(sorted_layers) - 1):
                if sorted_layers[i + 1] - sorted_layers[i] == 1:
                    adjacent_count += 1
                    break
    
    # Layer position analysis (early vs late layers)
    mid_layer = layers[len(layers) // 2] if layers else 0
    early_in_top = sum(1 for r in top_results for l in r.layers if l < mid_layer)
    late_in_top = sum(1 for r in top_results for l in r.layers if l >= mid_layer)
    
    return {
        "layer_frequency_in_top": dict(layer_freq.most_common()),
        "most_important_layers": [l for l, _ in layer_freq.most_common(5)],
        "size_distribution_in_top": dict(size_dist),
        "best_score_by_size": size_to_best,
        "optimal_combination_size": max(size_to_best.keys(), key=lambda k: size_to_best[k]) if size_to_best else 1,
        "structure_frequency_in_top": {s.value: c for s, c in structure_freq.most_common()},
        "dominant_structure": structure_freq.most_common(1)[0][0].value if structure_freq else "unknown",
        "adjacent_pairs_in_top": adjacent_count,
        "early_vs_late_ratio": early_in_top / late_in_top if late_in_top > 0 else float('inf'),
    }


def _generate_exhaustive_recommendation(
    best_combination: Tuple[int, ...],
    best_score: float,
    best_structure: StructureType,
    single_layer_best: int,
    single_layer_best_score: float,
    combination_beats_single: bool,
    improvement_over_single: float,
    patterns: Dict[str, Any],
    total_combinations: int,
) -> str:
    """Generate recommendation from exhaustive analysis."""
    parts = []
    
    parts.append(f"Tested {total_combinations} layer combinations.")
    
    if combination_beats_single and improvement_over_single > 0.05:
        layers_str = "+".join(f"L{l}" for l in best_combination)
        parts.append(
            f"BEST: {layers_str} ({best_structure.value}: {best_score:.3f}), "
            f"+{improvement_over_single:.3f} over single layer L{single_layer_best}."
        )
    else:
        parts.append(
            f"BEST: Single layer L{single_layer_best} ({best_score:.3f}). "
            f"Multi-layer combinations don't significantly improve."
        )
    
    # Pattern insights
    opt_size = patterns.get("optimal_combination_size", 1)
    if opt_size > 1:
        parts.append(f"Optimal combination size: {opt_size} layers.")
    
    important_layers = patterns.get("most_important_layers", [])
    if important_layers:
        layers_str = ", ".join(f"L{l}" for l in important_layers[:3])
        parts.append(f"Most important layers: {layers_str}.")
    
    dominant = patterns.get("dominant_structure", "unknown")
    parts.append(f"Dominant structure: {dominant}.")
    
    return " ".join(parts)