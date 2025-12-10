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