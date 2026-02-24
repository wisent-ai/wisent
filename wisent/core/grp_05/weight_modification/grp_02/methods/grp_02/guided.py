"""
Guided Weight Modification using Linearity Diagnostics.

This module implements data-driven weight modification that uses linearity
diagnostics to automatically:
1. Select optimal layers based on linear probe accuracy
2. Weight ablation strength by Fisher ratio per layer
3. Perform surgical single-layer ablation when appropriate
4. Validate collateral damage on unrelated benchmarks

Key innovations over blind parameter search:
- Layer selection based on measured linear separability, not optimization
- Fisher ratio-weighted ablation (high separability = stronger ablation)
- Surgical modification of only the layers with strong signal
- Zwiad delta validation to ensure minimal collateral damage
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from enum import Enum

from wisent.core.cli.cli_logger import setup_logger, bind
from wisent.core.constants import (
    DEFAULT_STRENGTH,
    GUIDED_MIN_LINEAR_SCORE, GUIDED_SURGICAL_TOP_K,
    GUIDED_FISHER_WEIGHT_MIN, GUIDED_FISHER_WEIGHT_MAX,
    GUIDED_MAX_DEGRADATION,
)

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair

__all__ = [
    "GuidedModificationConfig",
    "GuidedModificationResult",
    "LayerDiagnostics",
    "run_guided_modification",
    "compute_layer_diagnostics",
    "compute_fisher_weights",
    "select_surgical_layers",
    "validate_collateral_damage",
    "CollateralDamageReport",
]

_LOG = setup_logger(__name__)


class AblationMode(Enum):
    """Mode for guided ablation."""
    FULL = "full"  # Ablate all layers with signal
    SURGICAL = "surgical"  # Ablate only top-k layers
    ADAPTIVE = "adaptive"  # Adapt based on diagnostics


@dataclass
class GuidedModificationConfig:
    """Configuration for guided weight modification."""
    
    # Layer selection
    min_linear_score: float = GUIDED_MIN_LINEAR_SCORE
    """Minimum linear score to include a layer in ablation."""

    surgical_top_k: int = GUIDED_SURGICAL_TOP_K
    """Number of top layers for surgical mode."""
    
    # Fisher ratio weighting
    use_fisher_weights: bool = True
    """Weight ablation strength by Fisher ratio."""
    
    fisher_weight_scale: float = DEFAULT_STRENGTH
    """Scale factor for Fisher-based weights."""
    
    fisher_weight_min: float = GUIDED_FISHER_WEIGHT_MIN
    """Minimum weight (prevents zero ablation)."""

    fisher_weight_max: float = GUIDED_FISHER_WEIGHT_MAX
    """Maximum weight (prevents over-ablation)."""
    
    # Ablation mode
    mode: AblationMode = AblationMode.ADAPTIVE
    """Ablation mode: full, surgical, or adaptive."""
    
    # Validation
    validate_collateral: bool = True
    """Run collateral damage validation."""
    
    max_allowed_degradation: float = GUIDED_MAX_DEGRADATION
    """Maximum allowed degradation on unrelated benchmarks."""
    
    validation_benchmarks: Optional[List[str]] = None
    """Benchmarks to use for validation. If None, auto-select."""
    
    # Extraction strategy
    extraction_strategy: str = "chat_last"
    """Extraction strategy for computing directions."""
    
    # General
    base_strength: float = DEFAULT_STRENGTH
    """Base ablation strength before layer weighting."""
    
    normalize_vectors: bool = True
    """Normalize steering vectors."""
    
    verbose: bool = True
    """Print progress information."""


@dataclass
class LayerDiagnostics:
    """Diagnostics for a single layer."""
    
    layer_idx: int
    """Layer index (0-indexed)."""
    
    linear_score: float
    """Linear probe accuracy (0-1)."""
    
    knn_score: float
    """k-NN accuracy (0-1)."""
    
    fisher_ratio: float
    """Fisher discriminant ratio."""
    
    cohens_d: float
    """Cohen's d effect size."""
    
    variance_explained: float
    """Variance explained by primary direction."""
    
    extraction_strategy: str
    """Extraction strategy used."""
    
    recommended_weight: float
    """Recommended ablation weight based on diagnostics."""
    
    details: Dict[str, Any] = field(default_factory=dict)
    """Additional diagnostic details."""


@dataclass
class GuidedModificationResult:
    """Result of guided weight modification."""
    
    layers_modified: int
    """Number of layers modified."""
    
    total_parameters_modified: int
    """Total parameters modified."""
    
    layer_diagnostics: Dict[int, LayerDiagnostics]
    """Per-layer diagnostics."""
    
    layer_weights: Dict[int, float]
    """Applied weights per layer."""
    
    mode_used: AblationMode
    """Ablation mode that was used."""
    
    steering_vectors: Dict[int, Tensor]
    """Steering vectors per layer."""
    
    collateral_report: Optional["CollateralDamageReport"] = None
    """Collateral damage validation report."""
    
    recommendation: str = ""
    """Summary recommendation."""


@dataclass
class CollateralDamageReport:
    """Report on collateral damage to unrelated representations."""
    
    benchmarks_tested: List[str]
    """Benchmarks used for validation."""
    
    before_scores: Dict[str, float]
    """Linear probe scores before modification."""
    
    after_scores: Dict[str, float]
    """Linear probe scores after modification."""
    
    deltas: Dict[str, float]
    """Score deltas (before - after, positive = degradation)."""
    
    max_degradation: float
    """Maximum degradation across benchmarks."""
    
    mean_degradation: float
    """Mean degradation across benchmarks."""
    
    passed: bool
    """Whether validation passed (degradation below threshold)."""
    
    details: Dict[str, Any] = field(default_factory=dict)



# =============================================================================
# Functions imported from extracted modules for backward compatibility
# =============================================================================
from wisent.core.weight_modification.methods._guided_diagnostics import (
    compute_layer_diagnostics,
    _compute_single_layer_diagnostics,
    _compute_linear_probe_accuracy,
)
from wisent.core.weight_modification.methods._guided_scoring import (
    _compute_knn_accuracy,
    _compute_fisher_ratio,
    _compute_recommended_weight,
    compute_fisher_weights,
)
from wisent.core.weight_modification.methods._guided_runners import (
    select_surgical_layers,
    run_guided_modification,
    _compute_steering_vectors,
)
from wisent.core.weight_modification.methods._guided_validation import (
    validate_collateral_damage,
)
