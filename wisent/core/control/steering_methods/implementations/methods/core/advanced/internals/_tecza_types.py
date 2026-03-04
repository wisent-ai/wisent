"""
TECZA - Projected Representations for Independent Steering Manifolds.

A gradient-optimized multi-directional steering method that discovers multiple
refusal directions per layer, forming a coherent steering manifold.

Based on insights from:
- "The Geometry of Refusal in Large Language Models" (Wollschläger et al., 2025)
- "SOM Directions are Better than One" (Piras et al., 2025)

Key innovations:
1. Gradient-based direction optimization (not just difference-in-means)
2. Multiple directions per layer that form a coherent manifold
3. Representational independence constraint (soft, not strict orthogonality)
4. Retain loss to minimize side effects on harmless queries
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from wisent.core.control.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.utils.infra_tools.errors import InsufficientDataError
from wisent.core.utils.infra_tools.errors import InsufficientDataError

__all__ = [
    "TECZAMethod",
    "TECZAConfig",
    "MultiDirectionResult",
]


@dataclass
class TECZAConfig:
    """Configuration for TECZA steering method."""

    # Required fields (no defaults — must come from optimizer or explicit config)
    num_directions: int
    """Number of directions to discover per layer."""

    optimization_steps: int
    """Number of gradient descent steps for direction optimization."""

    learning_rate: float
    """Learning rate for direction optimization."""

    retain_weight: float
    """Weight for retain loss (preserving behavior on negative/harmless examples)."""

    independence_weight: float
    """Weight for representational independence loss between directions."""

    min_cosine_similarity: float
    """Minimum cosine similarity between directions (they should be related, not orthogonal)."""

    max_cosine_similarity: float
    """Maximum cosine similarity (avoid redundant directions)."""

    variance_threshold: float
    """Target cumulative variance for auto num_directions."""

    marginal_threshold: float
    """Minimum marginal variance for adding another direction."""

    max_directions: int
    """Maximum directions when using auto num_directions."""

    ablation_weight: float
    """Weight for ablation loss (making model comply with harmful after ablation)."""

    addition_weight: float
    """Weight for addition loss (making model refuse harmless after addition)."""

    separation_margin: float
    """Margin for separation loss (pos_mean - neg_mean must exceed this)."""

    perturbation_scale: float
    """Scale of noise added when perturbing CAA direction for additional directions."""

    universal_basis_noise: float
    """Noise scale when initializing from universal subspace basis."""

    # Optional fields with defaults
    auto_num_directions: bool = False
    """Automatically determine num_directions based on explained variance."""

    normalize: bool = True
    """Whether to L2-normalize the final directions."""

    use_caa_init: bool = True
    """Whether to initialize first direction using CAA (difference-in-means)."""

    use_universal_basis_init: bool = False
    """Whether to initialize from universal subspace basis if available."""

    cone_constraint: bool = True
    """Whether to constrain directions to form a polyhedral cone (all positive combinations)."""


@dataclass
class MultiDirectionResult:
    """Result containing multiple steering directions per layer."""
    
    directions: Dict[LayerName, torch.Tensor]
    """Per-layer directions tensor of shape [num_directions, hidden_dim]."""
    
    metadata: Dict[str, Any]
    """Training metadata including losses and diagnostics."""
    
    def get_primary_direction(self, layer: LayerName) -> torch.Tensor:
        """Get the primary (first/strongest) direction for a layer."""
        return self.directions[layer][0]
    
    def get_all_directions(self, layer: LayerName) -> torch.Tensor:
        """Get all directions for a layer as [num_directions, hidden_dim]."""
        return self.directions[layer]
    
    def to_single_direction_map(self) -> Dict[LayerName, torch.Tensor]:
        """Convert to single-direction format (for backward compatibility)."""
        return {layer: dirs[0] for layer, dirs in self.directions.items()}


