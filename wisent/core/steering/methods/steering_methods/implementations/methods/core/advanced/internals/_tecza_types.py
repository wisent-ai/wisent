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

from wisent.core.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.errors import InsufficientDataError
from wisent.core.constants import (
    DEFAULT_VARIANCE_THRESHOLD,
    TECZA_NUM_DIRECTIONS,
    TECZA_MARGINAL_THRESHOLD,
    TECZA_MAX_DIRECTIONS,
    DEFAULT_OPTIMIZATION_STEPS,
    TECZA_LEARNING_RATE,
    TECZA_RETAIN_WEIGHT,
    TECZA_INDEPENDENCE_WEIGHT,
    TECZA_ABLATION_WEIGHT,
    TECZA_ADDITION_WEIGHT,
    TECZA_MIN_COSINE_SIM,
    TECZA_MAX_COSINE_SIM,
)

__all__ = [
    "TECZAMethod",
    "TECZAConfig",
    "MultiDirectionResult",
]


@dataclass
class TECZAConfig:
    """Configuration for TECZA steering method."""
    
    num_directions: int = TECZA_NUM_DIRECTIONS
    """Number of directions to discover per layer. Set to 'auto' or -1 for automatic."""
    
    auto_num_directions: bool = False
    """Automatically determine num_directions based on explained variance."""
    
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD
    """Target cumulative variance for auto num_directions."""
    
    marginal_threshold: float = TECZA_MARGINAL_THRESHOLD
    """Minimum marginal variance for adding another direction."""
    
    max_directions: int = TECZA_MAX_DIRECTIONS
    """Maximum directions when using auto num_directions."""
    
    optimization_steps: int = DEFAULT_OPTIMIZATION_STEPS
    """Number of gradient descent steps for direction optimization."""
    
    learning_rate: float = TECZA_LEARNING_RATE
    """Learning rate for direction optimization."""
    
    retain_weight: float = TECZA_RETAIN_WEIGHT
    """Weight for retain loss (preserving behavior on negative/harmless examples)."""
    
    independence_weight: float = TECZA_INDEPENDENCE_WEIGHT
    """Weight for representational independence loss between directions."""
    
    ablation_weight: float = TECZA_ABLATION_WEIGHT
    """Weight for ablation loss (making model comply with harmful after ablation)."""
    
    addition_weight: float = TECZA_ADDITION_WEIGHT
    """Weight for addition loss (making model refuse harmless after addition)."""
    
    normalize: bool = True
    """Whether to L2-normalize the final directions."""
    
    use_caa_init: bool = True
    """Whether to initialize first direction using CAA (difference-in-means)."""
    
    use_universal_basis_init: bool = False
    """Whether to initialize from universal subspace basis if available."""
    
    cone_constraint: bool = True
    """Whether to constrain directions to form a polyhedral cone (all positive combinations)."""
    
    min_cosine_similarity: float = TECZA_MIN_COSINE_SIM
    """Minimum cosine similarity between directions (they should be related, not orthogonal)."""
    
    max_cosine_similarity: float = TECZA_MAX_COSINE_SIM
    """Maximum cosine similarity (avoid redundant directions)."""


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


