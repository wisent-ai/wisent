"""Multi-concept type definitions: enums, specs, configs, and results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "ConceptAction",
    "ConceptSpec",
    "MultiConceptConfig",
    "MultiConceptResult",
]


class ConceptAction(Enum):
    """Action to take on a concept direction."""
    SUPPRESS = "suppress"
    ENHANCE = "enhance"
    NEUTRAL = "neutral"


@dataclass
class ConceptSpec:
    """Specification for a single concept to modify."""

    name: str
    """Human-readable name for the concept."""

    steering_vectors: Dict[int, "Tensor"]
    """Per-layer steering vectors for this concept."""

    action: ConceptAction = ConceptAction.SUPPRESS
    """Action to take: suppress, enhance, or neutral."""

    strength: float = 1.0
    """Modification strength for this concept."""

    layer_weights: Optional[Dict[int, float]] = None
    """Optional per-layer weights. If None, uses uniform weights."""

    priority: int = 1
    """Priority for conflict resolution (higher = more important)."""


@dataclass
class MultiConceptConfig:
    """Configuration for multi-concept modification."""

    orthogonalize: bool = True
    """Orthogonalize concept directions to minimize interference."""

    orthogonalization_order: str = "priority"
    """Order for Gram-Schmidt: 'priority', 'variance', or 'sequential'."""

    max_interference: float = 0.3
    """Maximum allowed cosine similarity between concept directions."""

    warn_on_interference: bool = True
    """Warn if concept directions have high interference."""

    components: Optional[List[str]] = None
    """Weight components to modify. Default: attention out-proj + MLP down-proj."""

    norm_preserve: bool = True
    """Use norm-preserving projection."""

    use_null_space: bool = False
    """Use null-space projection to prevent interference with preserved activations."""

    null_space_epsilon: float = 1e-6
    """Tikhonov regularization for null-space projector SVD."""

    null_space_max_rank: Optional[int] = None
    """Optional SVD rank truncation for null-space projector."""

    accumulate_keys_across_concepts: bool = True
    """Accumulate preserved keys from each concept for subsequent concepts."""

    verbose: bool = True
    """Print progress information."""


@dataclass
class MultiConceptResult:
    """Result of multi-concept modification."""

    concepts_modified: List[str]
    """Names of concepts that were modified."""

    layers_modified: int
    """Total layers modified."""

    total_parameters_modified: int
    """Total parameters modified."""

    interference_matrix: Dict[Tuple[str, str], float]
    """Pairwise interference (cosine similarity) between concepts."""

    orthogonalized: bool
    """Whether directions were orthogonalized."""

    per_concept_stats: Dict[str, Dict[str, Any]]
    """Per-concept modification statistics."""

    warnings: List[str]
    """Any warnings generated during modification."""
