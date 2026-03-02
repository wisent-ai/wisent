"""
Multi-Concept Weight Modification.

This module implements simultaneous modification of multiple concepts
in a single pass. Key innovations:

1. Multi-Concept Orthogonalization: Abliterate multiple concepts at once
   (e.g., refusal + sycophancy + hallucination) while handling interference.

2. Bidirectional Ablation: Suppress some directions while enhancing others
   (e.g., remove refusal, amplify truthfulness).

3. Interference Minimization: Ensure concept directions don't interfere
   with each other during joint modification.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from enum import Enum

from wisent.core.constants import DEFAULT_STRENGTH, WM_MAX_INTERFERENCE
from wisent.core.cli.cli_logger import setup_logger, bind
from wisent.core.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

__all__ = [
    "MultiConceptConfig",
    "ConceptSpec",
    "MultiConceptResult",
    "run_multi_concept_modification",
    "orthogonalize_concept_directions",
    "compute_interference_matrix",
    "bidirectional_projection",
]

_LOG = setup_logger(__name__)


class ConceptAction(Enum):
    """Action to take on a concept direction."""
    SUPPRESS = "suppress"  # Remove/abliterate the direction
    ENHANCE = "enhance"    # Amplify the direction
    NEUTRAL = "neutral"    # Don't modify (used for constraints)


@dataclass
class ConceptSpec:
    """Specification for a single concept to modify."""
    
    name: str
    """Human-readable name for the concept."""
    
    steering_vectors: Dict[int, "Tensor"]
    """Per-layer steering vectors for this concept."""
    
    action: ConceptAction = ConceptAction.SUPPRESS
    """Action to take: suppress, enhance, or neutral."""
    
    strength: float = DEFAULT_STRENGTH
    """Modification strength for this concept."""
    
    layer_weights: Optional[Dict[int, float]] = None
    """Optional per-layer weights. If None, uses uniform weights."""
    
    priority: int = 1
    """Priority for conflict resolution (higher = more important)."""


@dataclass
class MultiConceptConfig:
    """Configuration for multi-concept modification."""
    
    # Orthogonalization
    orthogonalize: bool = True
    """Orthogonalize concept directions to minimize interference."""
    
    orthogonalization_order: str = "priority"
    """Order for Gram-Schmidt: 'priority', 'variance', or 'sequential'."""
    
    # Interference handling
    max_interference: float = WM_MAX_INTERFERENCE
    """Maximum allowed cosine similarity between concept directions."""
    
    warn_on_interference: bool = True
    """Warn if concept directions have high interference."""
    
    # Components
    components: Optional[List[str]] = None
    """Weight components to modify. Default: attention out-proj + MLP down-proj."""
    
    # Norm preservation
    norm_preserve: bool = True
    """Use norm-preserving projection."""
    
    # General
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


def compute_interference_matrix(
    concepts: List[ConceptSpec],
) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise interference between concept directions.
    
    Interference is measured as the mean absolute cosine similarity
    between the steering vectors across all layers.
    """
    interference = {}
    
    for i, concept_a in enumerate(concepts):
        for j, concept_b in enumerate(concepts):
            if i >= j:
                continue
            
            # Compute mean cosine similarity across shared layers
            shared_layers = set(concept_a.steering_vectors.keys()) & set(concept_b.steering_vectors.keys())
            
            if not shared_layers:
                interference[(concept_a.name, concept_b.name)] = 0.0
                continue
            
            similarities = []
            for layer in shared_layers:
                vec_a = F.normalize(concept_a.steering_vectors[layer].float(), p=2, dim=0)
                vec_b = F.normalize(concept_b.steering_vectors[layer].float(), p=2, dim=0)
                cos_sim = (vec_a @ vec_b).abs().item()
                similarities.append(cos_sim)
            
            mean_sim = sum(similarities) / len(similarities)
            interference[(concept_a.name, concept_b.name)] = mean_sim
            interference[(concept_b.name, concept_a.name)] = mean_sim
    
    return interference


def orthogonalize_concept_directions(
    concepts: List[ConceptSpec],
    config: MultiConceptConfig,
) -> List[ConceptSpec]:
    """
    Orthogonalize concept directions using Gram-Schmidt process.
    
    This ensures that modifying one concept doesn't interfere with others.
    The order of orthogonalization matters - earlier concepts are preserved
    more faithfully, later concepts are adjusted more.
    
    Args:
        concepts: List of concept specifications
        config: Configuration
        
    Returns:
        List of concepts with orthogonalized steering vectors
    """
    if len(concepts) < 2:
        return concepts
    
    # Determine order
    if config.orthogonalization_order == "priority":
        ordered = sorted(concepts, key=lambda c: -c.priority)
    elif config.orthogonalization_order == "variance":
        # Order by total vector magnitude (proxy for variance explained)
        def total_magnitude(c):
            return sum(v.norm().item() for v in c.steering_vectors.values())
        ordered = sorted(concepts, key=total_magnitude, reverse=True)
    else:
        ordered = concepts.copy()
    
    # Get all layers
    all_layers = set()
    for c in ordered:
        all_layers.update(c.steering_vectors.keys())
    
    # Orthogonalize per layer
    orthogonalized_concepts = []
    
    for concept in ordered:
        new_vectors = {}
        
        for layer in concept.steering_vectors.keys():
            vec = concept.steering_vectors[layer].float()
            
            # Subtract projections onto previous concepts
            for prev_concept in orthogonalized_concepts:
                if layer in prev_concept.steering_vectors:
                    prev_vec = prev_concept.steering_vectors[layer].float()
                    prev_vec_norm = F.normalize(prev_vec, p=2, dim=0)
                    
                    # Gram-Schmidt: v' = v - (v . u) * u
                    projection = (vec @ prev_vec_norm) * prev_vec_norm
                    vec = vec - projection
            
            # Renormalize if originally normalized
            if concept.steering_vectors[layer].norm() > 0:
                new_vectors[layer] = vec
        
        # Create new concept spec with orthogonalized vectors
        orthogonalized_concepts.append(ConceptSpec(
            name=concept.name,
            steering_vectors=new_vectors,
            action=concept.action,
            strength=concept.strength,
            layer_weights=concept.layer_weights,
            priority=concept.priority,
        ))
    
    return orthogonalized_concepts



from wisent.core.weight_modification.multi._multi_concept_helpers import (
    bidirectional_projection,
    run_multi_concept_modification,
)
