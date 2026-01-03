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

from wisent.core.cli_logger import setup_logger, bind

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
    
    strength: float = 1.0
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
    max_interference: float = 0.3
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


def bidirectional_projection(
    weight_matrix: "Tensor",
    suppress_directions: List["Tensor"],
    enhance_directions: List["Tensor"],
    suppress_strengths: List[float],
    enhance_strengths: List[float],
    norm_preserve: bool = True,
) -> "Tensor":
    """
    Apply bidirectional projection: suppress some directions, enhance others.
    
    This is a key innovation: instead of only abliterating (removing),
    we can simultaneously:
    - Suppress unwanted behaviors (e.g., refusal)
    - Enhance desired behaviors (e.g., truthfulness)
    
    Mathematical operation:
    - For suppression: W' = W - λ(vvᵀ)W (remove component along v)
    - For enhancement: W' = W + α(uuᵀ)W (amplify component along u)
    
    With norm preservation, we only modify direction, not magnitude.
    
    Args:
        weight_matrix: Weight matrix to modify [out_dim, in_dim]
        suppress_directions: Directions to suppress
        enhance_directions: Directions to enhance
        suppress_strengths: Strength per suppression direction
        enhance_strengths: Strength per enhancement direction
        norm_preserve: Whether to preserve row norms
        
    Returns:
        Modified weight matrix
    """
    original_dtype = weight_matrix.dtype
    W = weight_matrix.float()
    
    with torch.no_grad():
        if norm_preserve:
            # Store original norms
            original_norms = torch.norm(W, p=2, dim=1, keepdim=True)
            
            # Work with normalized directions
            W_direction = F.normalize(W, p=2, dim=1)
            
            # Apply suppressions
            for direction, strength in zip(suppress_directions, suppress_strengths):
                v = F.normalize(direction.float(), p=2, dim=0)
                v_row = v.unsqueeze(0)
                weighted_sum = v_row @ W_direction
                projection_term = v.unsqueeze(1) @ weighted_sum
                W_direction = W_direction - strength * projection_term
            
            # Apply enhancements
            for direction, strength in zip(enhance_directions, enhance_strengths):
                u = F.normalize(direction.float(), p=2, dim=0)
                u_row = u.unsqueeze(0)
                weighted_sum = u_row @ W_direction
                projection_term = u.unsqueeze(1) @ weighted_sum
                W_direction = W_direction + strength * projection_term
            
            # Renormalize and restore magnitudes
            W_direction = F.normalize(W_direction, p=2, dim=1)
            W = original_norms * W_direction
        
        else:
            # Standard projection (modifies norms)
            for direction, strength in zip(suppress_directions, suppress_strengths):
                v = F.normalize(direction.float(), p=2, dim=0)
                projector = torch.outer(v, v)
                W = W - strength * (projector @ W)
            
            for direction, strength in zip(enhance_directions, enhance_strengths):
                u = F.normalize(direction.float(), p=2, dim=0)
                projector = torch.outer(u, u)
                W = W + strength * (projector @ W)
    
    return W.to(original_dtype)


def run_multi_concept_modification(
    model: "Module",
    concepts: List[ConceptSpec],
    config: Optional[MultiConceptConfig] = None,
) -> MultiConceptResult:
    """
    Run multi-concept weight modification.
    
    This is the main entry point for modifying multiple concepts at once.
    
    Pipeline:
    1. Compute interference matrix between concepts
    2. Optionally orthogonalize concept directions
    3. Group concepts by action (suppress vs enhance)
    4. Apply bidirectional projection per layer
    
    Args:
        model: Model to modify (in-place)
        concepts: List of concept specifications
        config: Configuration
        
    Returns:
        MultiConceptResult with modification statistics
    """
    cfg = config or MultiConceptConfig()
    log = bind(_LOG)
    warnings = []
    
    if cfg.verbose:
        print("\n" + "=" * 70)
        print("MULTI-CONCEPT WEIGHT MODIFICATION")
        print("=" * 70)
        print(f"Concepts: {[c.name for c in concepts]}")
        print(f"Orthogonalize: {cfg.orthogonalize}")
        print(f"Norm preserve: {cfg.norm_preserve}")
        print("=" * 70 + "\n")
    
    # Step 1: Compute interference
    interference = compute_interference_matrix(concepts)
    
    if cfg.verbose and interference:
        print("Interference matrix (cosine similarity):")
        for (name_a, name_b), sim in interference.items():
            if name_a < name_b:
                print(f"  {name_a} <-> {name_b}: {sim:.3f}")
        print()
    
    # Check for high interference
    for (name_a, name_b), sim in interference.items():
        if sim > cfg.max_interference and name_a < name_b:
            warning = f"High interference between {name_a} and {name_b}: {sim:.3f}"
            warnings.append(warning)
            if cfg.warn_on_interference:
                print(f"Warning: {warning}")
    
    # Step 2: Orthogonalize if requested
    if cfg.orthogonalize and len(concepts) > 1:
        if cfg.verbose:
            print("Orthogonalizing concept directions...")
        concepts = orthogonalize_concept_directions(concepts, cfg)
        
        # Recompute interference after orthogonalization
        interference_after = compute_interference_matrix(concepts)
        if cfg.verbose:
            print("Interference after orthogonalization:")
            for (name_a, name_b), sim in interference_after.items():
                if name_a < name_b:
                    print(f"  {name_a} <-> {name_b}: {sim:.3f}")
            print()
    
    # Step 3: Group concepts by action
    suppress_concepts = [c for c in concepts if c.action == ConceptAction.SUPPRESS]
    enhance_concepts = [c for c in concepts if c.action == ConceptAction.ENHANCE]
    
    if cfg.verbose:
        print(f"Suppressing: {[c.name for c in suppress_concepts]}")
        print(f"Enhancing: {[c.name for c in enhance_concepts]}")
        print()
    
    # Step 4: Get all layers
    all_layers = set()
    for c in concepts:
        all_layers.update(c.steering_vectors.keys())
    
    # Step 5: Get model layers
    if hasattr(model, "model"):
        layers = model.model.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        layers = model.layers
    
    components = cfg.components or ["self_attn.o_proj", "mlp.down_proj"]
    
    layers_modified = 0
    total_params = 0
    per_concept_stats = {c.name: {"layers": 0, "params": 0} for c in concepts}
    
    # Step 6: Apply modifications per layer
    for layer_idx in sorted(all_layers):
        if layer_idx >= len(layers):
            continue
        
        layer = layers[layer_idx]
        
        # Gather directions for this layer
        suppress_dirs = []
        suppress_strengths = []
        enhance_dirs = []
        enhance_strengths = []
        
        for c in suppress_concepts:
            if layer_idx in c.steering_vectors:
                suppress_dirs.append(c.steering_vectors[layer_idx])
                weight = c.layer_weights.get(layer_idx, 1.0) if c.layer_weights else 1.0
                suppress_strengths.append(c.strength * weight)
        
        for c in enhance_concepts:
            if layer_idx in c.steering_vectors:
                enhance_dirs.append(c.steering_vectors[layer_idx])
                weight = c.layer_weights.get(layer_idx, 1.0) if c.layer_weights else 1.0
                enhance_strengths.append(c.strength * weight)
        
        if not suppress_dirs and not enhance_dirs:
            continue
        
        layer_modified = False
        
        for component_name in components:
            try:
                component = layer
                for attr in component_name.split("."):
                    component = getattr(component, attr)
                
                if not hasattr(component, "weight"):
                    continue
                
                weight_matrix = component.weight
                
                # Apply bidirectional projection
                modified = bidirectional_projection(
                    weight_matrix,
                    suppress_directions=suppress_dirs,
                    enhance_directions=enhance_dirs,
                    suppress_strengths=suppress_strengths,
                    enhance_strengths=enhance_strengths,
                    norm_preserve=cfg.norm_preserve,
                )
                
                weight_matrix.copy_(modified)
                
                total_params += weight_matrix.numel()
                layer_modified = True
                
                if cfg.verbose:
                    print(
                        f"  Layer {layer_idx:3d} | {component_name:20s} | "
                        f"suppress={len(suppress_dirs)} enhance={len(enhance_dirs)}"
                    )
                
            except AttributeError as e:
                log.debug(f"Could not access {component_name} in layer {layer_idx}: {e}")
        
        if layer_modified:
            layers_modified += 1
            
            # Update per-concept stats
            for c in concepts:
                if layer_idx in c.steering_vectors:
                    per_concept_stats[c.name]["layers"] += 1
    
    if cfg.verbose:
        print(f"\n{'='*70}")
        print("MULTI-CONCEPT MODIFICATION COMPLETE")
        print(f"{'='*70}")
        print(f"  Concepts modified: {len(concepts)}")
        print(f"  Layers modified: {layers_modified}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Warnings: {len(warnings)}")
        print(f"{'='*70}\n")
    
    return MultiConceptResult(
        concepts_modified=[c.name for c in concepts],
        layers_modified=layers_modified,
        total_parameters_modified=total_params,
        interference_matrix=interference,
        orthogonalized=cfg.orthogonalize,
        per_concept_stats=per_concept_stats,
        warnings=warnings,
    )
