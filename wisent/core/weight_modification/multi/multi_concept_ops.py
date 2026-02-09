"""Multi-concept operations: interference, orthogonalization, bidirectional projection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, TYPE_CHECKING

from .multi_concept_types import ConceptSpec, MultiConceptConfig

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "compute_interference_matrix",
    "orthogonalize_concept_directions",
    "bidirectional_projection",
]


def compute_interference_matrix(
    concepts: List[ConceptSpec],
) -> Dict[Tuple[str, str], float]:
    """Compute pairwise interference between concept directions.

    Interference is measured as the mean absolute cosine similarity
    between the steering vectors across all shared layers.
    """
    interference = {}

    for i, concept_a in enumerate(concepts):
        for j, concept_b in enumerate(concepts):
            if i >= j:
                continue

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
    """Orthogonalize concept directions using Gram-Schmidt process.

    Earlier concepts in the ordering are preserved more faithfully,
    later concepts are adjusted more.
    """
    if len(concepts) < 2:
        return concepts

    if config.orthogonalization_order == "priority":
        ordered = sorted(concepts, key=lambda c: -c.priority)
    elif config.orthogonalization_order == "variance":
        def total_magnitude(c):
            return sum(v.norm().item() for v in c.steering_vectors.values())
        ordered = sorted(concepts, key=total_magnitude, reverse=True)
    else:
        ordered = concepts.copy()

    orthogonalized_concepts = []

    for concept in ordered:
        new_vectors = {}

        for layer in concept.steering_vectors.keys():
            vec = concept.steering_vectors[layer].float()

            for prev_concept in orthogonalized_concepts:
                if layer in prev_concept.steering_vectors:
                    prev_vec = prev_concept.steering_vectors[layer].float()
                    prev_vec_norm = F.normalize(prev_vec, p=2, dim=0)
                    projection = (vec @ prev_vec_norm) * prev_vec_norm
                    vec = vec - projection

            if concept.steering_vectors[layer].norm() > 0:
                new_vectors[layer] = vec

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
    """Apply bidirectional projection: suppress some directions, enhance others.

    - For suppression: W' = W - s*(vv^T)W
    - For enhancement: W' = W + s*(uu^T)W

    With norm preservation, only direction is modified, not magnitude.
    """
    original_dtype = weight_matrix.dtype
    W = weight_matrix.float()

    with torch.no_grad():
        if norm_preserve:
            original_norms = torch.norm(W, p=2, dim=1, keepdim=True)
            W_direction = F.normalize(W, p=2, dim=1)

            for direction, strength in zip(suppress_directions, suppress_strengths):
                v = F.normalize(direction.float(), p=2, dim=0)
                v_row = v.unsqueeze(0)
                weighted_sum = v_row @ W_direction
                projection_term = v.unsqueeze(1) @ weighted_sum
                W_direction = W_direction - strength * projection_term

            for direction, strength in zip(enhance_directions, enhance_strengths):
                u = F.normalize(direction.float(), p=2, dim=0)
                u_row = u.unsqueeze(0)
                weighted_sum = u_row @ W_direction
                projection_term = u.unsqueeze(1) @ weighted_sum
                W_direction = W_direction + strength * projection_term

            W_direction = F.normalize(W_direction, p=2, dim=1)
            W = original_norms * W_direction

        else:
            for direction, strength in zip(suppress_directions, suppress_strengths):
                v = F.normalize(direction.float(), p=2, dim=0)
                projector = torch.outer(v, v)
                W = W - strength * (projector @ W)

            for direction, strength in zip(enhance_directions, enhance_strengths):
                u = F.normalize(direction.float(), p=2, dim=0)
                projector = torch.outer(u, u)
                W = W + strength * (projector @ W)

    return W.to(original_dtype)
