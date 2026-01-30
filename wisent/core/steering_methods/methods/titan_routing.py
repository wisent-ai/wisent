"""
Input-dependent direction routing for TITAN.

This module adds concept-aware steering by predicting which directions
to use based on the input activation pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from dataclasses import dataclass


class DirectionRoutingNetwork(nn.Module):
    """
    Predicts per-layer direction weights based on input activation.

    This enables concept-specific steering: different inputs can use
    different combinations of the learned directions.
    """

    def __init__(
        self,
        input_dim: int,
        num_directions: int,
        num_layers: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_directions = num_directions
        self.num_layers = num_layers

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Per-layer routing heads
        self.routing_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_directions)
            for _ in range(num_layers)
        ])

        # Temperature for softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, h: torch.Tensor) -> List[torch.Tensor]:
        """
        Predict direction weights for each layer.

        Args:
            h: Hidden state [batch, hidden_dim] or [hidden_dim]

        Returns:
            List of weight tensors, one per layer, each [batch, num_directions]
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)

        # Encode input
        encoded = self.encoder(h)

        # Get per-layer weights
        weights = []
        for head in self.routing_heads:
            logits = head(encoded)
            # Temperature-scaled softmax for sharper/softer routing
            w = F.softmax(logits / (self.temperature.abs() + 0.1), dim=-1)
            weights.append(w)

        return weights

    def get_weights_dict(
        self, h: torch.Tensor, layer_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Get weights as a dictionary keyed by layer name."""
        weights_list = self.forward(h)
        return {name: w for name, w in zip(layer_names, weights_list)}


@dataclass
class RoutingAnalysis:
    """Analysis of how routing network assigns directions to different inputs."""

    # Per-concept routing patterns
    concept_routing: Dict[int, Dict[str, torch.Tensor]]
    """For each concept ID, the average routing weights per layer."""

    # Routing diversity
    routing_entropy: float
    """Average entropy of routing decisions (higher = more diverse)."""

    # Dominant directions per concept
    dominant_directions: Dict[int, Dict[str, int]]
    """For each concept, which direction has highest weight per layer."""


def analyze_routing(
    routing_network: DirectionRoutingNetwork,
    activations: Dict[int, torch.Tensor],
    concept_assignments: Dict[int, int],
    layer_names: List[str],
) -> RoutingAnalysis:
    """
    Analyze how the routing network assigns directions to different concepts.

    Args:
        routing_network: Trained routing network
        activations: Dict mapping sample index to activation tensor
        concept_assignments: Dict mapping sample index to concept ID
        layer_names: List of layer names

    Returns:
        RoutingAnalysis with per-concept routing patterns
    """
    # Group activations by concept
    concept_activations: Dict[int, List[torch.Tensor]] = {}
    for idx, act in activations.items():
        concept_id = concept_assignments.get(idx, -1)
        if concept_id not in concept_activations:
            concept_activations[concept_id] = []
        concept_activations[concept_id].append(act)

    # Compute average routing per concept
    concept_routing = {}
    dominant_directions = {}
    all_entropies = []

    routing_network.eval()
    with torch.no_grad():
        for concept_id, acts in concept_activations.items():
            acts_tensor = torch.stack(acts)

            # Get routing weights for all samples in this concept
            weights_list = routing_network(acts_tensor)

            # Average across samples
            avg_weights = {}
            dominant = {}
            for layer_name, weights in zip(layer_names, weights_list):
                avg_w = weights.mean(dim=0)
                avg_weights[layer_name] = avg_w
                dominant[layer_name] = avg_w.argmax().item()

                # Compute entropy
                entropy = -(avg_w * (avg_w + 1e-8).log()).sum().item()
                all_entropies.append(entropy)

            concept_routing[concept_id] = avg_weights
            dominant_directions[concept_id] = dominant

    return RoutingAnalysis(
        concept_routing=concept_routing,
        routing_entropy=sum(all_entropies) / len(all_entropies) if all_entropies else 0.0,
        dominant_directions=dominant_directions,
    )


def compute_routing_diversity_loss(
    routing_weights: List[torch.Tensor],
    concept_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Loss to encourage diverse routing patterns across concepts.

    If concept_labels provided, encourages different concepts to use
    different directions. Otherwise, just encourages non-uniform routing.
    """
    if concept_labels is None:
        # Just encourage concentration (non-uniform weights)
        total_loss = torch.tensor(0.0)
        for weights in routing_weights:
            # Negative entropy = encourage concentration
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
            total_loss = total_loss - entropy  # Minimize entropy
        return total_loss / len(routing_weights)

    # With concept labels: encourage different concepts to route differently
    # Group by concept and compute cross-concept diversity
    unique_concepts = concept_labels.unique()
    if len(unique_concepts) <= 1:
        return torch.tensor(0.0)

    # Compute mean routing per concept
    concept_means = []
    for c in unique_concepts:
        mask = concept_labels == c
        if mask.sum() > 0:
            means = [w[mask].mean(dim=0) for w in routing_weights]
            concept_means.append(torch.stack(means))

    if len(concept_means) < 2:
        return torch.tensor(0.0)

    # Encourage different concepts to have different routing
    # Use negative cosine similarity between concept routing patterns
    diversity_loss = torch.tensor(0.0)
    count = 0
    for i in range(len(concept_means)):
        for j in range(i + 1, len(concept_means)):
            cos_sim = F.cosine_similarity(
                concept_means[i].flatten().unsqueeze(0),
                concept_means[j].flatten().unsqueeze(0),
            )
            diversity_loss = diversity_loss + cos_sim  # Minimize similarity
            count += 1

    return diversity_loss / count if count > 0 else torch.tensor(0.0)
