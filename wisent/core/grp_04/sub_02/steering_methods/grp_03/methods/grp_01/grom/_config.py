"""GROM config, networks, and geometry adaptation."""
from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from wisent.core.activations.core.atoms import LayerName

@dataclass
class GROMConfig:
    """Configuration for GROM steering method."""
    
    # Manifold configuration
    num_directions: int = 5
    """Number of directions per layer in the steering manifold."""
    
    # Layer configuration  
    steering_layers: Optional[List[int]] = None
    """Layer indices where steering can be applied. If None, auto-computed from num_layers."""
    
    sensor_layer: Optional[int] = None
    """Primary layer for gating decisions. If None, auto-computed from num_layers."""
    
    num_layers: Optional[int] = None
    """Total layers in the model. Used to auto-compute steering_layers and sensor_layer."""
    
    def resolve_layers(self, num_layers: int) -> None:
        """Resolve steering_layers and sensor_layer based on model's num_layers."""
        self.num_layers = num_layers
        if self.sensor_layer is None:
            # 75% through the network
            self.sensor_layer = int(num_layers * 0.75)
        if self.steering_layers is None:
            # Middle to late layers (50% to 90% of network)
            start = int(num_layers * 0.5)
            end = int(num_layers * 0.9)
            self.steering_layers = list(range(start, end))
    
    # Network architecture
    gate_hidden_dim: Optional[int] = None
    """Hidden dimension for gating network. If None, auto-computed as hidden_dim // 16."""
    
    intensity_hidden_dim: Optional[int] = None
    """Hidden dimension for intensity network. If None, auto-computed as hidden_dim // 32."""

    routing_hidden_dim: Optional[int] = None
    """Hidden dimension for direction routing network. If None, auto-computed as hidden_dim // 32."""

    input_dependent_routing: bool = True
    """If True, use a network to predict direction weights per-input. If False, use fixed weights."""

    def resolve_network_dims(self, hidden_dim: int) -> None:
        """Resolve network dimensions based on model's hidden dimension."""
        if self.gate_hidden_dim is None:
            # Scale with model size, but clamp to reasonable range [32, 512]
            self.gate_hidden_dim = max(32, min(512, hidden_dim // 16))
        if self.intensity_hidden_dim is None:
            # Scale with model size, but clamp to reasonable range [16, 256]
            self.intensity_hidden_dim = max(16, min(256, hidden_dim // 32))
    
    # Training
    optimization_steps: int = 200
    """Total optimization steps."""
    
    learning_rate: float = 0.005
    """Learning rate for all components."""
    
    warmup_steps: int = 20
    """Steps to warmup (train manifold only before adding networks)."""
    
    # Loss weights
    behavior_weight: float = 1.0
    """Weight for behavior effectiveness loss."""
    
    retain_weight: float = 0.2
    """Weight for retain loss (minimize side effects)."""
    
    sparse_weight: float = 0.05
    """Weight for sparsity loss (encourage sparse layer activation)."""
    
    smooth_weight: float = 0.02
    """Weight for smoothness loss (penalize abrupt intensity changes)."""
    
    independence_weight: float = 0.03
    """Weight for direction independence loss."""
    
    # Constraints
    max_alpha: float = 3.0
    """Maximum steering intensity."""
    
    gate_temperature: float = 0.5
    """Temperature for gate sigmoid."""
    
    min_cosine_similarity: float = 0.2
    """Minimum cosine similarity between directions."""
    
    max_cosine_similarity: float = 0.9
    """Maximum cosine similarity (avoid redundancy)."""
    
    # Initialization
    use_caa_init: bool = True
    """Initialize primary direction with CAA."""
    
    normalize: bool = True
    """L2-normalize directions."""
    
    # Geometry-adaptive configuration
    adapt_to_geometry: bool = True
    """Whether to analyze geometry and adapt configuration."""
    
    geometry_analysis_layer: Optional[int] = None
    """Layer to use for geometry analysis. If None, uses sensor_layer."""
    
    linear_threshold: float = 0.8
    """If linear score > threshold, simplify to single direction."""
    
    skip_gating_if_linear: bool = True
    """Skip gating network if structure is clearly linear."""
    
    auto_num_directions: bool = True
    """Automatically determine num_directions based on geometry."""


class GatingNetwork(nn.Module):
    """
    Learned gating network that predicts whether steering should activate.
    
    Takes hidden states from sensor layer, outputs gate value in [0, 1].
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, h: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        """
        Predict gate value.
        
        Args:
            h: Hidden state [batch, hidden_dim] or [hidden_dim]
            temperature: Sigmoid temperature
            
        Returns:
            Gate value in [0, 1]
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)
        logit = self.net(h).squeeze(-1)
        return torch.sigmoid(logit / temperature)


class IntensityNetwork(nn.Module):
    """
    Learned intensity network that predicts per-layer steering strength.
    
    Takes hidden states, outputs intensity values for each steering layer.
    """
    
    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int = 64, max_alpha: float = 3.0):
        super().__init__()
        self.max_alpha = max_alpha
        self.num_layers = num_layers
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_layers),
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict per-layer intensity.
        
        Args:
            h: Hidden state [batch, hidden_dim] or [hidden_dim]
            
        Returns:
            Intensity values [batch, num_layers] in [0, max_alpha]
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)
        raw = self.net(h)
        return torch.sigmoid(raw) * self.max_alpha


class DirectionWeightNetwork(nn.Module):
    """
    Learned network that predicts weights for combining directions in manifold.
    """
    
    def __init__(self, input_dim: int, num_directions: int, hidden_dim: int = 32):
        super().__init__()
        self.num_directions = num_directions
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_directions),
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict direction weights.
        
        Args:
            h: Hidden state [batch, hidden_dim] or [hidden_dim]
            
        Returns:
            Softmax weights [batch, num_directions]
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)
        logits = self.net(h)
        return F.softmax(logits, dim=-1)


@dataclass
class GeometryAdaptation:
    """Results from geometry analysis and adaptations made."""
    
    detected_structure: str
    """Primary detected structure type (linear, cone, manifold, etc.)."""
    
    structure_scores: Dict[str, float]
    """Scores for all structure types."""
    
    adaptations_made: List[str]
    """List of adaptations applied based on geometry."""
    
    original_num_directions: int
    """Originally configured num_directions."""
    
    adapted_num_directions: int
    """Num directions after adaptation."""
    
    gating_enabled: bool
    """Whether gating network is enabled."""
    
    recommendation: str
    """Steering method recommendation based on geometry."""
