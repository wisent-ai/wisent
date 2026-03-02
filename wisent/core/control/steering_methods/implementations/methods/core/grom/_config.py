"""GROM config, networks, and geometry adaptation."""
from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerName
from wisent.core.utils.config_tools.constants import (
    GROM_NUM_DIRECTIONS, GROM_OPTIMIZATION_STEPS, GROM_LEARNING_RATE,
    GROM_WARMUP_STEPS, GROM_BEHAVIOR_WEIGHT, GROM_RETAIN_WEIGHT,
    GROM_SPARSE_WEIGHT, GROM_SMOOTH_WEIGHT, GROM_INDEPENDENCE_WEIGHT,
    GROM_MAX_ALPHA, GROM_GATE_TEMPERATURE, GROM_MIN_COSINE_SIM,
    GROM_MAX_COSINE_SIM, GROM_LINEAR_THRESHOLD, GROM_HIDDEN_DIM,
    GROM_ROUTER_HIDDEN_DIM, GROM_INTENSITY_HIDDEN_DIM, GROM_ROUTER_TEMPERATURE,
    GROM_GATE_DIM_MIN, GROM_GATE_DIM_MAX, GROM_GATE_DIM_DIVISOR,
    GROM_INTENSITY_DIM_MIN, GROM_INTENSITY_DIM_MAX, GROM_INTENSITY_DIM_DIVISOR,
    GATING_HIDDEN_DIM_DIVISOR,
)

@dataclass
class GROMConfig:
    """Configuration for GROM steering method."""
    
    # Manifold configuration
    num_directions: int = GROM_NUM_DIRECTIONS
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
            raise ValueError(
                "sensor_layer must be specified explicitly. "
                "Pass an integer layer index."
            )
        if self.steering_layers is None:
            raise ValueError(
                "steering_layers must be specified explicitly. "
                "Pass a list of integer layer indices."
            )
    
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
            # Scale with model size, but clamp to reasonable range
            self.gate_hidden_dim = max(GROM_GATE_DIM_MIN, min(GROM_GATE_DIM_MAX, hidden_dim // GROM_GATE_DIM_DIVISOR))
        if self.intensity_hidden_dim is None:
            # Scale with model size, but clamp to reasonable range
            self.intensity_hidden_dim = max(GROM_INTENSITY_DIM_MIN, min(GROM_INTENSITY_DIM_MAX, hidden_dim // GROM_INTENSITY_DIM_DIVISOR))
    
    # Training
    optimization_steps: int = GROM_OPTIMIZATION_STEPS
    """Total optimization steps."""
    
    learning_rate: float = GROM_LEARNING_RATE
    """Learning rate for all components."""
    
    warmup_steps: int = GROM_WARMUP_STEPS
    """Steps to warmup (train manifold only before adding networks)."""
    
    # Loss weights
    behavior_weight: float = GROM_BEHAVIOR_WEIGHT
    """Weight for behavior effectiveness loss."""
    
    retain_weight: float = GROM_RETAIN_WEIGHT
    """Weight for retain loss (minimize side effects)."""
    
    sparse_weight: float = GROM_SPARSE_WEIGHT
    """Weight for sparsity loss (encourage sparse layer activation)."""
    
    smooth_weight: float = GROM_SMOOTH_WEIGHT
    """Weight for smoothness loss (penalize abrupt intensity changes)."""
    
    independence_weight: float = GROM_INDEPENDENCE_WEIGHT
    """Weight for direction independence loss."""
    
    # Constraints
    max_alpha: float = GROM_MAX_ALPHA
    """Maximum steering intensity."""
    
    gate_temperature: float = GROM_GATE_TEMPERATURE
    """Temperature for gate sigmoid."""
    
    min_cosine_similarity: float = GROM_MIN_COSINE_SIM
    """Minimum cosine similarity between directions."""
    
    max_cosine_similarity: float = GROM_MAX_COSINE_SIM
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
    
    linear_threshold: float = GROM_LINEAR_THRESHOLD
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
    
    def __init__(self, input_dim: int, hidden_dim: int = GROM_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // GATING_HIDDEN_DIM_DIVISOR),
            nn.GELU(),
            nn.Linear(hidden_dim // GATING_HIDDEN_DIM_DIVISOR, 1),
        )
    
    def forward(self, h: torch.Tensor, temperature: float = GROM_ROUTER_TEMPERATURE) -> torch.Tensor:
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
    
    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int = GROM_INTENSITY_HIDDEN_DIM, max_alpha: float = GROM_MAX_ALPHA):
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
    
    def __init__(self, input_dim: int, num_directions: int, hidden_dim: int = GROM_ROUTER_HIDDEN_DIM):
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
