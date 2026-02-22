"""
TETNO - Probabilistic Uncertainty-guided Layer Steering Engine.

A layer-adaptive conditional steering method that combines:
- Condition-based gating (from CAST paper)
- Uncertainty-guided intensity modulation (from DAC paper)
- Multi-layer steering with learned per-layer scaling

Key innovations:
1. Sensor layer detects when steering should activate via condition vector
2. Uncertainty (entropy/KL) determines steering intensity
3. Steering applied to configurable layer range with per-layer scaling
4. Supports multiple behaviors with independent conditions

Based on insights from:
- "Dynamic Activation Composition for Multifaceted Steering" (DAC)
- "Conditional Activation Steering" (CAST)
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F

from wisent.core.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.errors import InsufficientDataError

__all__ = [
    "TETNOMethod",
    "TETNOConfig",
    "TETNOResult",
]


@dataclass
class TETNOConfig:
    """Configuration for TETNO steering method."""
    
    # Layer configuration
    sensor_layer: Optional[int] = None
    """Layer index where condition gating is computed. If None, auto-computed from num_layers."""
    
    steering_layers: Optional[List[int]] = None
    """Layer indices where steering is applied. If None, auto-computed from num_layers."""
    
    num_layers: Optional[int] = None
    """Total layers in the model. Used to auto-compute steering_layers and sensor_layer."""
    
    per_layer_scaling: bool = True
    """Whether to learn/use different scaling per layer."""
    
    def resolve_layers(self, num_layers: int) -> None:
        """Resolve steering_layers and sensor_layer based on model's num_layers."""
        self.num_layers = num_layers
        if self.sensor_layer is None:
            # 75% through the network
            self.sensor_layer = int(num_layers * 0.75)
        if self.steering_layers is None:
            # Middle to late layers (50% to 85% of network)
            start = int(num_layers * 0.5)
            end = int(num_layers * 0.85)
            self.steering_layers = list(range(start, end))
    
    # Condition gating
    condition_threshold: float = 0.5
    """Threshold for condition activation (0-1)."""
    
    gate_temperature: float = 0.1
    """Temperature for sigmoid gating (lower = sharper)."""
    
    learn_threshold: bool = True
    """Whether to learn optimal threshold via grid search."""
    
    # Uncertainty-guided intensity
    use_entropy_scaling: bool = True
    """Enable entropy-based intensity modulation."""
    
    entropy_floor: float = 0.5
    """Minimum entropy to trigger scaling (below = no steering)."""
    
    entropy_ceiling: float = 2.0
    """Entropy at which max_alpha is reached."""
    
    max_alpha: float = 2.0
    """Maximum steering strength."""
    
    # Training
    optimization_steps: int = 100
    """Steps for condition vector optimization."""
    
    learning_rate: float = 0.01
    """Learning rate for optimization."""
    
    use_caa_init: bool = True
    """Initialize behavior vectors using CAA."""
    
    normalize: bool = True
    """L2-normalize vectors."""
    
    # Threshold search
    threshold_search_steps: int = 20
    """Number of threshold values to try in grid search."""


@dataclass
class TETNOResult:
    """Result containing TETNO steering components."""
    
    behavior_vectors: Dict[LayerName, torch.Tensor]
    """Per-layer behavior/steering vectors."""
    
    condition_vector: torch.Tensor
    """Condition vector for gating (at sensor layer)."""
    
    layer_scales: Dict[LayerName, float]
    """Per-layer scaling factors."""
    
    optimal_threshold: float
    """Learned or configured threshold."""
    
    metadata: Dict[str, Any]
    """Training metadata and diagnostics."""
    
    def get_behavior_vector(self, layer: LayerName) -> Optional[torch.Tensor]:
        """Get behavior vector for a specific layer."""
        return self.behavior_vectors.get(layer)
    
    def get_layer_scale(self, layer: LayerName) -> float:
        """Get scaling factor for a layer."""
        return self.layer_scales.get(layer, 1.0)
    
    def should_steer(self, hidden_state: torch.Tensor, threshold: Optional[float] = None) -> Tuple[bool, float]:
        """
        Determine if steering should activate based on condition.
        
        Args:
            hidden_state: Hidden state at sensor layer [batch, seq, hidden] or [hidden]
            threshold: Override threshold (uses optimal_threshold if None)
            
        Returns:
            Tuple of (should_steer: bool, gate_value: float)
        """
        thresh = threshold if threshold is not None else self.optimal_threshold
        
        # Flatten to [hidden] if needed
        if hidden_state.dim() > 1:
            hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1]).mean(dim=0)
        
        # Cosine similarity with condition vector
        h_norm = F.normalize(hidden_state, p=2, dim=-1)
        c_norm = F.normalize(self.condition_vector, p=2, dim=-1)
        similarity = (h_norm * c_norm).sum()
        
        return similarity.item() > thresh, similarity.item()
    
    def compute_gate(self, hidden_state: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """
        Compute soft gate value for steering.
        
        Args:
            hidden_state: Hidden state at sensor layer
            temperature: Sigmoid temperature
            
        Returns:
            Gate value in [0, 1]
        """
        if hidden_state.dim() > 1:
            hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1]).mean(dim=0)
        
        h_norm = F.normalize(hidden_state, p=2, dim=-1)
        c_norm = F.normalize(self.condition_vector, p=2, dim=-1)
        similarity = (h_norm * c_norm).sum()
        
        gate = torch.sigmoid((similarity - self.optimal_threshold) / temperature)
        return gate

