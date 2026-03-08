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

from wisent.core.control.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.control.steering_methods.configs.optimal import get_optimal
from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.utils.infra_tools.errors import InsufficientDataError
from wisent.core.utils.infra_tools.errors import InsufficientDataError

__all__ = [
    "TETNOMethod",
    "TETNOConfig",
    "TETNOResult",
]


@dataclass
class TETNOConfig:
    """Configuration for TETNO steering method."""

    # Required fields (no defaults — must come from optimizer or explicit config)
    condition_threshold: float
    """Threshold for condition activation."""

    gate_temperature: float
    """Temperature for sigmoid gating (lower = sharper)."""

    max_alpha: float
    """Maximum steering strength."""

    optimization_steps: int
    """Steps for condition vector optimization."""

    learning_rate: float
    """Learning rate for optimization."""

    entropy_floor: float
    """Minimum entropy to trigger scaling (below = no steering)."""

    entropy_ceiling: float
    """Entropy at which max_alpha is reached."""

    threshold_search_steps: int
    """Number of threshold values to try in grid search."""

    condition_margin: float
    """Margin for condition vector optimization loss."""

    min_layer_scale: float
    """Minimum per-layer scaling factor."""

    # Optional fields with defaults
    sensor_layer: Optional[int] = None
    """Layer index where condition gating is computed. If None, auto-computed."""

    steering_layers: Optional[List[int]] = None
    """Layer indices where steering is applied. If None, auto-computed."""

    num_layers: Optional[int] = None
    """Total layers in the model. Used to auto-compute steering_layers and sensor_layer."""

    per_layer_scaling: bool = field(default_factory=lambda: get_optimal("per_layer_scaling"))
    """Whether to learn/use different scaling per layer."""

    learn_threshold: bool = field(default_factory=lambda: get_optimal("learn_threshold"))
    """Whether to learn optimal threshold via grid search."""

    use_entropy_scaling: bool = field(default_factory=lambda: get_optimal("use_entropy_scaling"))
    """Enable entropy-based intensity modulation."""

    use_caa_init: bool = field(default_factory=lambda: get_optimal("use_caa_init"))
    """Initialize behavior vectors using CAA."""

    normalize: bool = field(default_factory=lambda: get_optimal("normalize"))
    """L2-normalize vectors."""

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
        if layer not in self.layer_scales:
            raise KeyError(f"No scale for layer {layer}. Available: {list(self.layer_scales.keys())}")
        return self.layer_scales[layer]
    
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
    
    def compute_gate(self, hidden_state: torch.Tensor, temperature: float) -> torch.Tensor:
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

