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
from wisent.core.contrastive_pairs.set import ContrastivePairSet
from wisent.core.errors import InsufficientDataError
from wisent.core.constants import (
    TETNO_CONDITION_THRESHOLD,
    TETNO_GATE_TEMPERATURE,
    TETNO_ENTROPY_FLOOR,
    TETNO_ENTROPY_CEILING,
    TETNO_MAX_ALPHA,
    DEFAULT_OPTIMIZATION_STEPS,
    TECZA_LEARNING_RATE,
    TETNO_THRESHOLD_SEARCH_STEPS,
)

__all__ = [
    "TETNOMethod",
    "TETNOConfig",
    "TETNOResult",
]

from wisent.core.steering_methods.methods.advanced._tetno_types import (
    TETNOConfig, TETNOResult,
)
from wisent.core.steering_methods.methods.advanced._tetno_training import TETNOTrainingMixin
from wisent.core.steering_methods.methods.advanced._tetno_scaling import TETNOScalingMixin

class TETNOMethod(TETNOTrainingMixin, TETNOScalingMixin, BaseSteeringMethod):
    """
    TETNO - Probabilistic Uncertainty-guided Layer Steering Engine.
    
    A layer-adaptive conditional steering method that:
    - Detects when to steer via condition vector at sensor layer
    - Modulates intensity based on model uncertainty
    - Applies steering across multiple layers with per-layer scaling
    
    Usage:
        method = TETNOMethod(sensor_layer=16, steering_layers=[13,14,15,16,17,18,19])
        result = method.train_tetno(behavior_pairs, condition_pairs)
        
        # At inference:
        gate = result.compute_gate(hidden_at_sensor)
        for layer in steering_layers:
            h[layer] += gate * intensity * result.layer_scales[layer] * result.behavior_vectors[layer]
    """
    
    name = "tetno"
    description = "Layer-adaptive conditional steering with uncertainty-guided intensity"
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # steering_layers and sensor_layer default to None - resolved at training time
        # based on actual num_layers in the model
        self.config = TETNOConfig(
            sensor_layer=kwargs.get("sensor_layer", None),  # Auto-resolve from num_layers
            steering_layers=kwargs.get("steering_layers", None),  # Auto-resolve from num_layers
            num_layers=kwargs.get("num_layers", None),
            per_layer_scaling=kwargs.get("per_layer_scaling", True),
            condition_threshold=kwargs.get("condition_threshold", TETNO_CONDITION_THRESHOLD),
            gate_temperature=kwargs.get("gate_temperature", TETNO_GATE_TEMPERATURE),
            learn_threshold=kwargs.get("learn_threshold", True),
            use_entropy_scaling=kwargs.get("use_entropy_scaling", True),
            entropy_floor=kwargs.get("entropy_floor", TETNO_ENTROPY_FLOOR),
            entropy_ceiling=kwargs.get("entropy_ceiling", TETNO_ENTROPY_CEILING),
            max_alpha=kwargs.get("max_alpha", TETNO_MAX_ALPHA),
            optimization_steps=kwargs.get("optimization_steps", DEFAULT_OPTIMIZATION_STEPS),
            learning_rate=kwargs.get("learning_rate", TECZA_LEARNING_RATE),
            use_caa_init=kwargs.get("use_caa_init", True),
            normalize=kwargs.get("normalize", True),
            threshold_search_steps=kwargs.get("threshold_search_steps", TETNO_THRESHOLD_SEARCH_STEPS),
        )
        self._training_logs: List[Dict[str, float]] = []
    
    def train(self, pair_set: ContrastivePairSet) -> LayerActivations:
        """
        Train TETNO from contrastive pairs (simplified interface).
        
        Uses the same pairs for both behavior and condition training.
        For full control, use train_tetno() instead.
        
        Args:
            pair_set: ContrastivePairSet with collected activations.
            
        Returns:
            LayerActivations with behavior vectors (for backward compatibility).
        """
        result = self.train_tetno(pair_set)
        
        # Return behavior vectors as LayerActivations
        dtype = self.kwargs.get("dtype", None)
        return LayerActivations(result.behavior_vectors, dtype=dtype)
    
    def train_tetno(
        self,
        behavior_pairs: ContrastivePairSet,
        condition_pairs: Optional[ContrastivePairSet] = None,
    ) -> TETNOResult:
        """
        Full TETNO training with separate behavior and condition pairs.
        
        Args:
            behavior_pairs: Pairs for training behavior vectors (what to steer)
            condition_pairs: Pairs for training condition vector (when to steer)
                            If None, uses behavior_pairs for both.
        
        Returns:
            TETNOResult with all trained components.
        """
        if condition_pairs is None:
            condition_pairs = behavior_pairs
        
        # Detect num_layers from available data and resolve config
        buckets = self._collect_from_set(behavior_pairs)
        if buckets:
            max_layer_idx = 0
            for layer_name in buckets.keys():
                try:
                    layer_idx = int(str(layer_name).split("_")[-1])
                    max_layer_idx = max(max_layer_idx, layer_idx)
                except (ValueError, IndexError):
                    pass
            detected_num_layers = max_layer_idx + 1
            if self.config.steering_layers is None or self.config.sensor_layer is None:
                self.config.resolve_layers(detected_num_layers)
        
        # 1. Train behavior vectors for steering layers
        behavior_vectors = self._train_behavior_vectors(behavior_pairs)
        
        if not behavior_vectors:
            raise InsufficientDataError(reason="No behavior vectors could be trained")
        
        # 2. Train condition vector at sensor layer
        condition_vector = self._train_condition_vector(condition_pairs)
        
        # 3. Learn per-layer scaling
        layer_scales = self._compute_layer_scales(behavior_vectors, behavior_pairs)
        
        # 4. Optionally learn optimal threshold
        if self.config.learn_threshold:
            optimal_threshold = self._learn_threshold(
                condition_vector, condition_pairs
            )
        else:
            optimal_threshold = self.config.condition_threshold
        
        return TETNOResult(
            behavior_vectors=behavior_vectors,
            condition_vector=condition_vector,
            layer_scales=layer_scales,
            optimal_threshold=optimal_threshold,
            metadata={
                "config": self.config.__dict__,
                "num_steering_layers": len(behavior_vectors),
                "training_logs": self._training_logs,
            }
        )
    
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy from logits.
    
    Args:
        logits: Raw logits [batch, vocab] or [vocab]
        
    Returns:
        Entropy value(s)
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def compute_intensity_from_entropy(
    entropy: torch.Tensor,
    floor: float = TETNO_ENTROPY_FLOOR,
    ceiling: float = TETNO_ENTROPY_CEILING,
    max_alpha: float = TETNO_MAX_ALPHA,
) -> torch.Tensor:
    """
    Compute steering intensity from entropy.
    
    Higher entropy → higher intensity (model is uncertain → steer more)
    
    Args:
        entropy: Entropy value(s)
        floor: Min entropy (below = 0 intensity)
        ceiling: Max entropy (above = max intensity)
        max_alpha: Maximum steering strength
        
    Returns:
        Intensity value(s) in [0, max_alpha]
    """
    # Normalize to [0, 1] range
    normalized = (entropy - floor) / (ceiling - floor)
    normalized = torch.clamp(normalized, 0.0, 1.0)
    
    # Scale to [0, max_alpha]
    return normalized * max_alpha
