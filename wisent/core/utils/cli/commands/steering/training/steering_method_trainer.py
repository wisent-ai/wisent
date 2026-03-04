"""
Unified steering method training interface for CLI commands.

This module provides a unified interface for training any steering method
(CAA, TECZA, TETNO, GROM) that can be used by the optimize-steering CLI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch

from wisent.core.control.steering_methods.registry import SteeringMethodRegistry
from wisent.core.utils.config_tools.constants import ARCHITECTURE_MODULE_LIMIT
from wisent.core.primitives.model_interface.core.activations.activations_collector import ActivationCollector
from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations
from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.primitives.models.core.atoms import SteeringPlan, SteeringVector



from wisent.core.utils.cli.steering.training.steering_method_trainer_core import (
    get_method_params_from_args, train_steering_vector_for_layer,
    train_steering_vectors, collect_activations_for_pair_set,
    create_steering_plan_from_activations, create_steering_plan_single_layer,
)
from wisent.core.utils.cli.steering.training.steering_method_trainer_loading import (
    load_optimal_steering_config, create_steering_method_from_config,
    get_optimal_steering_plan,
)


class SteeringMethodWrapper:
    """
    A wrapper that provides a CAAMethod-compatible interface for any steering method.
    
    This allows drop-in replacement of CAAMethod in existing code:
    
        # Before:
        caa_method = CAAMethod(kwargs={"normalize": True})
        steering_vector = caa_method.train_for_layer(pos_acts, neg_acts)
        
        # After:
        method = create_steering_method(method_name, args)
        steering_vector = method.train_for_layer(pos_acts, neg_acts)
    """
    
    def __init__(self, method_name: str, method_params: Optional[Dict[str, Any]] = None):
        self.method_name = method_name.lower()
        self.method_params = method_params or {}
    
    def train_for_layer(
        self,
        pos_acts: List[torch.Tensor],
        neg_acts: List[torch.Tensor],
    ) -> torch.Tensor:
        """Train steering vector for a single layer - CAAMethod compatible interface."""
        return train_steering_vector_for_layer(
            self.method_name,
            pos_acts,
            neg_acts,
            self.method_params,
        )


def create_steering_method(method_name: str, args=None) -> SteeringMethodWrapper:
    """
    Create a steering method with CAAMethod-compatible interface.
    
    This is a drop-in replacement factory that returns a method object
    that works exactly like CAAMethod but can be any steering method.
    
    Args:
        method_name: Name of the steering method (CAA, TECZA, TETNO, GROM)
        args: Optional CLI args to extract method-specific parameters
        
    Returns:
        SteeringMethodWrapper with train_for_layer method
        
    Example:
        # Drop-in replacement for CAAMethod
        method = create_steering_method("TECZA", args)
        steering_vector = method.train_for_layer(pos_acts, neg_acts)
    """
    method_name_lower = method_name.lower()
    
    if args is not None:
        method_params = get_method_params_from_args(method_name_lower, args)
    else:
        method_params = {}
    
    return SteeringMethodWrapper(method_name_lower, method_params)


class UnifiedSteeringTrainer:
    """
    Unified trainer that handles all steering methods transparently.
    
    Usage:
        trainer = UnifiedSteeringTrainer(model, method_name="tecza", method_params={...})
        
        # Per-layer training (works for all methods)
        vector = trainer.train_for_layer(layer, pos_acts, neg_acts)
        
        # Full training (for TECZA, TETNO, GROM)
        activations = trainer.train_full(pair_set)
    """
    
    def __init__(
        self,
        model,
        method_name: str,
        method_params: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.method_name = method_name.lower()
        self.method_params = method_params or {}
        self._collector = None
    
    @property
    def collector(self) -> ActivationCollector:
        if self._collector is None:
            self._collector = ActivationCollector(model=self.model, architecture_module_limit=ARCHITECTURE_MODULE_LIMIT)
        return self._collector
    
    def train_for_layer(
        self,
        layer: str,
        pos_acts: List[torch.Tensor],
        neg_acts: List[torch.Tensor],
    ) -> torch.Tensor:
        """Train steering vector for a single layer."""
        return train_steering_vector_for_layer(
            self.method_name,
            pos_acts,
            neg_acts,
            self.method_params,
        )
    
    def train_full(
        self,
        pair_set: ContrastivePairSet,
        layers: Optional[List[str]] = None,
    ) -> LayerActivations:
        """Train steering vectors for all layers using full method."""
        return train_steering_vectors(
            self.method_name,
            pair_set,
            self.method_params,
            layers,
        )
    
    def collect_and_train(
        self,
        pair_set: ContrastivePairSet,
        layers: List[str],
        aggregation: ExtractionStrategy = ExtractionStrategy.CHAT_LAST,
    ) -> LayerActivations:
        """Collect activations and train in one step."""
        # Collect activations
        updated_set = collect_activations_for_pair_set(
            self.model,
            pair_set,
            layers,
            aggregation,
        )
        
        # Train
        return self.train_full(updated_set, layers)
    
    def create_plan(
        self,
        layer_activations: LayerActivations,
        strength: float,
    ) -> SteeringPlan:
        """Create steering plan from trained vectors."""
        return create_steering_plan_from_activations(
            layer_activations,
            strength,
            f"{self.method_name.upper()} steering",
        )
    
    def supports_full_training(self) -> bool:
        """Check if this method supports full ContrastivePairSet training."""
        return self.method_name in ("tecza", "tetno", "grom")
    
    def get_method_description(self) -> str:
        """Get description of the current method."""
        definition = SteeringMethodRegistry.get(self.method_name)
        return definition.description


