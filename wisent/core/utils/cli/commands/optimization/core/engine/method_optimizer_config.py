"""Configuration and result dataclasses for MethodOptimizer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type

from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
from wisent.core.utils.config_tools.constants import DEFAULT_SCORE
from wisent.core.control.steering_methods.core.atoms import BaseSteeringMethod


@dataclass
class OptimizationConfig:
    """Configuration for a single optimization trial."""
    
    method_name: str
    """Name of the steering method (caa, tecza, tetno, grom)."""
    
    # Activation extraction parameters
    layers: List[str]
    """Layer indices to extract activations from."""
    
    token_aggregation: ExtractionStrategy
    """How to aggregate tokens within a sequence."""
    
    prompt_strategy: ExtractionStrategy
    """How to construct prompts for the model."""
    
    # Application parameters
    strength: float
    """Steering strength multiplier."""
    
    strategy: Optional[str] = None
    """Steering application strategy."""
    
    # Method-specific parameters
    method_params: Dict[str, Any] = field(default_factory=dict)
    """Method-specific parameters (num_directions, sensor_layer, etc.)."""
    
    def __hash__(self):
        return hash((
            self.method_name,
            tuple(self.layers),
            self.token_aggregation.value,
            self.prompt_strategy.value,
            self.strength,
            self.strategy,
            tuple(sorted(self.method_params.items())),
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "layers": self.layers,
            "token_aggregation": self.token_aggregation.value,
            "prompt_strategy": self.prompt_strategy.value,
            "strength": self.strength,
            "strategy": self.strategy,
            "method_params": self.method_params,
        }


@dataclass
class OptimizationResult:
    """Result of a single optimization trial."""
    
    config: OptimizationConfig
    """The configuration that was tested."""
    
    score: float
    """Primary evaluation score."""
    
    metrics: Dict[str, float] = field(default_factory=dict)
    """Additional metrics (accuracy, f1, etc.)."""
    
    steering_vectors: Optional[LayerActivations] = None
    """Trained steering vectors (optional, for caching)."""
    
    training_time: float = DEFAULT_SCORE
    """Time taken to train the method."""
    
    evaluation_time: float = DEFAULT_SCORE
    """Time taken to evaluate."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata from training."""


@dataclass
class OptimizationSummary:
    """Summary of optimization run."""
    
    best_result: OptimizationResult
    """Best result found."""
    
    all_results: List[OptimizationResult]
    """All results from the optimization."""
    
    method_name: str
    """Method that was optimized."""
    
    task_name: str
    """Task/benchmark used for evaluation."""
    
    total_time: float = DEFAULT_SCORE
    """Total optimization time."""
    
    configs_tested: int = 0
    """Number of configurations tested."""
    
    baseline_score: float = DEFAULT_SCORE
    """Baseline (unsteered) accuracy for comparison."""
    
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    """Baseline metrics (accuracy, correct, total)."""


