"""
Optuna-based classifier optimization for efficient hyperparameter search.

This module provides a modern, efficient optimization system that pre-generates
activations once and uses intelligent caching to avoid redundant training.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from wisent.core.classifier.classifier import Classifier
from wisent.core.utils import resolve_default_device, preferred_dtype
from wisent.core.errors import NoActivationDataError, ClassifierCreationError

from .activation_generator import ActivationData, ActivationGenerator, GenerationConfig
from .classifier_cache import CacheConfig, ClassifierCache


def get_model_dtype(model) -> torch.dtype:
    """
    Extract model's native dtype from parameters.

    Args:
        model: PyTorch model or wisent Model wrapper

    Returns:
        The model's native dtype
    """
    # Handle wisent Model wrapper
    if hasattr(model, "hf_model"):
        model_params = model.hf_model.parameters()
    else:
        model_params = model.parameters()

    try:
        return next(model_params).dtype
    except StopIteration:
        # Fallback if no parameters found
        return preferred_dtype()


logger = logging.getLogger(__name__)


@dataclass
class ClassifierOptimizationConfig:
    """Configuration for Optuna classifier optimization."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-0.6B"
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    model_dtype: Optional[torch.dtype] = None  # Auto-detect if None

    # Optuna settings
    n_trials: int = 100
    timeout: Optional[float] = None
    n_jobs: int = 1
    sampler_seed: int = 42

    # Model type search space
    model_types: list[str] = None

    # Hyperparameter ranges
    hidden_dim_range: tuple[int, int] = (32, 512)
    threshold_range: tuple[float, float] = (0.3, 0.9)

    # Training settings
    num_epochs_range: tuple[int, int] = (20, 100)
    learning_rate_range: tuple[float, float] = (1e-4, 1e-2)
    batch_size_options: list[int] = None

    # Evaluation settings
    cv_folds: int = 3
    test_size: float = 0.2
    random_state: int = 42

    # Optimization objective
    primary_metric: str = "f1"  # "accuracy", "f1", "auc", "precision", "recall"

    # Pruning settings
    enable_pruning: bool = True
    pruning_patience: int = 10

    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ["logistic", "mlp"]
        if self.batch_size_options is None:
            self.batch_size_options = [16, 32, 64]

        # Auto-detect device if needed
        if self.device == "auto":
            self.device = resolve_default_device()


@dataclass
class OptimizationResult:
    """Result from Optuna optimization."""

    best_params: dict[str, Any]
    best_value: float
    best_classifier: Classifier
    study: optuna.Study
    trial_results: list[dict[str, Any]]
    optimization_time: float
    cache_hits: int
    cache_misses: int

    def get_best_config(self) -> dict[str, Any]:
        """Get the best configuration found."""
        if not self.best_params:
            return {
                "model_type": "unknown",
                "layer": -1,
                "aggregation": "unknown",
                "threshold": 0.0,
                "hyperparameters": {},
            }

        return {
            "model_type": self.best_params["model_type"],
            "layer": self.best_params["layer"],
            "aggregation": self.best_params["aggregation"],
            "threshold": self.best_params["threshold"],
            "hyperparameters": {
                k: v
                for k, v in self.best_params.items()
                if k not in ["model_type", "layer", "aggregation", "threshold"]
            },
        }


