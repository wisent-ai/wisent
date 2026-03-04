"""
Optuna-based classifier optimization for efficient hyperparameter search.

This module provides a modern, efficient optimization system that pre-generates
activations once and uses intelligent caching to avoid redundant training.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from wisent.core.reading.classifiers.core.atoms import Classifier
from wisent.core.utils import resolve_default_device, preferred_dtype
from wisent.core.utils.infra_tools.errors import NoActivationDataError, ClassifierCreationError

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
    # Training settings (required)
    learning_rate_range: tuple[float, float]
    # Model configuration
    model_name: Optional[str] = None
    device: Optional[str] = None  # "auto", "cuda", "cpu", "mps"
    model_dtype: Optional[torch.dtype] = None  # Auto-detect if None

    # Optuna settings
    n_trials: int = None
    timeout: Optional[float] = None
    n_jobs: int = 1
    sampler_seed: int | None = None

    # Model type search space
    model_types: list[str] = field(default_factory=lambda: ["logistic", "mlp"])


    # Hyperparameter ranges
    hidden_dim_range: tuple[int, int] = None
    threshold_range: tuple[float, float] = None

    # Training settings
    num_epochs_range: tuple[int, int] = None
    batch_size_options: list[int] = None

    # Evaluation settings
    cv_folds: int = None
    test_size: float = None
    random_state: int | None = None

    # Optimization objective
    primary_metric: Optional[str] = None  # "accuracy", "f1", "auc", "precision", "recall"

    # Pruning settings
    enable_pruning: bool = True
    pruning_patience: int = None
    prune_accuracy_threshold: float = None
    pruner_startup_trials: int = None

    def __post_init__(self):
        if self.device is None:
            self.device = resolve_default_device()
        for field_name in ("n_trials", "hidden_dim_range", "threshold_range", "num_epochs_range", "batch_size_options", "cv_folds", "pruning_patience", "prune_accuracy_threshold", "pruner_startup_trials", "test_size"):
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} is required in ClassifierOptimizationConfig")


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


