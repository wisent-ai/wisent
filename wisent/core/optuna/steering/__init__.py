"""
Steering optimization pipeline for hyperparameter search using Optuna.

This module provides tools for optimizing steering vector hyperparameters:
- OptimizationConfig: Configuration dataclass for the pipeline
- OptimizationPipeline: Main pipeline class for running optimization
- ActivationCache: Caching system for activations
- WandBTracker: WandB integration for experiment tracking
"""

from .config import OptimizationConfig
from .pipeline import OptimizationPipeline
from .cache import ActivationCache
from .tracking import WandBTracker
from .generation import GenerationHelper
from .evaluation import EvaluationHelper
from .results import ResultsSaver
from .cli import main as run_cli

__all__ = [
    "OptimizationConfig",
    "OptimizationPipeline",
    "ActivationCache",
    "WandBTracker",
    "GenerationHelper",
    "EvaluationHelper",
    "ResultsSaver",
    "run_cli",
]
