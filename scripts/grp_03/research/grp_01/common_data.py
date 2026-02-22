"""
Data structures for research analysis.

Contains dataclasses used across research scripts.
"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class ActivationData:
    """Container for activation data from a single pair."""
    pair_id: int
    set_id: int
    benchmark: str
    layer: int
    strategy: str
    positive_activation: np.ndarray
    negative_activation: np.ndarray


@dataclass
class BenchmarkResults:
    """Results for a single benchmark."""
    name: str
    num_pairs: int
    strategies: Dict[str, Dict[str, float]] = field(default_factory=dict)
    geometry_metrics: Dict[str, float] = field(default_factory=dict)
    best_strategy: str = ""
    best_accuracy: float = 0.0
