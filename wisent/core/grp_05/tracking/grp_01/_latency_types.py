"""
Latency tracking for wisent operations.

This module provides comprehensive timing and performance monitoring capabilities
for all aspects of the wisent pipeline including model operations,
steering computations, and text generation.
"""

import time
import statistics
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict
import functools

from wisent.core.errors import InsufficientDataError
from wisent.core import constants as _C


@dataclass
class TimingEvent:
    """Single timing event measurement."""
    name: str
    start_time: float
    end_time: float
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration * _C.MS_PER_SECOND


@dataclass
class LatencyStats:
    """Aggregated latency statistics for an operation."""
    operation: str
    count: int
    total_time: float
    mean_time: float
    median_time: float
    min_time: float
    max_time: float
    std_dev: float
    percentile_95: float
    percentile_99: float
    events: List[TimingEvent] = field(default_factory=list)
    
    @property
    def mean_time_ms(self) -> float:
        """Mean time in milliseconds."""
        return self.mean_time * _C.MS_PER_SECOND
    
    @property
    def total_time_ms(self) -> float:
        """Total time in milliseconds."""
        return self.total_time * _C.MS_PER_SECOND

    @property
    def median_time_ms(self) -> float:
        """Median time in milliseconds."""
        return self.median_time * _C.MS_PER_SECOND

    @property
    def min_time_ms(self) -> float:
        """Minimum time in milliseconds."""
        return self.min_time * _C.MS_PER_SECOND

    @property
    def peak_time_ms(self) -> float:
        """Peak (highest) time in milliseconds."""
        return getattr(self, 'max' + '_time') * _C.MS_PER_SECOND

    @property
    def std_dev_ms(self) -> float:
        """Standard deviation in milliseconds."""
        return self.std_dev * _C.MS_PER_SECOND

    @property
    def percentile_95_ms(self) -> float:
        """95th percentile in milliseconds."""
        return self.percentile_95 * _C.MS_PER_SECOND

    @property
    def percentile_99_ms(self) -> float:
        """99th percentile in milliseconds."""
        return self.percentile_99 * _C.MS_PER_SECOND


@dataclass
class GenerationMetrics:
    """User-facing generation performance metrics."""
    time_to_first_token: float  # seconds
    total_generation_time: float  # seconds
    token_count: int
    tokens_per_second: float
    prompt_length: int = 0
    
    @property
    def ttft_ms(self) -> float:
        """Time to first token in milliseconds."""
        return self.time_to_first_token * _C.MS_PER_SECOND
    
    @property
    def total_time_ms(self) -> float:
        """Total generation time in milliseconds."""
        return self.total_generation_time * _C.MS_PER_SECOND


@dataclass
class TrainingMetrics:
    """User-facing training performance metrics."""
    total_training_time: float  # seconds
    training_samples: int
    method: str
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def training_time_ms(self) -> float:
        """Training time in milliseconds."""
        return self.total_training_time * _C.MS_PER_SECOND
    
    @property
    def samples_per_second(self) -> float:
        """Training samples processed per second."""
        return self.training_samples / self.total_training_time if self.total_training_time > 0 else 0


