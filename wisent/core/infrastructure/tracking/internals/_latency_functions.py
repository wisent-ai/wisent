"""Convenience functions and operations constants."""
import time
import logging
from typing import Any, Dict, Optional
from contextlib import contextmanager
from functools import wraps
from wisent.core.tracking._latency_types import LatencyStats

logger = logging.getLogger(__name__)

def get_global_tracker() -> LatencyTracker:
    """Get or create the global latency tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = LatencyTracker()
    return _global_tracker


def time_function(operation_name: Optional[str] = None):
    """
    Decorator to automatically time function execution.
    
    Args:
        operation_name: Name for the operation (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        name = operation_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracker = get_global_tracker()
            with tracker.time_operation(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def time_operation(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Global context manager for timing operations."""
    tracker = get_global_tracker()
    with tracker.time_operation(name, metadata) as event_ref:
        yield event_ref


def get_timing_summary() -> Dict[str, LatencyStats]:
    """Get timing summary from global tracker."""
    tracker = get_global_tracker()
    return tracker.get_stats()


def format_timing_summary(detailed: bool = False) -> str:
    """Format timing summary as a readable string."""
    tracker = get_global_tracker()
    stats = tracker.get_stats()
    return tracker.format_stats(stats, detailed)


def reset_timing() -> None:
    """Reset global timing data."""
    tracker = get_global_tracker()
    tracker.reset()


# Common operation names for user-facing metrics
class Operations:
    """Standard operation names for user-facing performance metrics."""
    # Core user-facing metrics
    TOTAL_TRAINING_TIME = "total_training_time"
    TIME_TO_FIRST_TOKEN = "time_to_first_token"
    RESPONSE_GENERATION = "response_generation"
    UNSTEERED_GENERATION = "unsteered_generation"
    STEERED_GENERATION = "steered_generation"
    
    # Batch processing
    BATCH_INFERENCE = "batch_inference"
    PER_RESPONSE = "per_response"
    
    # Training phases
    STEERING_VECTOR_TRAINING = "steering_vector_training"
    CLASSIFIER_TRAINING = "classifier_training"
    
    # Legacy (for backward compatibility)
    MODEL_LOADING = "model_loading"
    ACTIVATION_EXTRACTION = "activation_extraction"
