"""Core utility functions and classes."""

from .base_rotator import RotatorError, BaseRotator
from .device import (
    resolve_default_device,
    resolve_torch_device,
    resolve_device,
    set_default_dtype,
    get_default_dtype,
    preferred_dtype,
    device_optimized_dtype,
    steering_vector_dtype,
)
from .layer_combinations import get_layer_combinations, get_layer_combinations_count

__all__ = [
    'RotatorError',
    'BaseRotator',
    'resolve_default_device',
    'resolve_torch_device',
    'resolve_device',
    'set_default_dtype',
    'get_default_dtype',
    'preferred_dtype',
    'device_optimized_dtype',
    'steering_vector_dtype',
    'get_layer_combinations',
    'get_layer_combinations_count',
]
