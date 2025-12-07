"""Steering methods package."""

from .methods.caa import CAAMethod
from .rotator import SteeringMethodRotator
from .registry import (
    SteeringMethodRegistry,
    SteeringMethodDefinition,
    SteeringMethodParameter,
    SteeringMethodType,
    get_steering_method,
    list_steering_methods,
    is_valid_steering_method,
)

# Aliases for backward compatibility
CAA = CAAMethod
SteeringMethod = CAAMethod  # Default steering method

__all__ = [
    # Method classes
    "CAAMethod",
    "CAA",
    "SteeringMethod",
    "SteeringMethodRotator",
    # Registry
    "SteeringMethodRegistry",
    "SteeringMethodDefinition",
    "SteeringMethodParameter",
    "SteeringMethodType",
    # Convenience functions
    "get_steering_method",
    "list_steering_methods",
    "is_valid_steering_method",
]
