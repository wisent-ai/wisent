"""Steering methods package."""

from .methods.caa import CAAMethod
from .rotator import SteeringMethodRotator

# Aliases for backward compatibility
CAA = CAAMethod
SteeringMethod = CAAMethod  # Default steering method

__all__ = [
    "CAAMethod",
    "CAA",
    "SteeringMethod",
    "SteeringMethodRotator",
]
