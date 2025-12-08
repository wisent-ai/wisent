"""Steering methods package."""

from .methods.caa import CAAMethod
from .methods.prism import PRISMMethod, PRISMConfig, MultiDirectionResult
from .methods.pulse import PULSEMethod, PULSEConfig, PULSEResult
from .methods.titan import TITANMethod, TITANConfig, TITANResult, GatingNetwork, IntensityNetwork
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
PRISM = PRISMMethod
PULSE = PULSEMethod
TITAN = TITANMethod
SteeringMethod = CAAMethod  # Default steering method

__all__ = [
    # Method classes
    "CAAMethod",
    "CAA",
    "PRISMMethod",
    "PRISM",
    "PRISMConfig",
    "MultiDirectionResult",
    "PULSEMethod",
    "PULSE",
    "PULSEConfig",
    "PULSEResult",
    "TITANMethod",
    "TITAN",
    "TITANConfig",
    "TITANResult",
    "GatingNetwork",
    "IntensityNetwork",
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
