"""Steering methods package."""

from .methods.caa import CAAMethod
from .methods.advanced import TECZAMethod, TECZAConfig, MultiDirectionResult
from .methods.advanced import TETNOMethod, TETNOConfig, TETNOResult
from .methods.grom import GROMMethod, GROMConfig, GROMResult, GatingNetwork, IntensityNetwork
from .methods.nurt import NurtMethod, NurtConfig, NurtResult
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
TECZA = TECZAMethod
TETNO = TETNOMethod
GROM = GROMMethod
Nurt = NurtMethod
SteeringMethod = CAAMethod  # Default steering method

__all__ = [
    # Method classes
    "CAAMethod",
    "CAA",
    "TECZAMethod",
    "TECZA",
    "TECZAConfig",
    "MultiDirectionResult",
    "TETNOMethod",
    "TETNO",
    "TETNOConfig",
    "TETNOResult",
    "GROMMethod",
    "GROM",
    "GROMConfig",
    "GROMResult",
    "GatingNetwork",
    "IntensityNetwork",
    "NurtMethod",
    "Nurt",
    "NurtConfig",
    "NurtResult",
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
