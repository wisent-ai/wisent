import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if d.startswith(("grp_", "sub_", "mid_")))
    if _root != _base:
        __path__.append(_root)

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
