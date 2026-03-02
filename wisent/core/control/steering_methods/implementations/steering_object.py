"""
Unified Steering Objects for all steering methods.

Each steering method produces a SteeringObject that contains:
- The steering vectors/directions
- Method-specific components (gates, networks, thresholds)
- Metadata about training
- Methods to apply steering at inference time
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-export base classes
from wisent.core.control.steering_methods._steering_object_base import (
    LayerName,
    SteeringObjectMetadata,
    BaseSteeringObject,
)

# Re-export simple objects
from wisent.core.control.steering_methods._steering_object_simple import (
    SimpleSteeringObject,
    CAASteeringObject,
    OstrzeSteeringObject,
    MLPSteeringObject,
)

# Re-export advanced objects
from wisent.core.control.steering_methods._steering_object_advanced import (
    TECZASteeringObject,
    TETNOSteeringObject,
)

# Re-export GROM objects
from wisent.core.control.steering_methods._steering_object_grom import (
    GROMGateNetwork,
    GROMIntensityNetwork,
    GROMSteeringObject,
)


def create_steering_object(
    method: str,
    metadata: SteeringObjectMetadata,
    **kwargs,
) -> BaseSteeringObject:
    """Factory to create appropriate steering object for a method."""
    if method == 'caa':
        return CAASteeringObject(metadata, **kwargs)
    elif method == 'ostrze':
        return OstrzeSteeringObject(metadata, **kwargs)
    elif method == 'mlp':
        return MLPSteeringObject(metadata, **kwargs)
    elif method == 'tecza':
        return TECZASteeringObject(metadata, **kwargs)
    elif method == 'tetno':
        return TETNOSteeringObject(metadata, **kwargs)
    elif method == 'grom':
        return GROMSteeringObject(metadata, **kwargs)
    elif method == 'nurt':
        from wisent.core.control.steering_methods.methods.nurt import NurtSteeringObject
        return NurtSteeringObject(metadata, **kwargs)
    elif method == 'szlak':
        from wisent.core.control.steering_methods.methods.szlak import SzlakSteeringObject
        return SzlakSteeringObject(metadata, **kwargs)
    elif method == 'wicher':
        from wisent.core.control.steering_methods.methods.wicher import WicherSteeringObject
        return WicherSteeringObject(metadata, **kwargs)
    elif method == 'przelom':
        from wisent.core.control.steering_methods.methods.przelom import PrzelomSteeringObject
        return PrzelomSteeringObject(metadata, **kwargs)
    else:
        raise ValueError(f"Unknown steering method: {method}")


def load_steering_object(path: str) -> BaseSteeringObject:
    """Load a steering object from file."""
    return BaseSteeringObject.load(path)


__all__ = [
    "LayerName",
    "SteeringObjectMetadata",
    "BaseSteeringObject",
    "SimpleSteeringObject",
    "CAASteeringObject",
    "OstrzeSteeringObject",
    "MLPSteeringObject",
    "TECZASteeringObject",
    "TETNOSteeringObject",
    "GROMGateNetwork",
    "GROMIntensityNetwork",
    "GROMSteeringObject",
    "create_steering_object",
    "load_steering_object",
]
