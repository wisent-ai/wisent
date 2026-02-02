"""Directional weight modification package."""

from .core import (
    orthogonalize_direction,
    compute_projection_kernel,
    project_with_kernel,
    verify_weight_modification_preservation,
)
from .projection import project_component_norm_preserved, project_component
from .weights import project_weights_norm_preserved, project_weights
from .multi_direction import project_component_multi_direction, project_weights_multi_direction
from .hooks import TITANRuntimeHooks, PULSERuntimeHooks
from .hooks.titan import project_weights_titan
from .hooks.pulse import apply_titan_steering

__all__ = [
    "orthogonalize_direction",
    "compute_projection_kernel",
    "project_with_kernel",
    "verify_weight_modification_preservation",
    "project_component_norm_preserved",
    "project_component",
    "project_weights_norm_preserved",
    "project_weights",
    "project_component_multi_direction",
    "project_weights_multi_direction",
    "TITANRuntimeHooks",
    "PULSERuntimeHooks",
    "project_weights_titan",
    "apply_titan_steering",
]
