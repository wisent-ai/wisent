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
from .null_space import (
    PreservedKeyMatrix,
    compute_null_space_projector,
    project_delta_into_null_space,
    project_component_null_space,
    project_weights_null_space,
    bidirectional_projection_null_space,
)

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
    "PreservedKeyMatrix",
    "compute_null_space_projector",
    "project_delta_into_null_space",
    "project_component_null_space",
    "project_weights_null_space",
    "bidirectional_projection_null_space",
]
