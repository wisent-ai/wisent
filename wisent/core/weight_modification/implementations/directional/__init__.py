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
from .hooks import (
    GROMRuntimeHooks, TETNORuntimeHooks, NurtRuntimeHooks,
    SzlakRuntimeHooks, WicherRuntimeHooks,
)
from .hooks.grom import project_weights_grom
from .hooks.tetno import apply_grom_steering
from .hooks.transport.szlak import project_weights_szlak
from .hooks.transport.wicher import project_weights_wicher

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
    "GROMRuntimeHooks",
    "TETNORuntimeHooks",
    "NurtRuntimeHooks",
    "SzlakRuntimeHooks",
    "WicherRuntimeHooks",
    "project_weights_grom",
    "apply_grom_steering",
    "project_weights_szlak",
    "project_weights_wicher",
]
