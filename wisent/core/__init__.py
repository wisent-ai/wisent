import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)

from wisent.core.device import empty_device_cache, preferred_dtype, resolve_default_device, resolve_device, resolve_torch_device
# Note: SteeringMethod and SteeringType import temporarily disabled due to missing dependencies
# from .steering import SteeringMethod, SteeringType

# Universal Subspace Analysis (based on "Universal Weight Subspace Hypothesis")
from wisent.core.universal_subspace import (
    analyze_steering_vector_subspace,
    check_vector_quality,
    compress_steering_vectors,
    decompress_steering_vectors,
    compute_optimal_num_directions,
    initialize_from_universal_basis,
    verify_subspace_preservation,
    get_recommended_geometry_thresholds,
    SubspaceAnalysisConfig,
    SubspaceAnalysisResult,
    UniversalBasis,
)

# Geometry analysis with ICD and nonsense baseline
from wisent.core.geometry_runner import (
    compute_icd,
    compute_nonsense_baseline,
    generate_nonsense_activations,
    analyze_with_nonsense_baseline,
    GeometryTestResult,
    GeometrySearchResults,
    GeometryRunner,
)

__all__ = [
    # "SteeringMethod",
    # "SteeringType",
    "empty_device_cache",
    "preferred_dtype",
    "resolve_default_device",
    "resolve_device",
    "resolve_torch_device",
    # Universal Subspace Analysis
    "analyze_steering_vector_subspace",
    "check_vector_quality",
    "compress_steering_vectors",
    "decompress_steering_vectors",
    "compute_optimal_num_directions",
    "initialize_from_universal_basis",
    "verify_subspace_preservation",
    "get_recommended_geometry_thresholds",
    "SubspaceAnalysisConfig",
    "SubspaceAnalysisResult",
    "UniversalBasis",
    # Geometry analysis
    "compute_icd",
    "compute_nonsense_baseline",
    "generate_nonsense_activations",
    "analyze_with_nonsense_baseline",
    "GeometryTestResult",
    "GeometrySearchResults",
    "GeometryRunner",
]
