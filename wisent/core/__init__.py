import os as _os
_base = _os.path.dirname(__file__)
# Add only grp_* and grp_*/sub_* directories to __path__
# Do NOT add deeper descendants (prevents namespace collisions across platforms)
for _entry in sorted(_os.listdir(_base)):
    _grp = _os.path.join(_base, _entry)
    if _entry.startswith('grp_') and _os.path.isdir(_grp):
        __path__.append(_grp)
        for _sub in sorted(_os.listdir(_grp)):
            _sub_path = _os.path.join(_grp, _sub)
            if _sub.startswith('sub_') and _os.path.isdir(_sub_path):
                __path__.append(_sub_path)

from wisent.core.grp_05.utils import empty_device_cache, preferred_dtype, resolve_default_device, resolve_device, resolve_torch_device
# Note: SteeringMethod and SteeringType import temporarily disabled due to missing dependencies
# from .steering import SteeringMethod, SteeringType

# Universal Subspace Analysis (based on "Universal Weight Subspace Hypothesis")
from wisent.core.grp_04.sub_02.steering.grp_03.universal_subspace import (
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
from wisent.core.grp_03.sub_01.geometry.grp_01.geometry_runner import (
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
