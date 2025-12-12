from .utils.device import empty_device_cache, preferred_dtype, resolve_default_device, resolve_device, resolve_torch_device
# Note: SteeringMethod and SteeringType import temporarily disabled due to missing dependencies
# from .steering import SteeringMethod, SteeringType

# Universal Subspace Analysis (based on "Universal Weight Subspace Hypothesis")
from .universal_subspace import (
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
]
