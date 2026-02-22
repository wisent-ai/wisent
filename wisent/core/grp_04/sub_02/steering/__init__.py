import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    if _root != _base:
        __path__.append(_root)

"""Steering module for wisent contrastive steering."""

from .steering import SteeringType, SteeringMethod
from .multi_steering import MultiSteeringError, MultiSteering, run_multi_steer
from .universal_subspace import (
    SubspaceAnalysisConfig,
    SubspaceAnalysisResult,
    analyze_steering_vector_subspace,
    check_vector_quality,
    UniversalBasis,
    compute_universal_basis,
    compress_steering_vectors,
    decompress_steering_vectors,
    save_compressed_vectors,
    load_compressed_vectors,
)

__all__ = [
    'SteeringType',
    'SteeringMethod',
    'MultiSteeringError',
    'MultiSteering',
    'run_multi_steer',
    'SubspaceAnalysisConfig',
    'SubspaceAnalysisResult',
    'analyze_steering_vector_subspace',
    'check_vector_quality',
    'UniversalBasis',
    'compute_universal_basis',
    'compress_steering_vectors',
    'decompress_steering_vectors',
    'save_compressed_vectors',
    'load_compressed_vectors',
]
