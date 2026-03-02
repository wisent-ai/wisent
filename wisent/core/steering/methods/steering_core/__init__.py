import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)

"""Steering module for wisent contrastive steering.

Imports are lazy to avoid circular import chains on Python 3.10
when wisent.core.__init__ imports from this package via fully-qualified paths.
"""


def __getattr__(name):
    """Lazy imports to break circular dependency chains."""
    _steering_names = {'SteeringType', 'SteeringMethod'}
    _multi_names = {'MultiSteeringError', 'MultiSteering', 'run_multi_steer'}
    _subspace_names = {
        'SubspaceAnalysisConfig', 'SubspaceAnalysisResult',
        'analyze_steering_vector_subspace', 'check_vector_quality',
        'UniversalBasis', 'compute_universal_basis',
        'compress_steering_vectors', 'decompress_steering_vectors',
        'save_compressed_vectors', 'load_compressed_vectors',
    }
    if name in _steering_names:
        from .steering import SteeringType, SteeringMethod
        return locals()[name]
    if name in _multi_names:
        from .multi_steering import MultiSteeringError, MultiSteering, run_multi_steer
        return locals()[name]
    if name in _subspace_names:
        from .universal_subspace import (
            SubspaceAnalysisConfig, SubspaceAnalysisResult,
            analyze_steering_vector_subspace, check_vector_quality,
            UniversalBasis, compute_universal_basis,
            compress_steering_vectors, decompress_steering_vectors,
            save_compressed_vectors, load_compressed_vectors,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
