import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    if _root != _base:
        __path__.append(_root)

"""Weight modification method implementations."""

from .additive import (
    bake_steering_into_weights,
    bake_steering_into_component,
    bake_steering_with_kernel,
)
from .guided import (
    AblationMode,
    GuidedModificationConfig,
    LayerDiagnostics,
    GuidedModificationResult,
    CollateralDamageReport,
    compute_layer_diagnostics,
    compute_fisher_weights,
    select_surgical_layers,
    run_guided_modification,
)

__all__ = [
    'bake_steering_into_weights',
    'bake_steering_into_component',
    'bake_steering_with_kernel',
    'AblationMode',
    'GuidedModificationConfig',
    'LayerDiagnostics',
    'GuidedModificationResult',
    'CollateralDamageReport',
    'compute_layer_diagnostics',
    'compute_fisher_weights',
    'select_surgical_layers',
    'run_guided_modification',
]
