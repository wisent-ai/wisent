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
