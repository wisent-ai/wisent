"""
Weight modification for permanent steering.

This module provides methods to permanently modify model weights based on
steering vectors computed from contrastive pairs. Unlike temporary activation
steering (hooks), these modifications are baked into the model weights.

Two approaches:

1. **Orthogonal Abliteration** (Heretic-style):
   - W' = W - λ(vvᵀ)W
   - Removes dimension parallel to steering vector
   - Cannot express behavior in that direction anymore
   - Most aggressive, permanent removal

2. **Additive Weight Modification** (Steering-baked-in):
   - W' = W + αv (in appropriate form)
   - Adds bias toward steering direction in weights
   - More conservative, preserves capabilities better
   - Equivalent to "pre-computing" the steering

Both approaches allow exporting modified models that no longer need
runtime hooks or steering vectors.

Usage:
    from wisent.core.weight_modification import (
        abliterate_weights,
        bake_steering_into_weights,
        export_modified_model
    )

    # Option 1: Abliterate (remove capability)
    abliterate_weights(model, steering_vectors, strength=1.0)

    # Option 2: Bake in steering (enhance capability)
    bake_steering_into_weights(model, steering_vectors, alpha=1.0)

    # Export
    export_modified_model(model, "path/to/save")
"""

from wisent.core.weight_modification.abliteration import (
    abliterate_weights,
    abliterate_component,
    compute_abliteration_kernel,
    abliterate_with_kernel,
)
from wisent.core.weight_modification.additive import (
    bake_steering_into_weights,
    bake_steering_into_component,
    bake_steering_with_kernel,
)
from wisent.core.weight_modification.export import (
    export_modified_model,
    save_modified_weights,
    compare_models,
)
from wisent.core.weight_modification.utils import (
    get_modifiable_components,
    verify_modification,
    compute_modification_metrics,
)

__all__ = [
    # Abliteration (Heretic-style orthogonalization)
    "abliterate_weights",
    "abliterate_component",
    "compute_abliteration_kernel",
    "abliterate_with_kernel",
    # Additive (bake steering into weights)
    "bake_steering_into_weights",
    "bake_steering_into_component",
    "bake_steering_with_kernel",
    # Export
    "export_modified_model",
    "save_modified_weights",
    "compare_models",
    # Utils
    "get_modifiable_components",
    "verify_modification",
    "compute_modification_metrics",
]
