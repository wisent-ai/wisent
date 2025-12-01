"""
Weight modification for permanent steering.

This module provides methods to permanently modify model weights based on
steering vectors computed from contrastive pairs. Unlike temporary activation
steering (hooks), these modifications are baked into the model weights.

Three approaches:

1. **Norm-Preserving Biprojected Abliteration** (RECOMMENDED):
   - Based on Jim Lai's technique used by Arli AI
   - Decomposes weights into magnitude and direction
   - Ablates only the direction, preserves original magnitudes
   - Optionally orthogonalizes against harmless directions (biprojection)
   - Maintains model intelligence and reasoning capabilities
   - See: https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration

2. **Standard Abliteration** (Legacy, NOT recommended):
   - W' = W - λ(vvᵀ)W
   - Removes dimension parallel to steering vector
   - CHANGES WEIGHT NORMS - can degrade model quality

3. **Additive Weight Modification** (Steering-baked-in):
   - W' = W + αv (in appropriate form)
   - Adds bias toward steering direction in weights
   - More conservative, preserves capabilities better
   - Equivalent to "pre-computing" the steering

All approaches allow exporting modified models that no longer need
runtime hooks or steering vectors.

Usage:
    from wisent.core.weight_modification import (
        abliterate_weights,
        abliterate_weights_norm_preserved,
        bake_steering_into_weights,
        export_modified_model
    )

    # Option 1: Norm-preserving abliteration (RECOMMENDED)
    # By default, abliterate_weights uses norm_preserve=True
    abliterate_weights(model, steering_vectors, strength=1.0)

    # Or explicitly:
    abliterate_weights_norm_preserved(
        model,
        steering_vectors,
        harmless_vectors=harmless_vectors,  # Optional biprojection
        strength=1.0,
    )

    # Option 2: Bake in steering (enhance capability)
    bake_steering_into_weights(model, steering_vectors, alpha=1.0)

    # Export
    export_modified_model(model, "path/to/save")
"""

from wisent.core.weight_modification.abliteration import (
    abliterate_weights,
    abliterate_weights_norm_preserved,
    abliterate_component,
    abliterate_component_norm_preserved,
    compute_abliteration_kernel,
    abliterate_with_kernel,
    orthogonalize_direction,
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
    # Norm-Preserving Biprojected Abliteration (RECOMMENDED)
    "abliterate_weights",  # Default uses norm_preserve=True
    "abliterate_weights_norm_preserved",
    "abliterate_component_norm_preserved",
    "orthogonalize_direction",
    # Legacy abliteration (not recommended)
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
