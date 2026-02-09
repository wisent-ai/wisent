"""
Weight modification for permanent steering.

This module provides methods to permanently modify model weights based on
steering vectors computed from contrastive pairs. Unlike temporary activation
steering (hooks), these modifications are baked into the model weights.

Four approaches:

1. **Norm-Preserving Biprojected Directional Modification** (RECOMMENDED):
   - Decomposes weights into magnitude and direction
   - Projects only the direction, preserves original magnitudes
   - Optionally orthogonalizes against harmless directions (biprojection)
   - Maintains model intelligence and reasoning capabilities
   - Can be used to ADD or REMOVE behaviors

2. **Standard Directional Projection** (Legacy, NOT recommended):
   - W' = W - λ(vvᵀ)W
   - Removes dimension parallel to steering vector
   - CHANGES WEIGHT NORMS - can degrade model quality

3. **Additive Weight Modification** (Steering-baked-in):
   - W' = W + αv (in appropriate form)
   - Adds bias toward steering direction in weights
   - More conservative, preserves capabilities better
   - Equivalent to "pre-computing" the steering

4. **Null-Space Constrained Editing** (AlphaEdit-style):
   - P_null = I - V diag(S²/(S²+ε)) Vᵀ via SVD
   - Projects each weight delta into the null space of preserved keys
   - Provably preserves activations on protected inputs
   - Best for multi-concept editing with minimal interference

All approaches allow exporting modified models that no longer need
runtime hooks or steering vectors.

Usage:
    from wisent.core.weight_modification import (
        project_weights,
        project_weights_norm_preserved,
        bake_steering_into_weights,
        export_modified_model
    )

    # Option 1: Norm-preserving directional modification (RECOMMENDED)
    # By default, project_weights uses norm_preserve=True
    project_weights(model, steering_vectors, strength=1.0)

    # Or explicitly:
    project_weights_norm_preserved(
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

from wisent.core.weight_modification.directional import (
    project_weights,
    project_weights_norm_preserved,
    project_weights_multi_direction,
    project_weights_titan,
    project_component,
    project_component_norm_preserved,
    project_component_multi_direction,
    compute_projection_kernel,
    project_with_kernel,
    orthogonalize_direction,
    TITANRuntimeHooks,
    apply_titan_steering,
)
from wisent.core.weight_modification.methods.additive import (
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
from wisent.core.weight_modification.multi.multi_direction import (
    MultiDirectionConfig,
    MultiDirectionResult,
    train_and_bake_titan,
    train_and_bake_prism,
    train_and_bake_pulse,
    train_and_bake,
    bake_multi_directions,
    combine_directions,
)
from wisent.core.weight_modification.methods.guided import (
    GuidedModificationConfig,
    GuidedModificationResult,
    LayerDiagnostics,
    CollateralDamageReport,
    AblationMode,
    run_guided_modification,
    compute_layer_diagnostics,
    compute_fisher_weights,
    select_surgical_layers,
    validate_collateral_damage,
)
from wisent.core.weight_modification.multi.multi_concept import (
    MultiConceptConfig,
    ConceptSpec,
    ConceptAction,
    MultiConceptResult,
    run_multi_concept_modification,
    orthogonalize_concept_directions,
    compute_interference_matrix,
    bidirectional_projection,
)
from wisent.core.weight_modification.directional.null_space import (
    PreservedKeyMatrix,
    compute_null_space_projector,
    project_delta_into_null_space,
    project_component_null_space,
    project_weights_null_space,
    bidirectional_projection_null_space,
)

__all__ = [
    # Norm-Preserving Biprojected Directional Modification (RECOMMENDED)
    "project_weights",  # Default uses norm_preserve=True
    "project_weights_norm_preserved",
    "project_weights_multi_direction",  # PRISM multi-directional
    "project_weights_titan",  # TITAN weight modification
    "project_component_norm_preserved",
    "project_component_multi_direction",  # PRISM multi-directional
    "project_component",
    "compute_projection_kernel",
    "project_with_kernel",
    "orthogonalize_direction",
    # TITAN runtime hooks
    "TITANRuntimeHooks",
    "apply_titan_steering",
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
    # Multi-direction (TITAN/PRISM/PULSE baked into weights)
    "MultiDirectionConfig",
    "MultiDirectionResult",
    "train_and_bake_titan",
    "train_and_bake_prism",
    "train_and_bake_pulse",
    "train_and_bake",
    "bake_multi_directions",
    "combine_directions",
    # Guided modification (linearity-driven)
    "GuidedModificationConfig",
    "GuidedModificationResult",
    "LayerDiagnostics",
    "CollateralDamageReport",
    "AblationMode",
    "run_guided_modification",
    "compute_layer_diagnostics",
    "compute_fisher_weights",
    "select_surgical_layers",
    "validate_collateral_damage",
    # Multi-concept modification
    "MultiConceptConfig",
    "ConceptSpec",
    "ConceptAction",
    "MultiConceptResult",
    "run_multi_concept_modification",
    "orthogonalize_concept_directions",
    "compute_interference_matrix",
    "bidirectional_projection",
    # Null-space constrained modification (AlphaEdit-style)
    "PreservedKeyMatrix",
    "compute_null_space_projector",
    "project_delta_into_null_space",
    "project_component_null_space",
    "project_weights_null_space",
    "bidirectional_projection_null_space",
]
