"""Multi-Concept Weight Modification - main runner.

Re-exports types and ops for backward compatibility, and provides
run_multi_concept_modification as the main entry point.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from wisent.core.cli.cli_logger import setup_logger, bind

from .multi_concept_types import (
    ConceptAction, ConceptSpec, MultiConceptConfig, MultiConceptResult,
)
from .multi_concept_ops import (
    compute_interference_matrix, orthogonalize_concept_directions,
    bidirectional_projection,
)

if TYPE_CHECKING:
    from torch.nn import Module
    from wisent.core.weight_modification.directional.null_space import PreservedKeyMatrix

__all__ = [
    "ConceptAction", "ConceptSpec", "MultiConceptConfig", "MultiConceptResult",
    "run_multi_concept_modification", "orthogonalize_concept_directions",
    "compute_interference_matrix", "bidirectional_projection",
]

_LOG = setup_logger(__name__)


def run_multi_concept_modification(
    model: "Module",
    concepts: List[ConceptSpec],
    config: Optional[MultiConceptConfig] = None,
    preserved_keys: Optional["PreservedKeyMatrix"] = None,
) -> MultiConceptResult:
    """Run multi-concept weight modification.

    Pipeline:
    1. Compute interference matrix between concepts
    2. Optionally orthogonalize concept directions
    3. Group concepts by action (suppress vs enhance)
    4. Apply bidirectional projection per layer (with optional null-space constraint)
    """
    cfg = config or MultiConceptConfig()
    log = bind(_LOG)
    warnings = []

    if cfg.verbose:
        print("\n" + "=" * 70)
        print("MULTI-CONCEPT WEIGHT MODIFICATION")
        print("=" * 70)
        print(f"Concepts: {[c.name for c in concepts]}")
        print(f"Orthogonalize: {cfg.orthogonalize}, Null-space: {cfg.use_null_space}")
        print(f"Norm preserve: {cfg.norm_preserve}")
        print("=" * 70 + "\n")

    # Step 1: Compute interference
    interference = compute_interference_matrix(concepts)

    if cfg.verbose and interference:
        print("Interference matrix (cosine similarity):")
        for (name_a, name_b), sim in interference.items():
            if name_a < name_b:
                print(f"  {name_a} <-> {name_b}: {sim:.3f}")
        print()

    for (name_a, name_b), sim in interference.items():
        if sim > cfg.max_interference and name_a < name_b:
            warning = f"High interference between {name_a} and {name_b}: {sim:.3f}"
            warnings.append(warning)
            if cfg.warn_on_interference:
                print(f"Warning: {warning}")

    # Step 2: Orthogonalize if requested
    if cfg.orthogonalize and len(concepts) > 1:
        if cfg.verbose:
            print("Orthogonalizing concept directions...")
        concepts = orthogonalize_concept_directions(concepts, cfg)
        interference_after = compute_interference_matrix(concepts)
        if cfg.verbose:
            print("Interference after orthogonalization:")
            for (name_a, name_b), sim in interference_after.items():
                if name_a < name_b:
                    print(f"  {name_a} <-> {name_b}: {sim:.3f}")
            print()

    # Step 3: Group concepts by action
    suppress_concepts = [c for c in concepts if c.action == ConceptAction.SUPPRESS]
    enhance_concepts = [c for c in concepts if c.action == ConceptAction.ENHANCE]

    if cfg.verbose:
        print(f"Suppressing: {[c.name for c in suppress_concepts]}")
        print(f"Enhancing: {[c.name for c in enhance_concepts]}\n")

    # Step 4: Build null-space preserved keys if needed
    if cfg.use_null_space and preserved_keys is None:
        from wisent.core.weight_modification.directional.null_space import PreservedKeyMatrix
        preserved_keys = PreservedKeyMatrix(
            epsilon=cfg.null_space_epsilon, max_rank=cfg.null_space_max_rank,
        )

    # Step 5: Apply modifications per layer
    result = _apply_per_layer(
        model, concepts, suppress_concepts, enhance_concepts,
        cfg, preserved_keys, log,
    )

    layers_modified, total_params, per_concept_stats = result

    if cfg.verbose:
        print(f"\n{'='*70}\nMULTI-CONCEPT MODIFICATION COMPLETE\n{'='*70}")
        print(f"  Concepts: {len(concepts)}, Layers: {layers_modified}")
        print(f"  Parameters: {total_params:,}, Warnings: {len(warnings)}\n{'='*70}\n")

    return MultiConceptResult(
        concepts_modified=[c.name for c in concepts],
        layers_modified=layers_modified,
        total_parameters_modified=total_params,
        interference_matrix=interference,
        orthogonalized=cfg.orthogonalize,
        per_concept_stats=per_concept_stats,
        warnings=warnings,
    )


def _apply_per_layer(model, concepts, suppress_concepts, enhance_concepts, cfg, preserved_keys, log):
    """Apply bidirectional projection per layer (standard or null-space)."""
    all_layers = set()
    for c in concepts:
        all_layers.update(c.steering_vectors.keys())

    if hasattr(model, "model"):
        layers = model.model.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        layers = model.layers

    components = cfg.components or ["self_attn.o_proj", "mlp.down_proj"]
    layers_modified, total_params = 0, 0
    per_concept_stats = {c.name: {"layers": 0, "params": 0} for c in concepts}

    for layer_idx in sorted(all_layers):
        if layer_idx >= len(layers):
            continue

        layer = layers[layer_idx]
        suppress_dirs, suppress_strengths = _gather_directions(suppress_concepts, layer_idx)
        enhance_dirs, enhance_strengths = _gather_directions(enhance_concepts, layer_idx)

        if not suppress_dirs and not enhance_dirs:
            continue

        layer_modified = False
        P_null = preserved_keys.get_projector(layer_idx) if preserved_keys is not None else None

        for component_name in components:
            try:
                component = layer
                for attr in component_name.split("."):
                    component = getattr(component, attr)
                if not hasattr(component, "weight"):
                    continue

                weight_matrix = component.weight

                if cfg.use_null_space and P_null is not None:
                    from wisent.core.weight_modification.directional.null_space import (
                        bidirectional_projection_null_space,
                    )
                    modified = bidirectional_projection_null_space(
                        weight_matrix, suppress_dirs, enhance_dirs,
                        suppress_strengths, enhance_strengths,
                        P_null=P_null, norm_preserve=cfg.norm_preserve,
                    )
                else:
                    modified = bidirectional_projection(
                        weight_matrix, suppress_dirs, enhance_dirs,
                        suppress_strengths, enhance_strengths,
                        norm_preserve=cfg.norm_preserve,
                    )

                weight_matrix.copy_(modified)
                total_params += weight_matrix.numel()
                layer_modified = True

                if cfg.verbose:
                    mode = "null-space" if (cfg.use_null_space and P_null is not None) else "standard"
                    print(f"  Layer {layer_idx:3d} | {component_name:20s} | "
                          f"suppress={len(suppress_dirs)} enhance={len(enhance_dirs)} | {mode}")

            except AttributeError as e:
                log.debug(f"Could not access {component_name} in layer {layer_idx}: {e}")

        if layer_modified:
            layers_modified += 1
            for c in concepts:
                if layer_idx in c.steering_vectors:
                    per_concept_stats[c.name]["layers"] += 1

        # Accumulate keys for subsequent concepts
        if cfg.use_null_space and cfg.accumulate_keys_across_concepts and preserved_keys is not None:
            for c in concepts:
                if layer_idx in c.steering_vectors:
                    preserved_keys.accumulate({layer_idx: c.steering_vectors[layer_idx]})

    return layers_modified, total_params, per_concept_stats


def _gather_directions(concept_list, layer_idx):
    """Gather directions and strengths for a layer from a list of concepts."""
    dirs, strengths = [], []
    for c in concept_list:
        if layer_idx in c.steering_vectors:
            dirs.append(c.steering_vectors[layer_idx])
            weight = c.layer_weights.get(layer_idx, 1.0) if c.layer_weights else 1.0
            strengths.append(c.strength * weight)
    return dirs, strengths
