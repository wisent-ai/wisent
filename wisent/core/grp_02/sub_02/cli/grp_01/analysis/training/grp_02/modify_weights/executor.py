"""Weight modification execution for different methods."""

from __future__ import annotations

import json
import sys
import time
from typing import Dict, Optional, List, Any

import torch

from wisent.core.cli.cli_logger import setup_logger, bind
from wisent.core.constants import GUIDED_MAX_DEGRADATION, REPORT_LINE_WIDTH
from wisent.core.weight_modification import (
    project_weights, project_with_kernel,
    bake_steering_into_weights, bake_steering_with_kernel,
    export_modified_model,
)

_LOG = setup_logger(__name__)


def execute_standard_modification(
    args, model, tokenizer, steering_vectors: Dict[int, torch.Tensor],
    harmless_vectors: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Execute standard directional or additive weight modification."""
    log = bind(_LOG)
    norm_preserve = not getattr(args, 'no_norm_preserve', False)
    use_biprojection = not getattr(args, 'no_biprojection', False)

    if args.verbose:
        print(f"Modifying weights using {args.method} method...")

    if args.method == "directional":
        if args.use_kernel:
            stats = project_with_kernel(
                model, steering_vectors, harmless_vectors=harmless_vectors,
                max_weight=args.max_weight, max_weight_position=args.max_weight_position,
                min_weight=args.min_weight, min_weight_distance=args.min_weight_distance,
                components=args.components, normalize_vectors=args.normalize_vectors,
                norm_preserve=norm_preserve, use_biprojection=use_biprojection,
                verbose=args.verbose,
            )
        else:
            stats = project_weights(
                model, steering_vectors, harmless_vectors=harmless_vectors,
                components=args.components, strength=args.strength,
                normalize_vectors=args.normalize_vectors, norm_preserve=norm_preserve,
                use_biprojection=use_biprojection, verbose=args.verbose,
            )
    elif args.method == "additive":
        if args.use_kernel:
            stats = bake_steering_with_kernel(
                model, steering_vectors, max_alpha=args.max_weight,
                max_alpha_position=args.max_weight_position, min_alpha=args.min_weight,
                components=args.components, method=args.additive_method, verbose=args.verbose,
            )
        else:
            stats = bake_steering_into_weights(
                model, steering_vectors, components=args.components,
                alpha=args.alpha, method=args.additive_method, verbose=args.verbose,
            )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    if args.verbose:
        print(f"\nWeight modification complete!")
        print(f"  Layers modified: {stats['layers_modified']}")
        print(f"  Components modified: {stats['components_modified']}")
        print(f"  Parameters modified: {stats['total_parameters_modified']:,}")

    _export_model(args, model, tokenizer)
    log.info("Weight modification complete", extra={"method": args.method, "stats": stats})
    return stats


def execute_grom_mode(args, model, tokenizer, wisent_model, pairs):
    """Execute GROM weight modification mode."""
    from .method_training import train_grom_for_task
    from wisent.core.weight_modification.export import export_grom_model

    if args.verbose:
        print("Training GROM for full dynamic steering...")

    grom_result = train_grom_for_task(args, wisent_model, pairs)
    grom_mode = getattr(args, 'grom_mode', 'hybrid')

    if args.verbose:
        print(f"\nExporting GROM model (mode={grom_mode})...")

    export_grom_model(
        model=model, grom_result=grom_result, save_path=args.output_dir,
        tokenizer=tokenizer, mode=grom_mode, push_to_hub=args.push_to_hub,
        repo_id=args.repo_id if args.push_to_hub else None, commit_message=args.commit_message,
    )

    if args.verbose:
        print(f"\nGROM model exported to {args.output_dir}")
        print(f"  Mode: {grom_mode}")
        print(f"  Layers: {len(grom_result.layer_order)}")


def execute_tetno_mode(args, model, tokenizer, wisent_model, pairs):
    """Execute TETNO weight modification mode."""
    from .method_training import train_tetno_for_task
    from wisent.core.weight_modification.export import export_tetno_model

    if args.verbose:
        print("Training TETNO for conditional steering...")

    tetno_result = train_tetno_for_task(args, wisent_model, pairs)
    tetno_mode = getattr(args, 'grom_mode', 'hybrid')

    if args.verbose:
        print(f"\nExporting TETNO model (mode={tetno_mode})...")

    export_tetno_model(
        model=model, tetno_result=tetno_result, save_path=args.output_dir,
        tokenizer=tokenizer, mode=tetno_mode, strength=args.strength,
        push_to_hub=args.push_to_hub, repo_id=args.repo_id if args.push_to_hub else None,
        commit_message=args.commit_message,
    )

    if args.verbose:
        print(f"\nTETNO model exported to {args.output_dir}")
        print(f"  Layers: {len(tetno_result.behavior_vectors)}")
        print(f"  Threshold: {tetno_result.optimal_threshold:.3f}")


def execute_tecza_mode(args, model, tokenizer, wisent_model, pairs):
    """Execute TECZA weight modification mode."""
    from .method_training import train_tecza_for_task
    from wisent.core.weight_modification.export import export_tecza_model

    if args.verbose:
        print("Training TECZA for multi-directional steering...")

    tecza_result = train_tecza_for_task(args, wisent_model, pairs)
    tecza_mode = getattr(args, 'tecza_mode', 'weighted')

    if args.verbose:
        print(f"\nExporting TECZA model (mode={tecza_mode})...")

    export_tecza_model(
        model=model, tecza_result=tecza_result, save_path=args.output_dir,
        tokenizer=tokenizer, mode=tecza_mode, strength=args.strength,
        push_to_hub=args.push_to_hub, repo_id=args.repo_id if args.push_to_hub else None,
        commit_message=args.commit_message,
    )

    if args.verbose:
        num_dirs = next(iter(tecza_result.directions.values())).shape[0]
        print(f"\nTECZA model exported to {args.output_dir}")
        print(f"  Layers: {len(tecza_result.directions)}")
        print(f"  Directions per layer: {num_dirs}")



def execute_guided_modification(args, wisent_model, model, tokenizer):
    """Execute linearity-guided weight modification."""
    from wisent.core.weight_modification.methods.guided import (
        GuidedModificationConfig, AblationMode, run_guided_modification,
    )
    log = bind(_LOG)
    start_time = time.time()

    if args.verbose:
        print("\n" + "=" * REPORT_LINE_WIDTH)
        print("GUIDED WEIGHT MODIFICATION (Linearity-Driven)")
        print("=" * REPORT_LINE_WIDTH)

    pairs = generate_pairs_for_guided_mode(args)
    if not pairs:
        print("Error: No contrastive pairs generated for guided mode")
        sys.exit(1)

    mode_map = {"full": AblationMode.FULL, "surgical": AblationMode.SURGICAL, "adaptive": AblationMode.ADAPTIVE}

    config = GuidedModificationConfig(
        mode=mode_map.get(args.guided_mode, AblationMode.ADAPTIVE),
        surgical_top_k=args.surgical_top_k,
        min_linear_score=args.min_linear_score,
        use_fisher_weights=not getattr(args, 'no_fisher_weights', False),
        extraction_strategy=args.extraction_strategy,
        validate_collateral=getattr(args, 'validate_collateral', False),
        max_allowed_degradation=getattr(args, 'max_degradation', GUIDED_MAX_DEGRADATION),
        base_strength=args.strength,
        normalize_vectors=getattr(args, 'normalize_vectors', True),
        verbose=args.verbose,
    )

    result = run_guided_modification(
        model=model, pairs=pairs, wisent_model=wisent_model, config=config, components=args.components,
    )

    if getattr(args, 'save_diagnostics', None):
        from ._helpers.executor_helpers import save_diagnostics
        save_diagnostics(args.save_diagnostics, result)

    _export_model(args, model, tokenizer)

    if args.timing:
        print(f"\nTotal time: {time.time() - start_time:.2f}s")

    log.info("Guided modification complete", extra={"mode": result.mode_used.value})
    return {"layers_modified": result.layers_modified, "total_parameters_modified": result.total_parameters_modified}


def execute_multi_concept_modification(args, wisent_model, model, tokenizer, base_steering_vectors):
    """Execute multi-concept weight modification."""
    from wisent.core.weight_modification.multi_concept import (
        MultiConceptConfig, ConceptSpec, ConceptAction, run_multi_concept_modification,
    )
    log = bind(_LOG)

    if args.verbose:
        print("\n" + "=" * REPORT_LINE_WIDTH)
        print("MULTI-CONCEPT WEIGHT MODIFICATION")
        print("=" * REPORT_LINE_WIDTH)

    concepts = []
    action_map = {"suppress": ConceptAction.SUPPRESS, "enhance": ConceptAction.ENHANCE, "neutral": ConceptAction.NEUTRAL}

    for concept_str in args.concepts:
        parts = concept_str.split(":")
        if len(parts) < 2:
            print(f"Error: Invalid concept format '{concept_str}'")
            sys.exit(1)

        name, action_str = parts[0], parts[1].lower()
        strength = float(parts[2]) if len(parts) > 2 else 1.0

        if action_str not in action_map:
            print(f"Error: Unknown action '{action_str}'")
            sys.exit(1)

        vectors = base_steering_vectors if name == args.task or name == "base" else base_steering_vectors
        concepts.append(ConceptSpec(name=name, steering_vectors=vectors, action=action_map[action_str], strength=strength))

    config = MultiConceptConfig(
        orthogonalize=not getattr(args, 'no_orthogonalize', False),
        components=args.components, norm_preserve=not getattr(args, 'no_norm_preserve', False), verbose=args.verbose,
    )

    result = run_multi_concept_modification(model=model, concepts=concepts, config=config)
    _export_model(args, model, tokenizer)
    log.info("Multi-concept modification complete", extra={"concepts": result.concepts_modified})
    return {"layers_modified": result.layers_modified, "total_parameters_modified": result.total_parameters_modified}


def generate_pairs_for_guided_mode(args) -> List:
    """Generate contrastive pairs for guided mode diagnostics."""
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.contrastive_pairs.core.io.response import PositiveResponse, NegativeResponse

    pairs = []
    if args.task:
        try:
            from wisent.core.data_loaders import load_contrastive_pairs
            task_pairs = load_contrastive_pairs(task=args.task, num_pairs=args.num_pairs, model_name=args.model)
            for p in task_pairs:
                if hasattr(p, 'prompt') and hasattr(p, 'positive_response') and hasattr(p, 'negative_response'):
                    pairs.append(p)
        except Exception:
            try:
                from wisent.core.synthetic.generators.pairs_generator import generate_synthetic_pairs
                synthetic_pairs = generate_synthetic_pairs(trait=args.task, num_pairs=args.num_pairs)
                for sp in synthetic_pairs:
                    pairs.append(ContrastivePair(
                        prompt=sp.get('prompt', ''),
                        positive_response=PositiveResponse(model_response=sp.get('positive', '')),
                        negative_response=NegativeResponse(model_response=sp.get('negative', '')),
                    ))
            except Exception:
                pass
    return pairs


def _export_model(args, model, tokenizer):
    """Export modified model to output directory."""
    if args.push_to_hub and not args.repo_id:
        print("Error: --repo-id required when using --push-to-hub")
        sys.exit(1)

    export_modified_model(
        model, args.output_dir, tokenizer=tokenizer, push_to_hub=args.push_to_hub,
        repo_id=args.repo_id if args.push_to_hub else None, commit_message=args.commit_message,
    )


# Re-export szlak/wicher mode executors from helpers
from ._helpers.executor_helpers import (  # noqa: E402
    execute_szlak_mode, execute_wicher_mode,
)
