"""
CLI command for modifying model weights using steering vectors.

This module implements the modify-weights command which permanently modifies
model weights using either directional projection or additive methods.

By default, uses Norm-Preserving Biprojected Directional Modification
which maintains model quality by preserving weight norms.
"""

import json
import sys
import time
from pathlib import Path
import torch

from wisent.core.utils.device import resolve_default_device
from wisent.core.cli_logger import setup_logger, bind
from wisent.core.models.wisent_model import WisentModel
from wisent.core.weight_modification import (
    project_weights,
    project_with_kernel,
    bake_steering_into_weights,
    bake_steering_with_kernel,
    export_modified_model,
)

_LOG = setup_logger(__name__)


def execute_modify_weights(args):
    """
    Execute weight modification command.

    Pipeline:
    1. Generate/load steering vectors (from task, trait, or file)
    2. Optionally load harmless vectors for biprojection
    3. Load model
    4. Modify weights (norm-preserving directional projection or additive)
    5. Export modified model
    """
    # Expand task if it's a skill or risk name
    from wisent.core.task_selector import expand_task_if_skill_or_risk
    if getattr(args, 'task', None):
        args.task = expand_task_if_skill_or_risk(args.task)
    
    log = bind(_LOG)
    start_time = time.time()

    # Determine norm_preserve and use_biprojection from args
    norm_preserve = not getattr(args, 'no_norm_preserve', False)
    use_biprojection = not getattr(args, 'no_biprojection', False)

    if args.verbose:
        print("\n" + "=" * 80)
        print("WEIGHT MODIFICATION")
        print("=" * 80)
        print(f"Method: {args.method}")
        if args.method == "directional":
            print(f"Norm-Preserving: {norm_preserve} {'(RECOMMENDED)' if norm_preserve else '(NOT recommended)'}")
            print(f"Biprojection: {use_biprojection}")
        print(f"Model: {args.model}")
        print(f"Output: {args.output_dir}")
        print("=" * 80 + "\n")

    # Step 1: Get steering vectors
    if args.steering_vectors:
        # Load pre-computed steering vectors
        if args.verbose:
            print(f"Loading steering vectors from {args.steering_vectors}...")

        vector_path = Path(args.steering_vectors)

        if vector_path.suffix == '.pt':
            # Load PyTorch format (from train-unified-goodness or similar)
            checkpoint = torch.load(args.steering_vectors, map_location=resolve_default_device(), weights_only=False)

            # Handle different .pt file formats
            if 'steering_vectors' in checkpoint:
                # Format from train-unified-goodness: {layer_idx: tensor}
                raw_vectors = checkpoint['steering_vectors']
                # Check if keys are already 0-indexed or need conversion
                first_key = next(iter(raw_vectors.keys()))
                if isinstance(first_key, str):
                    # String keys like "14" - convert to int, assume 0-indexed
                    steering_vectors = {
                        int(layer): vec if isinstance(vec, torch.Tensor) else torch.tensor(vec)
                        for layer, vec in raw_vectors.items()
                    }
                else:
                    # Integer keys - already 0-indexed
                    steering_vectors = {
                        layer: vec if isinstance(vec, torch.Tensor) else torch.tensor(vec)
                        for layer, vec in raw_vectors.items()
                    }
            elif 'vector' in checkpoint:
                # Single vector format: needs layer info
                layer = checkpoint.get('layer', checkpoint.get('best_layer', 14))
                steering_vectors = {layer: checkpoint['vector']}
            else:
                # Assume direct dict format {layer: vector}
                steering_vectors = {
                    int(k): v if isinstance(v, torch.Tensor) else torch.tensor(v)
                    for k, v in checkpoint.items()
                    if isinstance(k, (int, str)) and str(k).isdigit()
                }

            if args.verbose:
                print(f"✓ Loaded {len(steering_vectors)} steering vectors from .pt file")
                if 'metadata' in checkpoint:
                    meta = checkpoint['metadata']
                    if 'benchmarks_used' in meta:
                        print(f"  Trained on {len(meta['benchmarks_used'])} benchmarks")
                    if 'optimal_scale' in meta:
                        print(f"  Optimal steering scale: {meta['optimal_scale']}")
                print()
        else:
            # Load JSON format
            with open(args.steering_vectors, 'r') as f:
                vector_data = json.load(f)

            # Convert 1-indexed layer numbers from JSON to 0-indexed for internal use
            steering_vectors = {
                int(layer) - 1: torch.tensor(vector)
                for layer, vector in vector_data["steering_vectors"].items()
            }

            if args.verbose:
                print(f"✓ Loaded {len(steering_vectors)} steering vectors\n")

    elif args.task:
        # Parse task type
        task_lower = args.task.lower()
        
        if task_lower == "personalization":
            # Personalization: requires --trait
            if not args.trait:
                raise ValueError("--trait is required when --task personalization")
            
            if args.verbose:
                print(f"Generating steering vectors from trait '{args.trait}'...")

            from wisent.core.cli.generate_vector_from_synthetic import execute_generate_vector_from_synthetic

            # Create temp args for vector generation
            class VectorArgs:
                pass

            vector_args = VectorArgs()
            vector_args.trait = args.trait
            vector_args.model = args.model
            vector_args.num_pairs = args.num_pairs
            vector_args.similarity_threshold = getattr(args, 'similarity_threshold', 0.8)
            vector_args.layers = str(args.layers) if args.layers is not None else "all"
            vector_args.token_aggregation = args.token_aggregation
            vector_args.prompt_strategy = args.prompt_strategy
            vector_args.method = "caa"
            vector_args.normalize = args.normalize_vectors
            vector_args.verbose = args.verbose
            vector_args.timing = getattr(args, 'timing', False)
            vector_args.intermediate_dir = None
            vector_args.keep_intermediate = False
            vector_args.device = None
            vector_args.accept_low_quality_vector = getattr(args, 'accept_low_quality_vector', False)
            vector_args.pairs_cache_dir = getattr(args, 'pairs_cache_dir', None)
            vector_args.force_regenerate = False

            # Use temp file for steering vectors
            import tempfile
            temp_vector_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            vector_args.output = temp_vector_file.name
            temp_vector_file.close()

            # Generate vectors
            execute_generate_vector_from_synthetic(vector_args)

            # Load generated vectors
            with open(vector_args.output, 'r') as f:
                vector_data = json.load(f)

            # Convert 1-indexed layer numbers from JSON to 0-indexed for internal use
            steering_vectors = {
                int(layer) - 1: torch.tensor(vector)
                for layer, vector in vector_data["steering_vectors"].items()
            }

            # Optionally save steering vectors
            if getattr(args, 'save_steering_vectors', None):
                import shutil
                shutil.copy(vector_args.output, args.save_steering_vectors)
                if args.verbose:
                    print(f"✓ Saved steering vectors to {args.save_steering_vectors}")

            # Clean up temp file
            import os
            os.unlink(vector_args.output)

            if args.verbose:
                print(f"✓ Generated {len(steering_vectors)} steering vectors\n")

        elif task_lower == "refusal":
            # Refusal: use synthetic pairs with refusal trait
            if args.verbose:
                print("Generating steering vectors for refusal/compliance...")

            from wisent.core.cli.generate_vector_from_synthetic import execute_generate_vector_from_synthetic

            class VectorArgs:
                pass

            vector_args = VectorArgs()
            vector_args.trait = "refusal"
            vector_args.model = args.model
            vector_args.num_pairs = args.num_pairs
            vector_args.similarity_threshold = getattr(args, 'similarity_threshold', 0.8)
            vector_args.layers = str(args.layers) if args.layers is not None else "all"
            vector_args.token_aggregation = args.token_aggregation
            vector_args.prompt_strategy = args.prompt_strategy
            vector_args.method = "caa"
            vector_args.normalize = args.normalize_vectors
            vector_args.verbose = args.verbose
            vector_args.timing = getattr(args, 'timing', False)
            vector_args.intermediate_dir = None
            vector_args.keep_intermediate = False
            vector_args.device = None
            vector_args.accept_low_quality_vector = getattr(args, 'accept_low_quality_vector', False)
            vector_args.pairs_cache_dir = getattr(args, 'pairs_cache_dir', None)
            vector_args.force_regenerate = False

            import tempfile
            temp_vector_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            vector_args.output = temp_vector_file.name
            temp_vector_file.close()

            execute_generate_vector_from_synthetic(vector_args)

            with open(vector_args.output, 'r') as f:
                vector_data = json.load(f)

            steering_vectors = {
                int(layer) - 1: torch.tensor(vector)
                for layer, vector in vector_data["steering_vectors"].items()
            }

            if getattr(args, 'save_steering_vectors', None):
                import shutil
                shutil.copy(vector_args.output, args.save_steering_vectors)
                if args.verbose:
                    print(f"✓ Saved steering vectors to {args.save_steering_vectors}")

            import os
            os.unlink(vector_args.output)

            if args.verbose:
                print(f"✓ Generated {len(steering_vectors)} steering vectors\n")

        elif task_lower == "custom":
            # Custom evaluator: requires --trait for vector generation
            if not args.trait:
                raise ValueError("--trait is required when --task custom (needed to generate steering vectors)")
            
            if args.verbose:
                print(f"Generating steering vectors from trait '{args.trait}' for custom evaluation...")

            from wisent.core.cli.generate_vector_from_synthetic import execute_generate_vector_from_synthetic

            class VectorArgs:
                pass

            vector_args = VectorArgs()
            vector_args.trait = args.trait
            vector_args.model = args.model
            vector_args.num_pairs = args.num_pairs
            vector_args.similarity_threshold = getattr(args, 'similarity_threshold', 0.8)
            vector_args.layers = str(args.layers) if args.layers is not None else "all"
            vector_args.token_aggregation = args.token_aggregation
            vector_args.prompt_strategy = args.prompt_strategy
            vector_args.method = "caa"
            vector_args.normalize = args.normalize_vectors
            vector_args.verbose = args.verbose
            vector_args.timing = getattr(args, 'timing', False)
            vector_args.intermediate_dir = None
            vector_args.keep_intermediate = False
            vector_args.device = None
            vector_args.accept_low_quality_vector = getattr(args, 'accept_low_quality_vector', False)
            vector_args.pairs_cache_dir = getattr(args, 'pairs_cache_dir', None)
            vector_args.force_regenerate = False

            import tempfile
            temp_vector_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            vector_args.output = temp_vector_file.name
            temp_vector_file.close()

            execute_generate_vector_from_synthetic(vector_args)

            with open(vector_args.output, 'r') as f:
                vector_data = json.load(f)

            steering_vectors = {
                int(layer) - 1: torch.tensor(vector)
                for layer, vector in vector_data["steering_vectors"].items()
            }

            if getattr(args, 'save_steering_vectors', None):
                import shutil
                shutil.copy(vector_args.output, args.save_steering_vectors)
                if args.verbose:
                    print(f"✓ Saved steering vectors to {args.save_steering_vectors}")

            import os
            os.unlink(vector_args.output)

            if args.verbose:
                print(f"✓ Generated {len(steering_vectors)} steering vectors\n")

        elif "," in args.task:
            # Multiple benchmarks: use unified goodness training
            benchmarks = [b.strip() for b in args.task.split(",")]
            if args.verbose:
                print(f"Generating steering vectors from {len(benchmarks)} benchmarks: {', '.join(benchmarks)}...")

            from wisent.core.cli.train_unified_goodness import execute_train_unified_goodness

            class UnifiedArgs:
                pass

            unified_args = UnifiedArgs()
            unified_args.task = args.task  # Pass comma-separated list
            unified_args.exclude_benchmarks = None
            unified_args.max_benchmarks = getattr(args, 'max_benchmarks', None)
            unified_args.cap_pairs_per_benchmark = getattr(args, 'cap_pairs_per_benchmark', None)
            unified_args.train_ratio = getattr(args, 'train_ratio', 0.8)
            unified_args.seed = getattr(args, 'seed', 42)
            unified_args.model = args.model
            unified_args.device = getattr(args, 'device', None)
            unified_args.layer = None
            unified_args.layers = args.layers
            unified_args.token_aggregation = args.token_aggregation if hasattr(args, 'token_aggregation') else 'continuation'
            unified_args.prompt_strategy = args.prompt_strategy if hasattr(args, 'prompt_strategy') else 'chat_template'
            unified_args.method = "caa"
            unified_args.normalize = args.normalize_vectors if hasattr(args, 'normalize_vectors') else False
            unified_args.no_normalize = not unified_args.normalize
            unified_args.skip_evaluation = True
            unified_args.evaluate_steering_scales = "0.0,1.0"
            unified_args.save_pairs = None
            unified_args.save_report = None
            unified_args.verbose = args.verbose
            unified_args.timing = args.timing if hasattr(args, 'timing') else False

            import tempfile
            import os
            temp_vector_file = tempfile.NamedTemporaryFile(mode='w', suffix='.pt', delete=False)
            unified_args.output = temp_vector_file.name
            temp_vector_file.close()

            execute_train_unified_goodness(unified_args)

            checkpoint = torch.load(unified_args.output, map_location=resolve_default_device(), weights_only=False)
            
            if 'steering_vectors' in checkpoint:
                raw_vectors = checkpoint['steering_vectors']
            elif 'all_layer_vectors' in checkpoint:
                raw_vectors = checkpoint['all_layer_vectors']
            elif 'steering_vector' in checkpoint and 'layer_index' in checkpoint:
                raw_vectors = {checkpoint['layer_index']: checkpoint['steering_vector']}
            else:
                raw_vectors = {
                    k: v for k, v in checkpoint.items()
                    if isinstance(k, (int, str)) and str(k).isdigit()
                }
            
            steering_vectors = {}
            for layer, vec in raw_vectors.items():
                layer_idx = int(layer) if isinstance(layer, str) else layer
                steering_vectors[layer_idx] = vec if isinstance(vec, torch.Tensor) else torch.tensor(vec)

            if hasattr(args, 'save_steering_vectors') and args.save_steering_vectors:
                import shutil
                shutil.copy(unified_args.output, args.save_steering_vectors)
                if args.verbose:
                    print(f"✓ Saved steering vectors to {args.save_steering_vectors}")

            os.unlink(unified_args.output)

            if args.verbose:
                print(f"✓ Generated {len(steering_vectors)} steering vectors from {len(benchmarks)} benchmarks\n")

        else:
            # Single benchmark: use task-based generation
            # Check for optimal config first
            optimal_config = None
            use_optimal = getattr(args, 'use_optimal', True)
            
            if use_optimal:
                try:
                    from wisent.core.config_manager import get_cached_optimization
                    optimal_result = get_cached_optimization(args.model, args.task, method="*")
                    if optimal_result:
                        optimal_config = {
                            "method": optimal_result.method,
                            "layer": optimal_result.layer,
                            "strength": optimal_result.strength,
                            "strategy": optimal_result.strategy,
                            "token_aggregation": optimal_result.token_aggregation,
                            "prompt_strategy": optimal_result.prompt_strategy,
                            "score": optimal_result.score,
                            # Method-specific params
                            "num_directions": optimal_result.num_directions,
                            "direction_weighting": optimal_result.direction_weighting,
                            "retain_weight": optimal_result.retain_weight,
                            "sensor_layer": optimal_result.sensor_layer,
                            "condition_threshold": optimal_result.condition_threshold,
                            "gate_temperature": optimal_result.gate_temperature,
                            "gate_hidden_dim": optimal_result.gate_hidden_dim,
                            "intensity_hidden_dim": optimal_result.intensity_hidden_dim,
                        }
                        if args.verbose:
                            print(f"\n📊 Found optimal steering config for {args.task}!")
                            print(f"   Method: {optimal_config['method']}")
                            print(f"   Layer: {optimal_config['layer']}")
                            print(f"   Strength: {optimal_config['strength']}")
                            print(f"   Score: {optimal_config['score']:.3f}")
                            print(f"   → Using optimal settings\n")
                except Exception:
                    pass
            
            if args.verbose:
                print(f"Generating steering vectors from task '{args.task}'...")

            from wisent.core.cli.generate_vector_from_task import execute_generate_vector_from_task

            class VectorArgs:
                pass

            vector_args = VectorArgs()
            vector_args.task = args.task
            vector_args.trait_label = getattr(args, 'trait_label', 'correctness')
            vector_args.model = args.model
            vector_args.num_pairs = args.num_pairs
            
            # Use optimal config if available, otherwise use provided args
            if optimal_config and args.layers is None:
                vector_args.layers = str(optimal_config['layer'])
            else:
                vector_args.layers = str(args.layers) if args.layers is not None else "all"
            
            # Map token aggregation from stored format
            if optimal_config:
                token_agg_map = {
                    "last_token": "final",
                    "mean_pooling": "average",
                    "first_token": "first",
                    "max_pooling": "max",
                }
                vector_args.token_aggregation = token_agg_map.get(
                    optimal_config['token_aggregation'], 
                    args.token_aggregation
                )
                vector_args.method = optimal_config['method'].lower()
            else:
                vector_args.token_aggregation = args.token_aggregation
                vector_args.method = "caa"
            
            vector_args.prompt_strategy = args.prompt_strategy
            vector_args.normalize = args.normalize_vectors
            vector_args.verbose = args.verbose
            vector_args.timing = getattr(args, 'timing', False)
            vector_args.intermediate_dir = None
            vector_args.keep_intermediate = False
            vector_args.device = None
            vector_args.accept_low_quality_vector = getattr(args, 'accept_low_quality_vector', False)
            vector_args.use_optimal = use_optimal
            
            # Pass optimal config for method-specific params
            if optimal_config:
                vector_args._optimal_config = optimal_config

            # Use temp file for steering vectors
            import tempfile
            temp_vector_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            vector_args.output = temp_vector_file.name
            temp_vector_file.close()

            # Generate vectors
            execute_generate_vector_from_task(vector_args)

            # Load generated vectors
            with open(vector_args.output, 'r') as f:
                vector_data = json.load(f)

            # Convert 1-indexed layer numbers from JSON to 0-indexed for internal use
            steering_vectors = {
                int(layer) - 1: torch.tensor(vector)
                for layer, vector in vector_data["steering_vectors"].items()
            }

            # Optionally save steering vectors
            if getattr(args, 'save_steering_vectors', None):
                import shutil
                shutil.copy(vector_args.output, args.save_steering_vectors)
                if args.verbose:
                    print(f"✓ Saved steering vectors to {args.save_steering_vectors}")

            # Clean up temp file
            import os
            os.unlink(vector_args.output)

            if args.verbose:
                print(f"✓ Generated {len(steering_vectors)} steering vectors")
                if optimal_config:
                    print(f"   (using optimal {optimal_config['method']} config with score={optimal_config['score']:.3f})\n")
                else:
                    print()

    # Step 1.5: Load harmless vectors for biprojection (if provided)
    harmless_vectors = None
    if args.method == "directional" and use_biprojection and hasattr(args, 'harmless_vectors') and args.harmless_vectors:
        if args.verbose:
            print(f"Loading harmless vectors from {args.harmless_vectors}...")

        with open(args.harmless_vectors, 'r') as f:
            harmless_data = json.load(f)

        harmless_vectors = {
            int(layer) - 1: torch.tensor(vector)
            for layer, vector in harmless_data["steering_vectors"].items()
        }

        if args.verbose:
            print(f"✓ Loaded {len(harmless_vectors)} harmless vectors for biprojection\n")

    # Step 2: Load model
    if args.verbose:
        print(f"Loading model '{args.model}'...")

    wisent_model = WisentModel(args.model, device=getattr(args, 'device', None))
    model = wisent_model.hf_model  # Get underlying HF model for weight modification
    tokenizer = wisent_model.tokenizer

    if args.verbose:
        print(f"✓ Model loaded with {wisent_model.num_layers} layers\n")

    # Step 3: Modify weights
    if args.verbose:
        print(f"Modifying weights using {args.method} method...")
        print()

    if args.method == "directional":
        # Directional projection method (norm-preserving by default)
        if args.use_kernel:
            # Use kernel-based layer weighting
            stats = project_with_kernel(
                model,
                steering_vectors,
                harmless_vectors=harmless_vectors,
                max_weight=args.max_weight,
                max_weight_position=args.max_weight_position,
                min_weight=args.min_weight,
                min_weight_distance=args.min_weight_distance,
                components=args.components,
                normalize_vectors=args.normalize_vectors,
                norm_preserve=norm_preserve,
                use_biprojection=use_biprojection,
                verbose=args.verbose,
            )
        else:
            # Uniform directional projection
            stats = project_weights(
                model,
                steering_vectors,
                harmless_vectors=harmless_vectors,
                components=args.components,
                strength=args.strength,
                normalize_vectors=args.normalize_vectors,
                norm_preserve=norm_preserve,
                use_biprojection=use_biprojection,
                verbose=args.verbose,
            )

    elif args.method == "additive":
        # Additive method
        if args.use_kernel:
            # Use kernel-based layer weighting
            stats = bake_steering_with_kernel(
                model,
                steering_vectors,
                max_alpha=args.max_weight,  # Use max_weight for alpha
                max_alpha_position=args.max_weight_position,
                min_alpha=args.min_weight,
                components=args.components,
                method=args.additive_method,
                verbose=args.verbose,
            )
        else:
            # Uniform additive
            stats = bake_steering_into_weights(
                model,
                steering_vectors,
                components=args.components,
                alpha=args.alpha,
                method=args.additive_method,
                verbose=args.verbose,
            )

    if args.verbose:
        print()
        print("✓ Weight modification complete!")
        print(f"  Layers modified: {stats['layers_modified']}")
        print(f"  Components modified: {stats['components_modified']}")
        print(f"  Parameters modified: {stats['total_parameters_modified']:,}")
        if args.method == "directional":
            print(f"  Norms preserved: {stats.get('norm_preserved', 'N/A')}")
        print()

    # Step 4: Export model
    if args.verbose:
        print(f"Exporting modified model to {args.output_dir}...")

    # Validate push-to-hub requirements
    if args.push_to_hub and not args.repo_id:
        print("✗ Error: --repo-id required when using --push-to-hub")
        sys.exit(1)

    export_modified_model(
        model,
        args.output_dir,
        tokenizer=tokenizer,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id if args.push_to_hub else None,
        commit_message=args.commit_message,
    )

    if args.verbose:
        print(f"✓ Model exported to {args.output_dir}")
        if args.push_to_hub:
            print(f"✓ Model uploaded to HuggingFace Hub: {args.repo_id}")

    # Timing
    if args.timing:
        elapsed = time.time() - start_time
        print(f"\n⏱  Total time: {elapsed:.2f}s")

    if args.verbose:
        print("\n" + "=" * 80)
        print("WEIGHT MODIFICATION COMPLETE")
        print("=" * 80)
        print(f"Modified model: {args.output_dir}")
        print(f"Method: {args.method}")
        if args.method == "directional":
            print(f"Norm-preserving: {norm_preserve}")
            print(f"Biprojection: {use_biprojection and harmless_vectors is not None}")
        print(f"Layers modified: {stats['layers_modified']}")
        print(f"Parameters modified: {stats['total_parameters_modified']:,}")
        print("=" * 80 + "\n")

    log.info("Weight modification complete", extra={
        "method": args.method,
        "output_dir": args.output_dir,
        "norm_preserve": norm_preserve if args.method == "directional" else None,
        "stats": stats,
    })
