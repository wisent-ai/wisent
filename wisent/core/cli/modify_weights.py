"""
CLI command for modifying model weights using steering vectors.

This module implements the modify-weights command which permanently modifies
model weights using either abliteration or additive methods.

By default, uses Norm-Preserving Biprojected Abliteration (Jim Lai's technique)
which maintains model quality by preserving weight norms.
"""

import json
import sys
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wisent.core.cli_logger import setup_logger, bind
from wisent.core.weight_modification import (
    abliterate_weights,
    abliterate_with_kernel,
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
    4. Modify weights (norm-preserving abliteration or additive)
    5. Export modified model
    """
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
        if args.method == "abliteration":
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
        # Generate steering vectors from task
        if args.verbose:
            print(f"Generating steering vectors from task '{args.task}'...")

        from wisent.core.cli.generate_vector_from_task import execute_generate_vector_from_task

        # Create temp args for vector generation
        class VectorArgs:
            pass

        vector_args = VectorArgs()
        vector_args.task = args.task
        vector_args.trait_label = args.trait_label
        vector_args.model = args.model
        vector_args.num_pairs = args.num_pairs
        vector_args.layers = str(args.layers) if args.layers is not None else "all"
        vector_args.token_aggregation = args.token_aggregation
        vector_args.prompt_strategy = args.prompt_strategy
        vector_args.method = "caa"
        vector_args.normalize = args.normalize_vectors
        vector_args.verbose = args.verbose
        vector_args.timing = args.timing
        vector_args.intermediate_dir = None
        vector_args.keep_intermediate = False
        vector_args.device = None

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
        if args.save_steering_vectors:
            import shutil
            shutil.copy(vector_args.output, args.save_steering_vectors)
            if args.verbose:
                print(f"✓ Saved steering vectors to {args.save_steering_vectors}")

        # Clean up temp file
        import os
        os.unlink(vector_args.output)

        if args.verbose:
            print(f"✓ Generated {len(steering_vectors)} steering vectors\n")

    elif args.trait:
        # Generate steering vectors from synthetic trait
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
        vector_args.similarity_threshold = args.similarity_threshold
        vector_args.layers = str(args.layers) if args.layers is not None else "all"
        vector_args.token_aggregation = args.token_aggregation
        vector_args.prompt_strategy = args.prompt_strategy
        vector_args.method = "caa"
        vector_args.normalize = args.normalize_vectors
        vector_args.verbose = args.verbose
        vector_args.timing = args.timing
        vector_args.intermediate_dir = None
        vector_args.keep_intermediate = False
        vector_args.device = None

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
        if args.save_steering_vectors:
            import shutil
            shutil.copy(vector_args.output, args.save_steering_vectors)
            if args.verbose:
                print(f"✓ Saved steering vectors to {args.save_steering_vectors}")

        # Clean up temp file
        import os
        os.unlink(vector_args.output)

        if args.verbose:
            print(f"✓ Generated {len(steering_vectors)} steering vectors\n")

    # Step 1.5: Load harmless vectors for biprojection (if provided)
    harmless_vectors = None
    if args.method == "abliteration" and use_biprojection and hasattr(args, 'harmless_vectors') and args.harmless_vectors:
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

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.verbose:
        print(f"✓ Model loaded\n")

    # Step 3: Modify weights
    if args.verbose:
        print(f"Modifying weights using {args.method} method...")
        print()

    if args.method == "abliteration":
        # Abliteration method (norm-preserving by default)
        if args.use_kernel:
            # Use kernel-based layer weighting
            stats = abliterate_with_kernel(
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
            # Uniform abliteration
            stats = abliterate_weights(
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
        if args.method == "abliteration":
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
        if args.method == "abliteration":
            print(f"Norm-preserving: {norm_preserve}")
            print(f"Biprojection: {use_biprojection and harmless_vectors is not None}")
        print(f"Layers modified: {stats['layers_modified']}")
        print(f"Parameters modified: {stats['total_parameters_modified']:,}")
        print("=" * 80 + "\n")

    log.info("Weight modification complete", extra={
        "method": args.method,
        "output_dir": args.output_dir,
        "norm_preserve": norm_preserve if args.method == "abliteration" else None,
        "stats": stats,
    })
