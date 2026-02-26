"""Steering vector generation for optimize-weights."""
import json
import os
import tempfile

import torch
from wisent.core.constants import DEFAULT_RANDOM_SEED, DEFAULT_SPLIT_RATIO, SIMILARITY_THRESHOLD
from wisent.core.utils import resolve_default_device
from wisent.core.cli.optimization.specific.optimize_weights_training import _train_multi_direction_method


def _generate_steering_vectors(args, num_pairs: int, num_layers: int = None) -> tuple[dict[int, torch.Tensor], list[str], list[str]]:
    """Generate steering vectors from trait or task.

    Args:
        args: Command line arguments
        num_pairs: Number of contrastive pairs to generate
        num_layers: Number of model layers (optional, not used - kept for backwards compatibility)

    Returns:
        Tuple of (steering_vectors, positive_examples, negative_examples)
    """
    from argparse import Namespace

    if args.steering_vectors:
        with open(args.steering_vectors, "r") as f:
            data = json.load(f)
        vectors = {
            int(layer) - 1: torch.tensor(vec)
            for layer, vec in data["steering_vectors"].items()
        }
        # Try to load pairs from the same file if available
        positive_examples = data.get("positive_examples", [])
        negative_examples = data.get("negative_examples", [])
        return vectors, positive_examples, negative_examples

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_output = f.name

    # Also create a temp file for pairs
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_pairs = f.name

    try:
        # Determine task type
        task_lower = (args.task or "").lower()
        
        if task_lower == "personalization":
            # Personalization: requires --trait
            if not getattr(args, 'trait', None):
                raise ValueError("--trait is required when --task personalization")
            
            from wisent.core.cli.generate_vector_from_synthetic import execute_generate_vector_from_synthetic

            vector_args = Namespace(
                trait=args.trait,
                num_pairs=num_pairs,
                output=temp_output,
                model=args.model,
                device=args.device,
                similarity_threshold=getattr(args, 'similarity_threshold', SIMILARITY_THRESHOLD),
                verbose=False,
                timing=False,
                layers=args.layers,
                method="caa",
                normalize=True,
                keep_intermediate=True,
                intermediate_dir=os.path.dirname(temp_pairs),
                pairs_cache_dir=getattr(args, 'pairs_cache_dir', None),
                force_regenerate=False,
                nonsense=False,
                nonsense_mode=None,
                accept_low_quality_vector=True,
            )

            execute_generate_vector_from_synthetic(vector_args)

        elif task_lower == "refusal":
            # Refusal: use synthetic pairs with refusal trait
            from wisent.core.cli.generate_vector_from_synthetic import execute_generate_vector_from_synthetic

            vector_args = Namespace(
                trait="refusal",
                num_pairs=num_pairs,
                output=temp_output,
                model=args.model,
                device=args.device,
                similarity_threshold=getattr(args, 'similarity_threshold', SIMILARITY_THRESHOLD),
                verbose=False,
                timing=False,
                layers=args.layers,
                method="caa",
                normalize=True,
                keep_intermediate=True,
                intermediate_dir=os.path.dirname(temp_pairs),
                pairs_cache_dir=getattr(args, 'pairs_cache_dir', None),
                force_regenerate=False,
                nonsense=False,
                nonsense_mode=None,
                accept_low_quality_vector=True,
            )

            execute_generate_vector_from_synthetic(vector_args)

        elif task_lower == "custom":
            # Custom evaluator: requires --trait for vector generation
            if not getattr(args, 'trait', None):
                raise ValueError("--trait is required when --task custom (needed to generate steering vectors)")
            
            from wisent.core.cli.generate_vector_from_synthetic import execute_generate_vector_from_synthetic

            vector_args = Namespace(
                trait=args.trait,
                num_pairs=num_pairs,
                output=temp_output,
                model=args.model,
                device=args.device,
                similarity_threshold=getattr(args, 'similarity_threshold', SIMILARITY_THRESHOLD),
                verbose=False,
                timing=False,
                layers=args.layers,
                method="caa",
                normalize=True,
                keep_intermediate=True,
                intermediate_dir=os.path.dirname(temp_pairs),
                pairs_cache_dir=getattr(args, 'pairs_cache_dir', None),
                force_regenerate=False,
                nonsense=False,
                nonsense_mode=None,
                accept_low_quality_vector=True,
            )

            execute_generate_vector_from_synthetic(vector_args)

        elif "," in (args.task or ""):
            # Multiple benchmarks: generate unified steering vector
            from wisent.core.cli.train_unified_goodness import execute_train_unified_goodness

            # Use .pt format for train_unified_goodness output
            temp_output_pt = temp_output.replace('.json', '.pt')

            # Parse layers - if 'all' or None, use None to let train_unified_goodness pick ALL layers
            layers_arg = args.layers if hasattr(args, 'layers') else None
            if layers_arg == 'all' or layers_arg is None:
                layers_arg = None  # Will use ALL layers (train_unified_goodness default)
            
            vector_args = Namespace(
                task=args.task,  # Pass comma-separated benchmarks
                exclude_benchmarks=None,
                max_benchmarks=getattr(args, 'max_benchmarks', None),
                cap_pairs_per_benchmark=getattr(args, 'cap_pairs_per_benchmark', None),
                train_ratio=getattr(args, 'train_ratio', DEFAULT_SPLIT_RATIO),
                seed=getattr(args, 'seed', DEFAULT_RANDOM_SEED),
                model=args.model,
                device=args.device,
                layer=None,
                layers=layers_arg,
                method='caa',
                normalize=True,
                no_normalize=False,
                skip_evaluation=True,
                evaluate_steering_scales="0.0,1.0",
                save_pairs=None,
                save_report=None,
                output=temp_output_pt,
                verbose=False,
                timing=False,
            )

            execute_train_unified_goodness(vector_args)

            # Load the .pt file
            checkpoint = torch.load(temp_output_pt, map_location=resolve_default_device(), weights_only=False)

            # Handle different checkpoint formats
            if 'all_layer_vectors' in checkpoint:
                raw_vectors = checkpoint['all_layer_vectors']
            elif 'steering_vector' in checkpoint and 'layer_index' in checkpoint:
                raw_vectors = {checkpoint['layer_index']: checkpoint['steering_vector']}
            else:
                raw_vectors = {
                    k: v for k, v in checkpoint.items()
                    if isinstance(k, (int, str)) and str(k).isdigit()
                }

            vectors = {}
            for layer, vec in raw_vectors.items():
                layer_idx = int(layer) if isinstance(layer, str) else layer
                vectors[layer_idx] = vec if isinstance(vec, torch.Tensor) else torch.tensor(vec)

            # Extract eval pairs from checkpoint for pooled evaluation
            eval_pairs = checkpoint.get('eval_pairs', [])
            
            # Store eval pairs in args for the evaluator to use
            args._pooled_eval_pairs = eval_pairs
            args._benchmarks_used = checkpoint.get('benchmarks_used', [])

            # Clean up
            if os.path.exists(temp_output_pt):
                os.unlink(temp_output_pt)

            # Return vectors and empty examples (eval pairs stored in args)
            return vectors, [], []

        elif getattr(args, 'trait', None):
            # Trait-based: use synthetic pairs (when --trait is provided without --task)
            from wisent.core.cli.generate_vector_from_synthetic import execute_generate_vector_from_synthetic

            vector_args = Namespace(
                trait=args.trait,
                num_pairs=num_pairs,
                output=temp_output,
                model=args.model,
                device=args.device,
                similarity_threshold=getattr(args, 'similarity_threshold', SIMILARITY_THRESHOLD),
                verbose=False,
                timing=False,
                layers=args.layers,
                method="caa",
                normalize=True,
                keep_intermediate=True,
                intermediate_dir=os.path.dirname(temp_pairs),
                pairs_cache_dir=getattr(args, 'pairs_cache_dir', None),
                force_regenerate=False,
                nonsense=False,
                nonsense_mode=None,
                accept_low_quality_vector=True,
            )

            execute_generate_vector_from_synthetic(vector_args)

        else:
            # Single benchmark: use task-based generation
            from wisent.core.cli.generate_vector_from_task import execute_generate_vector_from_task

            vector_args = Namespace(
                task=args.task,
                trait_label="correctness",
                num_pairs=num_pairs,
                output=temp_output,
                model=args.model,
                device=args.device,
                verbose=False,
                timing=False,
                layers=args.layers,
                method="caa",
                normalize=True,
                keep_intermediate=True,
                intermediate_dir=os.path.dirname(temp_pairs),
                accept_low_quality_vector=True,
            )

            execute_generate_vector_from_task(vector_args)

        with open(temp_output, "r") as f:
            data = json.load(f)

        vectors = {
            int(layer) - 1: torch.tensor(vec)
            for layer, vec in data["steering_vectors"].items()
        }

        # Try to extract positive/negative examples from the output or intermediate files
        positive_examples = data.get("positive_examples", [])
        negative_examples = data.get("negative_examples", [])

        # If not in output, look for pairs in intermediate directory
        if not positive_examples or not negative_examples:
            intermediate_dir = os.path.dirname(temp_pairs)
            # Search for any *_pairs.json file (the actual filename is {trait}_pairs.json)
            import glob
            pairs_files = glob.glob(os.path.join(intermediate_dir, "*_pairs.json"))
            # Filter out enriched files (which have _with_activations in name)
            pairs_files = [f for f in pairs_files if "_with_activations" not in f]

            for pairs_file in pairs_files:
                with open(pairs_file, "r") as f:
                    pairs_data = json.load(f)
                if "pairs" in pairs_data:
                    for pair in pairs_data["pairs"]:
                        if "positive_response" in pair and "model_response" in pair["positive_response"]:
                            positive_examples.append(pair["positive_response"]["model_response"])
                        if "negative_response" in pair and "model_response" in pair["negative_response"]:
                            negative_examples.append(pair["negative_response"]["model_response"])

        # If using multi-direction method (grom/tecza/tetno), train on pairs with activations
        method = getattr(args, 'method', 'directional')
        if method in ('grom', 'tecza', 'tetno'):
            vectors = _train_multi_direction_method(
                args, vectors, intermediate_dir, method
            )

        return vectors, positive_examples, negative_examples

    finally:
        if os.path.exists(temp_output):
            os.unlink(temp_output)
        if os.path.exists(temp_pairs):
            os.unlink(temp_pairs)

