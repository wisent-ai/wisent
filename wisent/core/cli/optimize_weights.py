"""
Optimize weights command - unified optimization for any trait/task with any evaluator.

This command runs an Optuna-based optimization loop to find optimal weight
modification parameters. Works with:

- Any trait (synthetic pairs): refusal, sycophancy, verbosity, personality
- Any task (lm-eval): hellaswag, arc_easy, gsm8k, etc.
- Any evaluator: refusal detection, task accuracy, semantic similarity, LLM judge

Pipeline per trial:
1. Generate steering vector (from trait or task, or use cached)
2. Load fresh model copy
3. Apply weight modification with trial parameters
4. Evaluate using chosen evaluator
5. Return score to Optuna
6. Repeat until target reached or max trials
7. Apply best parameters and save optimized model
"""

import json
import os
import re
import tempfile
import time
import subprocess
from dataclasses import dataclass
from typing import Any, Callable

import torch
from wisent.core.utils.device import resolve_default_device


def upload_to_s3(local_path: str, s3_bucket: str, s3_key: str) -> bool:
    """Upload a file or directory to S3."""
    try:
        if os.path.isdir(local_path):
            cmd = ["aws", "s3", "sync", local_path, f"s3://{s3_bucket}/{s3_key}", "--quiet"]
        else:
            cmd = ["aws", "s3", "cp", local_path, f"s3://{s3_bucket}/{s3_key}", "--quiet"]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"   Warning: S3 upload failed: {e}")
        return False


def download_from_s3(s3_bucket: str, s3_key: str, local_path: str) -> bool:
    """Download a file or directory from S3."""
    try:
        s3_path = f"s3://{s3_bucket}/{s3_key}"
        # Check if it exists
        check_cmd = ["aws", "s3", "ls", s3_path]
        result = subprocess.run(check_cmd, capture_output=True)
        if result.returncode != 0:
            return False
        # Download
        if s3_key.endswith('/'):
            cmd = ["aws", "s3", "sync", s3_path, local_path, "--quiet"]
        else:
            os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
            cmd = ["aws", "s3", "cp", s3_path, local_path, "--quiet"]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception:
        return False

from wisent.core.errors import UnknownTypeError, InsufficientDataError

from wisent.core.models.wisent_model import WisentModel
from wisent.core.evaluators.steering_evaluators import (
    SteeringEvaluatorFactory,
    EvaluatorConfig,
    RefusalEvaluator,
    TaskEvaluator,
    PersonalizationEvaluator as SharedPersonalizationEvaluator,
)
from wisent.core.opti.core.atoms import HPOConfig
from wisent.core.opti.methods.opti_weights import WeightsOptimizer, WeightsOptimizerConfig
from wisent.core.utils.device import resolve_default_device, preferred_dtype


@dataclass
class OptimizationResult:
    """Final result from optimization."""
    best_params: dict
    best_score: float
    target_achieved: bool
    total_time: float
    total_trials: int


def execute_optimize_weights(args):
    """
    Execute the optimize-weights command.

    Runs Optuna optimization to find optimal weight modification parameters
    for any trait or task.
    """
    # Expand task if it's a skill or risk name
    from wisent.core.task_selector import expand_task_if_skill_or_risk
    if getattr(args, 'task', None):
        args.task = expand_task_if_skill_or_risk(args.task)
    
    start_time = time.time()

    print(f"\n{'='*80}")
    print("WEIGHT MODIFICATION OPTIMIZATION")
    print(f"{'='*80}")
    print(f"   Model: {args.model}")
    if args.trait:
        print(f"   Mode: Trait-based (synthetic pairs)")
        print(f"   Trait: {args.trait[:60]}...")
    elif args.task:
        print(f"   Mode: Task-based (lm-eval)")
        print(f"   Task: {args.task}")
    else:
        print(f"   Mode: Pre-computed vectors")
        print(f"   Vectors: {args.steering_vectors}")
    
    # Determine evaluator type from task for display
    task_lower = (args.task or "").lower()
    if task_lower == "refusal":
        evaluator_display = "refusal"
    elif task_lower == "personalization" or (not task_lower and getattr(args, 'trait', None)):
        evaluator_display = f"personalization ({args.trait})"
    elif task_lower == "custom":
        evaluator_display = f"custom ({args.custom_evaluator})"
    elif "," in (args.task or ""):
        evaluator_display = "pooled (multi-benchmark)"
    elif task_lower:
        evaluator_display = f"task ({args.task})"
    else:
        evaluator_display = "unknown"
    print(f"   Evaluator: {evaluator_display}")
    print(f"   Target: {args.target_metric} = {args.target_value}")
    print(f"   Trials: {args.trials}")
    print(f"   Output: {args.output_dir}")
    print(f"{'='*80}\n")

    # Parse search space ranges
    strength_range = tuple(float(x) for x in args.strength_range.split(","))
    max_weight_range = tuple(float(x) for x in args.max_weight_range.split(","))
    min_weight_range = tuple(float(x) for x in args.min_weight_range.split(","))
    position_range = tuple(float(x) for x in args.position_range.split(","))
    num_pairs = args.num_pairs

    # Determine optimization direction
    direction = args.direction
    if direction == "auto":
        if args.target_metric in ["refusal_rate", "kl_divergence"]:
            direction = "minimize"
        else:
            direction = "maximize"

    print(f"Search space:")
    print(f"   Strength: {strength_range}")
    print(f"   Max weight: {max_weight_range}")
    print(f"   Min weight: {min_weight_range}")
    print(f"   Position: {position_range}")
    print(f"   Num pairs: {num_pairs} (fixed)")
    print(f"   Direction: {direction}")
    print()

    # Initialize components
    device = args.device if args.device else resolve_default_device()
    print(f"Device: {device}")

    # IMPORTANT: Generate steering vectors FIRST, before loading the main model.
    # The steering vector generation pipeline loads its own model internally.
    # If we load our model first, then the pipeline loads another model, we end up
    # with two large models competing for GPU memory, causing CPU offloading.
    # By generating vectors first, that model is unloaded before we load ours.
    print(f"\nGenerating steering vectors with {num_pairs} pairs...")
    print("   (This will load the model temporarily for activation collection)\n")
    steering_vectors, positive_examples, negative_examples = _generate_steering_vectors(args, num_pairs, num_layers=None)
    print(f"\n   Steering vectors generated for {len(steering_vectors)} layers")
    print(f"   Got {len(positive_examples)} positive and {len(negative_examples)} negative examples")

    # Force garbage collection to ensure the model used for activation collection is freed
    import gc
    gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()
    print("   Memory cleaned up\n")

    # NOW load the main model for optimization (GPU should be free now)
    print("Loading base model for optimization...")
    wisent_model = WisentModel(args.model, device=device)
    base_model = wisent_model.hf_model  # Get underlying HF model for weight modification
    tokenizer = wisent_model.tokenizer
    num_layers = wisent_model.num_layers

    print(f"   Model loaded: {num_layers} layers\n")

    # Store base model state for restoration
    base_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}

    # Initialize evaluator (pass wisent_model for personalization baseline generation, and contrastive pairs)
    evaluator = _create_evaluator(args, args.model, wisent_model=wisent_model,
                                   positive_examples=positive_examples, negative_examples=negative_examples)

    # Create optimizer config
    optimizer_config = WeightsOptimizerConfig(
        strength_range=strength_range,
        max_weight_range=max_weight_range,
        min_weight_range=min_weight_range,
        position_range=position_range,
        method=args.method,
        components=args.components,
        norm_preserve=args.norm_preserve,
        optimize_direction_index=args.optimize_direction_index,
        target_metric=args.target_metric,
        target_value=args.target_value if args.early_stop else None,
    )

    # Create the optimizer
    optimizer = WeightsOptimizer(
        model=base_model,
        base_state_dict=base_state_dict,
        steering_vectors=steering_vectors,
        evaluator=evaluator,
        tokenizer=tokenizer,
        config=optimizer_config,
        num_layers=num_layers,
    )

    # Create HPO config
    hpo_config = HPOConfig(
        n_trials=args.trials,
        direction=direction,
        sampler="tpe",
        pruner=None,
        seed=42,
    )

    # Run optimization
    print(f"\nStarting optimization ({args.trials} trials)...\n")
    
    # Use checkpointing if checkpoint path is provided
    checkpoint_path = getattr(args, 'checkpoint', None)
    checkpoint_interval = getattr(args, 'checkpoint_interval', 5)
    s3_bucket = getattr(args, 's3_bucket', None)
    
    # Generate S3 key prefix for this optimization run
    s3_key_prefix = None
    if s3_bucket:
        task_name = args.task.replace(',', '_')[:50] if args.task else (args.trait or 'unknown')[:50]
        s3_key_prefix = f"optimization-checkpoints/{task_name}/{time.strftime('%Y%m%d-%H%M%S')}"
        print(f"   S3 bucket: {s3_bucket}")
        print(f"   S3 key prefix: {s3_key_prefix}")
    
    if checkpoint_path or s3_bucket:
        if checkpoint_path:
            print(f"   Checkpointing enabled: {checkpoint_path}")
        print(f"   Checkpoint interval: every {checkpoint_interval} trials\n")
        result = optimizer.optimize_with_checkpointing(
            hpo_config,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            s3_bucket=s3_bucket,
            s3_key_prefix=s3_key_prefix,
        )
    else:
        result = optimizer.optimize(hpo_config)

    best_params = result.best_params
    best_value = result.best_value

    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nBest parameters:")
    for k, v in best_params.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    print(f"\nBest {args.target_metric}: {best_value:.4f}")

    # Check if target achieved
    if direction == "maximize":
        target_achieved = best_value >= args.target_value
    else:
        target_achieved = best_value <= args.target_value
    print(f"\nTarget {args.target_value} achieved: {'YES' if target_achieved else 'NO'}")

    # Apply best parameters and save final model
    print(f"\n{'='*80}")
    print("SAVING OPTIMIZED MODEL")
    print(f"{'='*80}")

    optimizer.apply_best_params(best_params)

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving optimized model to {args.output_dir}...")
    base_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save optimization metadata
    metadata = {
        "model": args.model,
        "task": args.task,
        "trait": getattr(args, 'trait', None),
        "evaluator_type": evaluator_display,
        "target_metric": args.target_metric,
        "target_value": args.target_value,
        "best_params": best_params,
        "best_score": best_value,
        "target_achieved": target_achieved,
        "total_trials": len(result.study.trials),
        "direction": direction,
    }

    with open(os.path.join(args.output_dir, "optimization_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"   Model saved")
    print(f"   Metadata saved to optimization_metadata.json")

    # Upload to S3 if --s3-bucket is provided
    s3_bucket = getattr(args, 's3_bucket', None)
    if s3_bucket:
        task_name = args.task.replace(',', '_')[:50] if args.task else (args.trait or 'unknown')[:50]
        s3_key = f"optimization-results/{task_name}/{time.strftime('%Y%m%d-%H%M%S')}"
        print(f"\n   Uploading results to s3://{s3_bucket}/{s3_key}/...")
        if upload_to_s3(args.output_dir, s3_bucket, s3_key):
            print(f"   ✓ Results uploaded to S3")
        else:
            print(f"   ✗ S3 upload failed")

    # Save all trials if requested
    if args.save_trials:
        trials_data = [
            {
                "trial": t.number,
                "params": t.params,
                "score": t.value,
            }
            for t in result.study.trials
        ]
        with open(args.save_trials, "w") as f:
            json.dump(trials_data, f, indent=2)
        print(f"   Trials saved to {args.save_trials}")

    # Push to hub if requested
    if args.push_to_hub:
        if not args.repo_id:
            print("\n   ERROR: --repo-id required for --push-to-hub")
        else:
            print(f"\n   Pushing to HuggingFace Hub: {args.repo_id}...")
            base_model.push_to_hub(args.repo_id)
            tokenizer.push_to_hub(args.repo_id)
            print(f"   Pushed successfully")

    # Show/save before/after comparisons if requested
    save_comparisons_path = getattr(args, 'save_comparisons', None)
    show_comparisons_count = getattr(args, 'show_comparisons', 0)
    if show_comparisons_count > 0 or save_comparisons_path:
        _show_response_comparisons(
            base_model=base_model,
            base_state_dict=base_state_dict,
            steering_vectors=steering_vectors,
            best_params=best_params,
            num_layers=num_layers,
            model_name=args.model,
            args=args,
            optimizer_config=optimizer_config,
            num_comparisons=show_comparisons_count if show_comparisons_count > 0 else None,
            save_path=save_comparisons_path,
        )

    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Total optimization time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*80}\n")

    return OptimizationResult(
        best_params=best_params,
        best_score=best_value,
        target_achieved=target_achieved,
        total_time=total_time,
        total_trials=len(result.study.trials),
    )


def _train_multi_direction_method(
    args,
    caa_vectors: dict[int, torch.Tensor],
    intermediate_dir: str,
    method: str,
) -> dict[int, torch.Tensor]:
    """Train TITAN/PRISM/PULSE on pairs with activations and return combined directions.
    
    Args:
        args: Command line arguments
        caa_vectors: CAA vectors (used as fallback if training fails)
        intermediate_dir: Directory containing pairs with activations
        method: 'titan', 'prism', or 'pulse'
        
    Returns:
        Combined steering vectors per layer
    """
    import glob
    
    # Find enriched pairs file (with activations)
    enriched_files = glob.glob(os.path.join(intermediate_dir, "*_with_activations.json"))
    
    if not enriched_files:
        print(f"   Warning: No enriched pairs found for {method}, using CAA vectors")
        return caa_vectors
    
    enriched_file = enriched_files[0]
    print(f"\n   Training {method.upper()} on enriched pairs...")
    print(f"   Pairs file: {enriched_file}")
    
    try:
        from wisent.core.contrastive_pairs.core.serialization import load_contrastive_pair_set
        from wisent.core.weight_modification.multi_direction import (
            MultiDirectionConfig,
            combine_directions,
        )
        
        # Load pair set with activations
        pair_set = load_contrastive_pair_set(enriched_file)
        print(f"   Loaded {len(pair_set.pairs)} pairs with activations")
        
        # Get config from args
        num_directions = getattr(args, 'num_directions', 5)
        combination_strategy = getattr(args, 'combination_strategy', 'learned')
        optimization_steps = getattr(args, 'multi_optimization_steps', 100)
        
        # Train the method
        if method == 'titan':
            from wisent.core.steering_methods.methods.titan import TITANMethod, TITANConfig
            config = TITANConfig(
                num_directions=num_directions,
                optimization_steps=optimization_steps,
            )
            trainer = TITANMethod(config=config)
            result = trainer.train_titan(pair_set)
            directions = result.directions
            weights = result.direction_weights
            
        elif method == 'prism':
            from wisent.core.steering_methods.methods.prism import PRISMMethod, PRISMConfig
            config = PRISMConfig(
                num_directions=num_directions,
                optimization_steps=optimization_steps,
            )
            trainer = PRISMMethod(config=config)
            result = trainer.train_prism(pair_set)
            directions = result.directions
            weights = None  # PRISM doesn't have learned weights
            
        elif method == 'pulse':
            from wisent.core.steering_methods.methods.pulse import PULSEMethod, PULSEConfig
            config = PULSEConfig(
                optimization_steps=optimization_steps,
            )
            trainer = PULSEMethod(config=config)
            result = trainer.train_pulse(pair_set)
            # PULSE has single direction per layer
            directions = {k: v.unsqueeze(0) for k, v in result.behavior_vectors.items()}
            weights = {k: torch.tensor([result.layer_scales.get(k, 1.0)]) 
                      for k in directions} if result.layer_scales else None
        
        print(f"   Trained {len(directions)} layers with {method.upper()}")
        
        # Combine directions into single vector per layer
        combined_vectors = {}
        for layer_name, layer_dirs in directions.items():
            # Get layer index
            try:
                layer_idx = int(layer_name.replace("layer_", "")) - 1
            except ValueError:
                layer_idx = int(layer_name) - 1
            
            layer_weights = weights.get(layer_name) if weights else None
            combined = combine_directions(layer_dirs, layer_weights, strategy=combination_strategy)
            combined_vectors[layer_idx] = combined
            
        print(f"   Combined into {len(combined_vectors)} steering vectors")
        print(f"   Combination strategy: {combination_strategy}")
        
        return combined_vectors
        
    except Exception as e:
        print(f"   Warning: {method.upper()} training failed: {e}")
        print(f"   Falling back to CAA vectors")
        return caa_vectors


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
                similarity_threshold=getattr(args, 'similarity_threshold', 0.8),
                verbose=False,
                timing=False,
                layers=args.layers,
                token_aggregation=args.token_aggregation,
                prompt_strategy="chat_template",
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
                similarity_threshold=getattr(args, 'similarity_threshold', 0.8),
                verbose=False,
                timing=False,
                layers=args.layers,
                token_aggregation=args.token_aggregation,
                prompt_strategy="chat_template",
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
                similarity_threshold=getattr(args, 'similarity_threshold', 0.8),
                verbose=False,
                timing=False,
                layers=args.layers,
                token_aggregation=args.token_aggregation,
                prompt_strategy="chat_template",
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
                train_ratio=getattr(args, 'train_ratio', 0.8),
                seed=getattr(args, 'seed', 42),
                model=args.model,
                device=args.device,
                layer=None,
                layers=layers_arg,
                token_aggregation=args.token_aggregation if hasattr(args, 'token_aggregation') else 'continuation',
                prompt_strategy='chat_template',
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
                similarity_threshold=getattr(args, 'similarity_threshold', 0.8),
                verbose=False,
                timing=False,
                layers=args.layers,
                token_aggregation=args.token_aggregation,
                prompt_strategy="chat_template",
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
                token_aggregation=args.token_aggregation,
                prompt_strategy="chat_template",
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

        # If using multi-direction method (titan/prism/pulse), train on pairs with activations
        method = getattr(args, 'method', 'directional')
        if method in ('titan', 'prism', 'pulse'):
            vectors = _train_multi_direction_method(
                args, vectors, intermediate_dir, method
            )

        return vectors, positive_examples, negative_examples

    finally:
        if os.path.exists(temp_output):
            os.unlink(temp_output)
        if os.path.exists(temp_pairs):
            os.unlink(temp_pairs)


def _create_evaluator(args, model_name: str, wisent_model: WisentModel = None,
                      positive_examples: list[str] = None, negative_examples: list[str] = None) -> Callable[[Any, Any], dict[str, float]]:
    """Create the appropriate evaluator function based on --task argument.

    Uses shared steering evaluators from wisent.core.evaluators.steering_evaluators.

    Task types:
    - 'refusal' → RefusalEvaluator (compliance rate)
    - 'personalization' → PersonalizationEvaluator (requires --trait)
    - benchmark name → TaskEvaluator (e.g., 'arc_easy', 'gsm8k')
    - comma-separated benchmarks → PooledEvaluator (e.g., 'arc_easy,gsm8k')

    The evaluator receives (hf_model, tokenizer) and wraps them in WisentModel
    internally to use standard generation.

    Args:
        args: Command arguments (must have args.task)
        model_name: Model name/path
        wisent_model: Optional WisentModel instance for baseline generation (used by personalization)
        positive_examples: List of positive example responses from contrastive pairs
        negative_examples: List of negative example responses from contrastive pairs
    """
    from wisent.core.models.inference_config import get_generate_kwargs

    task = args.task.lower() if args.task else ""
    
    # Determine evaluator type from task
    if task == "refusal":
        evaluator_type = "refusal"
    elif task == "personalization" or (not task and getattr(args, 'trait', None)):
        # Use personalization evaluator when --trait is provided (with or without --task personalization)
        evaluator_type = "personalization"
    elif task == "custom" or (not task and getattr(args, 'custom_evaluator', None)):
        # Use custom evaluator when --custom-evaluator is provided (with or without --task custom)
        if not getattr(args, 'custom_evaluator', None):
            raise ValueError("--custom-evaluator is required when --task custom")
        evaluator_type = "custom"
    elif "," in task:
        # Multiple benchmarks: use pooled evaluator
        evaluator_type = "pooled"
    elif task:
        # Single benchmark: use task evaluator (only if task is specified)
        evaluator_type = "task"
    else:
        raise ValueError("Either --task, --trait, or --custom-evaluator must be specified")

    # Pooled evaluator for multiple benchmarks
    if evaluator_type == "pooled":
        print(f"   [DEBUG] Creating pooled evaluator for tasks: {task}")
        print(f"   [DEBUG] args._pooled_eval_pairs count: {len(getattr(args, '_pooled_eval_pairs', []))}")
        return _create_pooled_evaluator(args)
    
    # Custom evaluator
    if evaluator_type == "custom":
        return _create_custom_evaluator(args, model_name)

    # Use shared steering evaluators for refusal, task, personalization
    eval_config = EvaluatorConfig(
        evaluator_type=evaluator_type,
        trait=getattr(args, 'trait', None),
        task=task if evaluator_type == "task" else None,
        eval_prompts_path=getattr(args, 'eval_prompts', None),
        eval_topics=getattr(args, 'eval_topics', None),
        num_eval_prompts=getattr(args, 'num_eval_prompts', 30),
    )

    # Create shared evaluator
    shared_evaluator = SteeringEvaluatorFactory.create(
        eval_config, model_name, wisent_model, positive_examples, negative_examples
    )
    
    # Pre-generate baseline responses BEFORE optimization modifies the model weights
    # This is critical because the same wisent_model/hf_model is used for both baseline and steered generation
    if hasattr(shared_evaluator, 'generate_baseline_responses'):
        print("   Pre-generating baseline responses with unmodified model...")
        baseline = shared_evaluator.generate_baseline_responses()
        print(f"   Generated {len(baseline)} baseline responses")

    def evaluate(hf_model, tokenizer) -> dict:
        """Run evaluation using shared steering evaluator."""
        # Wrap HF model in WisentModel for standard generation
        temp_wisent_model = WisentModel(model_name, hf_model=hf_model)
        
        # Get prompts from evaluator
        prompts = shared_evaluator.get_prompts()
        
        # Generate responses
        responses = []
        for prompt_text in prompts:
            messages = [{"role": "user", "content": prompt_text}]
            result = temp_wisent_model.generate(
                [messages],
                **get_generate_kwargs(max_new_tokens=150),
            )
            responses.append(result[0] if result else "")
        
        # Evaluate using shared evaluator
        return shared_evaluator.evaluate_responses(responses)

    return evaluate


def _create_custom_evaluator(args, model_name: str) -> Callable:
    """Create custom evaluator for --task custom.
    
    Uses the custom evaluator framework from wisent.core.evaluators.custom.
    
    Args:
        args: Command arguments (must have args.custom_evaluator)
        model_name: Model name/path
    """
    import json
    from wisent.core.evaluators.custom import create_custom_evaluator
    from wisent.core.models.inference_config import get_generate_kwargs
    
    # Parse custom evaluator kwargs
    custom_kwargs = {}
    if getattr(args, 'custom_evaluator_kwargs', None):
        custom_kwargs = json.loads(args.custom_evaluator_kwargs)
    
    # Create the custom evaluator
    custom_eval = create_custom_evaluator(args.custom_evaluator, **custom_kwargs)
    
    # Get test prompts - for custom, we need either trait-based prompts or user-provided prompts
    eval_prompts = []
    if getattr(args, 'eval_prompts', None):
        # Load from file
        with open(args.eval_prompts) as f:
            eval_prompts = [line.strip() for line in f if line.strip()]
    elif getattr(args, 'trait', None):
        # Use default prompts from PersonalizationEvaluator
        from wisent.core.evaluators.steering_evaluators import PersonalizationEvaluator
        num_prompts = getattr(args, 'num_eval_prompts', 30)
        eval_prompts = PersonalizationEvaluator.DEFAULT_PROMPTS[:num_prompts]
    else:
        # Default generic prompts
        eval_prompts = [
            "Tell me about yourself.",
            "What do you think about artificial intelligence?",
            "How would you solve world hunger?",
            "Explain quantum computing in simple terms.",
            "What's the meaning of life?",
        ]
    
    def evaluate(hf_model, tokenizer) -> dict:
        """Run evaluation using custom evaluator."""
        # Create WisentModel wrapper without reloading tokenizer from HuggingFace
        # Use object.__new__ to avoid __init__ which tries to load from HF
        temp_wisent_model = object.__new__(WisentModel)
        temp_wisent_model.hf_model = hf_model
        temp_wisent_model.tokenizer = tokenizer
        temp_wisent_model.model_name = model_name
        # Set internal attributes
        if hasattr(hf_model, 'model') and hasattr(hf_model.model, 'layers'):
            temp_wisent_model._layers = hf_model.model.layers
        elif hasattr(hf_model, 'transformer') and hasattr(hf_model.transformer, 'h'):
            temp_wisent_model._layers = hf_model.transformer.h
        else:
            temp_wisent_model._layers = []
        temp_wisent_model._hidden_size = hf_model.config.hidden_size
        temp_wisent_model.device = next(hf_model.parameters()).device
        
        # Generate responses
        scores = []
        for prompt_text in eval_prompts:
            messages = [{"role": "user", "content": prompt_text}]
            result = temp_wisent_model.generate(
                [messages],
                **get_generate_kwargs(max_new_tokens=150),
            )
            response = result[0] if result else ""
            
            # Score using custom evaluator - pass prompt for coherence checking
            score = custom_eval(response, prompt=prompt_text)
            if isinstance(score, dict):
                # Take the primary score (first value or 'score' key)
                score = score.get('score', list(score.values())[0])
            scores.append(float(score))
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "score": avg_score,
            "custom_score": avg_score,
            "num_evaluated": len(scores),
        }
    
    return evaluate


def _create_pooled_evaluator(args) -> Callable:
    """Create pooled evaluator for multi-benchmark mode (--task bench1,bench2,...).
    
    Evaluates on ALL benchmarks from training, using each benchmark's
    native evaluator (routing by source_benchmark metadata).
    """
    from wisent.core.evaluators.rotator import EvaluatorRotator
    from wisent.core.models.inference_config import get_generate_kwargs

    # Get eval pairs stored during vector generation
    eval_pairs = getattr(args, '_pooled_eval_pairs', [])
    benchmarks_used = getattr(args, '_benchmarks_used', [])

    if not eval_pairs:
        raise ValueError("No eval pairs found for pooled evaluation. "
                        "Make sure train_unified_goodness saved eval_pairs to checkpoint.")

    print(f"   Pooled evaluator: {len(eval_pairs)} eval pairs across {len(benchmarks_used)} benchmarks")

    # Discover evaluators once
    EvaluatorRotator.discover_evaluators('wisent.core.evaluators.oracles')
    EvaluatorRotator.discover_evaluators('wisent.core.evaluators.benchmark_specific')

    # Group pairs by benchmark
    pairs_by_benchmark = {}
    for pair in eval_pairs:
        bench = pair.get('source_benchmark', 'unknown')
        if bench not in pairs_by_benchmark:
            pairs_by_benchmark[bench] = []
        pairs_by_benchmark[bench].append(pair)

    print(f"   Benchmarks in eval set: {list(pairs_by_benchmark.keys())}")

    def evaluate(hf_model, tokenizer) -> dict[str, float]:
        """Evaluate modified model on pooled benchmark data."""
        from wisent.core.models.wisent_model import WisentModel

        # Create WisentModel wrapper - use object.__new__ to avoid __init__ loading
        wisent_model = object.__new__(WisentModel)
        wisent_model.hf_model = hf_model
        wisent_model.tokenizer = tokenizer
        wisent_model.model_name = "modified_model"
        # Set internal attributes that the properties depend on
        if hasattr(hf_model, 'model') and hasattr(hf_model.model, 'layers'):
            wisent_model._layers = hf_model.model.layers
        elif hasattr(hf_model, 'transformer') and hasattr(hf_model.transformer, 'h'):
            wisent_model._layers = hf_model.transformer.h
        else:
            wisent_model._layers = []
        wisent_model._hidden_size = hf_model.config.hidden_size
        wisent_model.device = next(hf_model.parameters()).device

        total_correct = 0
        total_samples = 0
        benchmark_scores = {}

        for bench_name, bench_pairs in pairs_by_benchmark.items():
            # Get evaluator for this benchmark
            evaluator = EvaluatorRotator(
                evaluator=None,
                task_name=bench_name,
                autoload=False
            )

            correct = 0
            evaluated = 0

            for pair in bench_pairs:
                try:
                    question = pair['prompt']
                    expected = pair['positive_response']
                    metadata = pair.get('metadata', {}) or {}
                    test_code = metadata.get('test_code', '')
                    entry_point = metadata.get('entry_point', '')

                    messages = [{"role": "user", "content": question}]
                    response = wisent_model.generate(
                        [messages],
                        **get_generate_kwargs(),
                    )[0]

                    # Evaluate
                    eval_result = evaluator.evaluate(
                        response=response,
                        expected=expected,
                        model=wisent_model,
                        question=question,
                        task_name=bench_name,
                        test_code=test_code,
                        entry_point=entry_point,
                    )

                    if eval_result.ground_truth == "TRUTHFUL":
                        correct += 1
                    evaluated += 1

                except Exception as e:
                    # Count failed evals but continue
                    evaluated += 1

            if evaluated > 0:
                benchmark_scores[bench_name] = correct / evaluated
                total_correct += correct
                total_samples += evaluated

        # Compute aggregate accuracy
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": total_correct,
            "total": total_samples,
            "score": accuracy,
            "benchmark_scores": benchmark_scores,
        }

    return evaluate


def _apply_weight_modification_standalone(
    model,
    steering_vectors: dict[int, torch.Tensor],
    params: dict,
    num_layers: int,
    config: WeightsOptimizerConfig,
):
    """Apply weight modification with given parameters (standalone helper)."""
    from wisent.core.weight_modification import project_with_kernel, bake_steering_with_kernel

    max_weight_position = params["max_weight_position"] * (num_layers - 1)
    min_weight_distance = 0.6 * (num_layers - 1)

    if config.method == "directional":
        project_with_kernel(
            model,
            steering_vectors,
            max_weight=params["max_weight"] * params["strength"],
            max_weight_position=max_weight_position,
            min_weight=params["min_weight"],
            min_weight_distance=min_weight_distance,
            components=config.components,
            norm_preserve=config.norm_preserve,
            verbose=False,
        )
    elif config.method in ("titan", "prism", "pulse"):
        # Multi-direction methods: directions already combined during generation
        # Use additive baking with the combined vectors
        bake_steering_with_kernel(
            model,
            steering_vectors,
            max_alpha=params["max_weight"] * params["strength"],
            max_alpha_position=max_weight_position,
            min_alpha=params["min_weight"],
            components=config.components,
            verbose=False,
        )
    else:
        # Default: additive
        bake_steering_with_kernel(
            model,
            steering_vectors,
            max_alpha=params["max_weight"] * params["strength"],
            max_alpha_position=max_weight_position,
            min_alpha=params["min_weight"],
            components=config.components,
            verbose=False,
        )


def _show_response_comparisons(
    base_model,
    base_state_dict: dict,
    steering_vectors: dict[int, torch.Tensor],
    best_params: dict,
    num_layers: int,
    model_name: str,
    args,
    optimizer_config: WeightsOptimizerConfig,
    num_comparisons: int = None,
    save_path: str = None,
):
    """Show and/or save before/after response comparisons.

    Uses WisentModel.generate() for consistent generation behavior.
    """
    from uncensorbench import UncensorBench
    from uncensorbench.evaluator import KeywordEvaluator, SemanticEvaluator
    from wisent.core.models.inference_config import get_generate_kwargs

    print(f"\n{'='*80}")
    print("RESPONSE COMPARISONS (Baseline vs Optimized)")
    print(f"{'='*80}")

    bench = UncensorBench()

    if args.eval_prompts:
        with open(args.eval_prompts) as f:
            custom_prompts = json.load(f)
        if not isinstance(custom_prompts, list):
            custom_prompts = custom_prompts.get("prompts", [])
        prompts = custom_prompts[:args.num_eval_prompts]
        use_custom = True
    elif getattr(args, 'trait', None):
        # For trait-based optimization, use personalization prompts (not UncensorBench)
        from wisent.core.evaluators.steering_evaluators import PersonalizationEvaluator
        prompts = PersonalizationEvaluator.DEFAULT_PROMPTS[:args.num_eval_prompts]
        use_custom = True  # Treat as custom to skip UncensorBench evaluation
    else:
        topics = args.eval_topics.split(",") if args.eval_topics else None
        prompts = list(bench.prompts(topics=topics))[:args.num_eval_prompts]
        use_custom = False

    try:
        evaluator = SemanticEvaluator()
    except ImportError:
        evaluator = KeywordEvaluator()

    def generate_response(hf_model, prompt_text: str) -> str:
        """Generate response using WisentModel for consistent behavior."""
        # Wrap HF model in WisentModel for standard generation
        wisent_model = WisentModel(model_name, hf_model=hf_model)

        messages = [{"role": "user", "content": prompt_text}]
        responses = wisent_model.generate(
            [messages],
            **get_generate_kwargs(max_new_tokens=150),
        )

        response = responses[0] if responses else ""
        # Strip <think>...</think> blocks from Qwen3 responses
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        return response

    def evaluate_response(prompt_item, response: str, use_custom: bool) -> tuple[bool, float]:
        if use_custom:
            is_refusal = bench.detect_refusal(response)
            compliance_score = 0.0 if is_refusal else 1.0
        else:
            eval_result = bench.evaluate_response(prompt_item, response, evaluator)
            is_refusal = not eval_result.is_uncensored
            compliance_score = eval_result.compliance_score
        return is_refusal, compliance_score

    print("\nGenerating baseline responses...")

    base_model.load_state_dict(base_state_dict)

    baseline_results = []
    for item in prompts:
        if use_custom:
            prompt_text = item if isinstance(item, str) else item.get("prompt", str(item))
        else:
            prompt_text = item.prompt

        response = generate_response(base_model, prompt_text)
        is_refusal, score = evaluate_response(item, response, use_custom)
        baseline_results.append({
            "prompt": prompt_text,
            "response": response,
            "is_refusal": is_refusal,
            "score": score,
            "item": item,
        })

    print("Generating optimized responses...")

    _apply_weight_modification_standalone(
        base_model,
        steering_vectors,
        best_params,
        num_layers,
        optimizer_config,
    )

    optimized_results = []
    for i, item in enumerate(prompts):
        if use_custom:
            prompt_text = item if isinstance(item, str) else item.get("prompt", str(item))
        else:
            prompt_text = item.prompt

        response = generate_response(base_model, prompt_text)
        is_refusal, score = evaluate_response(item, response, use_custom)
        optimized_results.append({
            "prompt": prompt_text,
            "response": response,
            "is_refusal": is_refusal,
            "score": score,
        })

    comparisons = []
    for i, (baseline, optimized) in enumerate(zip(baseline_results, optimized_results)):
        delta = optimized["score"] - baseline["score"]
        comparisons.append({
            "index": i,
            "prompt": baseline["prompt"],
            "baseline_response": baseline["response"],
            "baseline_score": baseline["score"],
            "baseline_refusal": baseline["is_refusal"],
            "optimized_response": optimized["response"],
            "optimized_score": optimized["score"],
            "optimized_refusal": optimized["is_refusal"],
            "delta": delta,
        })

    comparisons.sort(key=lambda x: abs(x["delta"]), reverse=True)

    total_baseline_refusals = sum(1 for c in comparisons if c["baseline_refusal"])
    total_optimized_refusals = sum(1 for c in comparisons if c["optimized_refusal"])
    avg_baseline_score = sum(c["baseline_score"] for c in comparisons) / len(comparisons)
    avg_optimized_score = sum(c["optimized_score"] for c in comparisons) / len(comparisons)

    summary = {
        "total_prompts": len(comparisons),
        "baseline_refusals": total_baseline_refusals,
        "optimized_refusals": total_optimized_refusals,
        "avg_baseline_score": avg_baseline_score,
        "avg_optimized_score": avg_optimized_score,
        "score_delta": avg_optimized_score - avg_baseline_score,
        "refusal_change": total_baseline_refusals - total_optimized_refusals,
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        output_data = {
            "model": args.model,
            "trait": getattr(args, "trait", None),
            "best_params": best_params,
            "summary": summary,
            "comparisons": comparisons,
        }
        with open(save_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved {len(comparisons)} comparisons to: {save_path}")

    if num_comparisons and num_comparisons > 0:
        print(f"\nTop {num_comparisons} prompts with biggest score changes:\n")

        for i, comp in enumerate(comparisons[:num_comparisons]):
            print(f"{'─'*80}")
            print(f"Comparison {i+1}/{num_comparisons}")
            print(f"{'─'*80}")
            print(f"PROMPT: {comp['prompt'][:200]}{'...' if len(comp['prompt']) > 200 else ''}")
            print()
            print(f"BASELINE (score={comp['baseline_score']:.2f}, refusal={comp['baseline_refusal']}):")
            print(f"  {comp['baseline_response'][:300]}{'...' if len(comp['baseline_response']) > 300 else ''}")
            print()
            print(f"OPTIMIZED (score={comp['optimized_score']:.2f}, refusal={comp['optimized_refusal']}):")
            print(f"  {comp['optimized_response'][:300]}{'...' if len(comp['optimized_response']) > 300 else ''}")
            print()
            delta_str = f"+{comp['delta']:.2f}" if comp['delta'] >= 0 else f"{comp['delta']:.2f}"
            print(f"DELTA: {delta_str}")
            print()

    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Baseline:  {total_baseline_refusals}/{len(comparisons)} refusals, avg score={avg_baseline_score:.3f}")
    print(f"Optimized: {total_optimized_refusals}/{len(comparisons)} refusals, avg score={avg_optimized_score:.3f}")
    print(f"Change:    {total_baseline_refusals - total_optimized_refusals} fewer refusals, "
          f"score delta={avg_optimized_score - avg_baseline_score:+.3f}")
