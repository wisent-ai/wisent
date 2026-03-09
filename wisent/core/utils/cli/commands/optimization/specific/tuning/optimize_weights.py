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

import gc
import json
import os
import time
from dataclasses import dataclass

import torch

from wisent.core.utils.config_tools.constants import DISPLAY_TRUNCATION_EVAL, DISPLAY_TRUNCATION_SHORT, TRIALS_PER_DIMENSION_MULTIPLIER
from wisent.core.utils.infra_tools.errors import UnknownTypeError, InsufficientDataError
from wisent.core.primitives.models.wisent_model import WisentModel
from wisent.core.reading.evaluators.steering_evaluators import (
    SteeringEvaluatorFactory,
    EvaluatorConfig,
    RefusalEvaluator,
    TaskEvaluator,
    PersonalizationEvaluator as SharedPersonalizationEvaluator,
)
from wisent.core.utils.services.optimization.core.atoms import HPOConfig
from wisent.core.utils.services.optimization.methods.opti_weights import WeightsOptimizer, WeightsOptimizerConfig
from wisent.core.utils import resolve_default_device, preferred_dtype
from wisent.core.utils.cli.optimization.specific.optimize_weights_vectors import _generate_steering_vectors
from wisent.core.utils.cli.optimization.specific.optimize_weights_evaluators import _create_evaluator
from wisent.core.utils.cli.optimization.specific.optimize_weights_finalize import (
    upload_to_gcs, _finalize_optimization,
)


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
    from wisent.core.control.tasks.base.task_selector import expand_task_if_skill_or_risk
    if getattr(args, 'task', None):
        args.task = expand_task_if_skill_or_risk(args.task)
    
    start_time = time.time()

    print(f"\n{'='*80}")
    print("WEIGHT MODIFICATION OPTIMIZATION")
    print(f"{'='*80}")
    print(f"   Model: {args.model}")
    if args.trait:
        print(f"   Mode: Trait-based (synthetic pairs)")
        print(f"   Trait: {args.trait[:DISPLAY_TRUNCATION_EVAL]}...")
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
        weight_min_distance_fraction=args.weight_min_distance_fraction,
        method=args.method,
        components=args.components,
        norm_preserve=args.norm_preserve,
        optimize_direction_index=args.optimize_direction_index,
        target_metric=args.target_metric,
        target_value=args.target_value if args.early_stop else None,
        kernel_center_divisor=args.kernel_center_divisor,
        kernel_sigma_divisor=args.kernel_sigma_divisor,
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

    # Derive n_trials from weight search space dimensions
    weight_dimensions = [strength_range, max_weight_range, min_weight_range, position_range]
    if args.optimize_direction_index:
        weight_dimensions.append(("direction_index",))
    n_trials = getattr(args, 'trials', None) or len(weight_dimensions) * TRIALS_PER_DIMENSION_MULTIPLIER

    # Create HPO config
    hpo_config = HPOConfig(
        n_trials=n_trials,
        direction=direction,
        sampler="tpe",
        pruner=None,
        seed=args.seed,
    )

    # Run optimization
    print(f"\nStarting optimization ({n_trials} trials)...\n")
    
    # Use checkpointing if checkpoint path is provided
    checkpoint_path = getattr(args, 'checkpoint', None)
    checkpoint_interval = args.checkpoint_interval
    gcs_bucket = getattr(args, 'gcs_bucket', None)

    # Generate GCS key prefix for this optimization run
    gcs_key_prefix = None
    if gcs_bucket:
        task_name = args.task.replace(',', '_')[:DISPLAY_TRUNCATION_SHORT] if args.task else (args.trait or 'unknown')[:DISPLAY_TRUNCATION_SHORT]
        gcs_key_prefix = f"optimization-checkpoints/{task_name}/{time.strftime('%Y%m%d-%H%M%S')}"
        print(f"   GCS bucket: {gcs_bucket}")
        print(f"   GCS key prefix: {gcs_key_prefix}")

    if checkpoint_path or gcs_bucket:
        if checkpoint_path:
            print(f"   Checkpointing enabled: {checkpoint_path}")
        print(f"   Checkpoint interval: every {checkpoint_interval} trials\n")
        result = optimizer.optimize_with_checkpointing(
            hpo_config,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            gcs_bucket=gcs_bucket,
            gcs_key_prefix=gcs_key_prefix,
        )
    else:
        result = optimizer.optimize(hpo_config)


    return _finalize_optimization(
        args=args, result=result, direction=direction,
        evaluator_display=evaluator_display, base_model=base_model,
        tokenizer=tokenizer, optimizer=optimizer,
        steering_vectors=steering_vectors, base_state_dict=base_state_dict,
        num_layers=num_layers, optimizer_config=optimizer_config,
        start_time=start_time,
    )
