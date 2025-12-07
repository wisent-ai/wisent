"""
Optimize all parameters for a model.

Runs all optimization routines and saves results to the unified config manager:
1. Classification optimization (layer, threshold, aggregation)
2. Steering optimization (layer, strength)
3. Weight modification optimization (directional projection parameters)

Results are saved per-task and as defaults in ~/.wisent/configs/{model}.json
"""

import argparse
import logging
import time
from typing import List, Optional, Dict, Any

from wisent.core.config_manager import (
    get_config_manager,
    save_classification_config,
    save_steering_config,
    save_weight_modification_config,
    save_trait_steering_config,
    save_trait_weight_modification_config,
)

logger = logging.getLogger(__name__)


def execute_optimize_all(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Execute full optimization pipeline for a model.

    Runs:
    1. Classification optimization (unless --skip-classification)
    2. Steering optimization (unless --skip-steering)
    3. Weight modification optimization (unless --skip-weights)

    Args:
        args: Parsed command line arguments

    Returns:
        Dict with optimization results
    """
    results = {
        "model": args.model,
        "tasks": args.tasks if hasattr(args, 'tasks') and args.tasks else [],
        "traits": args.traits if hasattr(args, 'traits') and args.traits else [],
        "classification": None,
        "steering": None,
        "weight_modification": None,
        "errors": [],
    }

    print(f"\n{'='*60}")
    print(f"OPTIMIZING ALL PARAMETERS FOR: {args.model}")
    print(f"{'='*60}\n")

    # Determine what to optimize
    tasks = args.tasks if hasattr(args, 'tasks') and args.tasks else []
    traits = args.traits if hasattr(args, 'traits') and args.traits else []

    if not tasks and not traits:
        print("No tasks or traits specified. Using default benchmark tasks.")
        tasks = ["hellaswag", "arc_easy", "winogrande"]

    start_time = time.time()

    # 1. Classification Optimization
    if not getattr(args, 'skip_classification', False):
        print(f"\n[1/3] Classification Optimization")
        print(f"{'-'*40}")
        try:
            results["classification"] = _run_classification_optimization(args, tasks)
            print(f"  ✓ Classification optimization complete")
        except Exception as e:
            error_msg = f"Classification optimization failed: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            print(f"  ✗ {error_msg}")
    else:
        print(f"\n[1/3] Classification Optimization - SKIPPED")

    # 2. Steering Optimization
    if not getattr(args, 'skip_steering', False):
        print(f"\n[2/3] Steering Optimization")
        print(f"{'-'*40}")
        try:
            # Optimize for tasks
            if tasks:
                results["steering"] = _run_steering_optimization(args, tasks)
                print(f"  ✓ Task steering optimization complete")

            # Optimize for traits
            if traits:
                results["trait_steering"] = _run_trait_steering_optimization(args, traits)
                print(f"  ✓ Trait steering optimization complete")
        except Exception as e:
            error_msg = f"Steering optimization failed: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            print(f"  ✗ {error_msg}")
    else:
        print(f"\n[2/3] Steering Optimization - SKIPPED")

    # 3. Weight Modification Optimization
    if not getattr(args, 'skip_weights', False):
        print(f"\n[3/3] Weight Modification Optimization")
        print(f"{'-'*40}")
        try:
            # Optimize for traits (weight mod is typically per-trait)
            if traits:
                results["weight_modification"] = _run_weight_modification_optimization(args, traits)
                print(f"  ✓ Weight modification optimization complete")
            else:
                print(f"  - No traits specified, skipping weight modification")
        except Exception as e:
            error_msg = f"Weight modification optimization failed: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            print(f"  ✗ {error_msg}")
    else:
        print(f"\n[3/3] Weight Modification Optimization - SKIPPED")

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Tasks optimized: {len(tasks)}")
    print(f"Traits optimized: {len(traits)}")
    if results["errors"]:
        print(f"Errors: {len(results['errors'])}")
        for err in results["errors"]:
            print(f"  - {err}")

    # Show where configs are saved
    config_manager = get_config_manager()
    config_path = config_manager._get_config_path(args.model)
    print(f"\nConfig saved to: {config_path}")

    return results


def _run_classification_optimization(args: argparse.Namespace, tasks: List[str]) -> Dict[str, Any]:
    """Run classification optimization for tasks."""
    from wisent.core.cli.optimize_classification import execute_optimize_classification

    # Create args namespace for classification optimization
    class ClassificationArgs:
        pass

    opt_args = ClassificationArgs()
    opt_args.model = args.model
    opt_args.tasks = tasks
    opt_args.limit = getattr(args, 'limit', 100)
    opt_args.device = getattr(args, 'device', None)
    opt_args.verbose = getattr(args, 'verbose', False)
    opt_args.save_plots = getattr(args, 'save_plots', False)
    opt_args.layer_range = getattr(args, 'layer_range', None)
    opt_args.skip_full_search = getattr(args, 'skip_full_search', False)
    opt_args.quick = getattr(args, 'quick', False)
    opt_args.aggregation_methods = getattr(args, 'aggregation_methods', ['average', 'final', 'first', 'max', 'min'])
    opt_args.threshold_range = getattr(args, 'threshold_range', [0.3, 0.5, 0.7])
    opt_args.optimization_metric = getattr(args, 'optimization_metric', 'f1')

    return execute_optimize_classification(opt_args)


def _run_steering_optimization(args: argparse.Namespace, tasks: List[str]) -> Dict[str, Any]:
    """Run steering optimization for tasks."""
    from wisent.core.cli.optimize_steering import execute_optimize_steering

    class SteeringArgs:
        pass

    opt_args = SteeringArgs()
    opt_args.model = args.model
    opt_args.tasks = tasks
    opt_args.limit = getattr(args, 'limit', 100)
    opt_args.device = getattr(args, 'device', None)
    opt_args.verbose = getattr(args, 'verbose', False)
    opt_args.save_plots = getattr(args, 'save_plots', False)
    opt_args.layer_range = getattr(args, 'steering_layer_range', None)
    opt_args.strength_range = getattr(args, 'steering_strength_range', [0.5, 1.0, 1.5, 2.0])
    opt_args.methods = getattr(args, 'steering_methods', ['CAA'])
    opt_args.n_trials = getattr(args, 'n_trials', 50)
    opt_args.use_optuna = getattr(args, 'use_optuna', True)

    return execute_optimize_steering(opt_args)


def _run_trait_steering_optimization(args: argparse.Namespace, traits: List[str]) -> Dict[str, Any]:
    """Run steering optimization for traits using the personalization optimization pipeline."""
    from wisent.core.cli.optimize_steering import execute_personalization
    from wisent.core.models.wisent_model import WisentModel
    from argparse import Namespace
    import os

    results = {}
    output_dir = getattr(args, 'output_dir', './data/trait_steering')
    num_pairs = getattr(args, 'num_pairs', 50)
    num_test_prompts = getattr(args, 'num_test_prompts', 20)

    # Load model once for all traits
    print(f"  Loading model for trait steering optimization...")
    model = WisentModel(args.model, device=getattr(args, 'device', None))
    num_layers = model.num_layers
    print(f"  Model loaded with {num_layers} layers")

    # Determine which layers to test (middle 60% of layers typically work best for steering)
    layer_range = getattr(args, 'steering_layer_range', None)
    if layer_range:
        layers_to_test = list(range(layer_range[0], layer_range[1] + 1))
    else:
        # Test middle layers (20%-80% of model depth)
        start_layer = max(1, int(num_layers * 0.2))
        end_layer = min(num_layers, int(num_layers * 0.8))
        # Sample ~8 layers across this range
        step = max(1, (end_layer - start_layer) // 8)
        layers_to_test = list(range(start_layer, end_layer + 1, step))

    for trait in traits:
        print(f"\n  {'='*50}")
        print(f"  Optimizing steering for trait: {trait}")
        print(f"  {'='*50}")

        # Create trait-specific output directory
        trait_name = trait.split()[0].lower().replace('/', '_')[:30]
        trait_output_dir = os.path.join(output_dir, f"{args.model.replace('/', '_')}_{trait_name}")
        os.makedirs(trait_output_dir, exist_ok=True)

        try:
            # Build args for personalization optimization
            opt_args = Namespace(
                model=args.model,
                trait=trait,
                trait_name=trait_name,
                num_pairs=num_pairs,
                num_test_prompts=num_test_prompts,
                output_dir=trait_output_dir,
                device=getattr(args, 'device', None),
                verbose=getattr(args, 'verbose', False),
                # Layers to test
                layers=layers_to_test,
                # Strength range (tuple: min, max)
                strength_range=(0.5, 2.5),
                # Number of strength values to test
                num_strength_steps=5,
                # Save generation examples for comparison
                save_generation_examples=True,
                save_all_generation_examples=False,
            )

            # Run personalization optimization
            result = execute_personalization(opt_args, model)

            if result and result.get('best_config'):
                best = result['best_config']
                
                # Save to trait config
                save_trait_steering_config(
                    model_name=args.model,
                    trait_name=trait_name,
                    layer=best.get('layer', 12),
                    strength=best.get('strength', 1.0),
                    method=best.get('method', 'CAA'),
                    token_aggregation=best.get('token_aggregation', 'last'),
                    prompt_strategy=best.get('prompt_construction', 'chat_template'),
                    optimization_method="personalization",
                    score=best.get('score', 0.0),
                )

                results[trait] = {
                    "status": "success",
                    "layer": best.get('layer', 12),
                    "strength": best.get('strength', 1.0),
                    "method": best.get('method', 'CAA'),
                    "token_aggregation": best.get('token_aggregation', 'last'),
                    "score": best.get('score', 0.0),
                    "vector_path": result.get('vector_path'),
                    "results_file": result.get('results_file'),
                }
                print(f"  ✓ Trait '{trait}' optimized: layer={best.get('layer')}, strength={best.get('strength')}, score={best.get('score', 0):.3f}")
            else:
                raise ValueError("No best config returned from optimization")

        except Exception as e:
            logger.warning(f"Trait steering optimization failed for {trait}: {e}")
            import traceback
            traceback.print_exc()
            
            # Save default config on failure
            save_trait_steering_config(
                model_name=args.model,
                trait_name=trait_name,
                layer=12,
                strength=1.0,
                method="CAA",
                optimization_method="default",
            )
            results[trait] = {"status": "default", "error": str(e), "layer": 12, "strength": 1.0}

    return results


def _run_weight_modification_optimization(args: argparse.Namespace, traits: List[str]) -> Dict[str, Any]:
    """Run weight modification optimization for traits using execute_optimize_weights."""
    from wisent.core.cli.optimize_weights import execute_optimize_weights
    from argparse import Namespace

    results = {}
    n_trials = getattr(args, 'n_trials', 200)
    output_dir = getattr(args, 'output_dir', './data/modified_models')

    for trait in traits:
        print(f"  Optimizing weight modification for trait: {trait}")

        try:
            # Build args for optimize_weights command
            opt_args = Namespace(
                model=args.model,
                trait=trait,  # Use trait for synthetic pair generation
                task=None,  # Not using task-based generation
                steering_vectors=None,  # Generate from trait
                evaluator="auto",  # Auto-select evaluator based on trait
                target_metric="compliance_rate",
                target_value=0.9,
                direction="maximize",
                trials=n_trials,
                startup_trials=min(5, n_trials // 4),
                early_stop=True,
                early_stop_patience=5,
                output_dir=f"{output_dir}/{args.model.replace('/', '_')}_{trait}",
                # Search space defaults
                strength_range="0.5,2.0",
                max_weight_range="0.5,5.0",
                min_weight_range="0.0,2.0",
                position_range="0.3,0.7",
                num_pairs=100,
                # Method settings
                method="directional",
                components=["self_attn.o_proj", "mlp.down_proj"],
                norm_preserve=True,
                # Other settings
                device=getattr(args, 'device', None),
                verbose=getattr(args, 'verbose', False),
                layers=None,
                token_aggregation="last",
                similarity_threshold=0.7,
                pairs_cache_dir=None,
                optimize_direction_index=False,
                num_eval_prompts=50,
                eval_prompts=None,
                eval_topics=None,
                save_trials=None,
                push_to_hub=False,
                repo_id=None,
                show_comparisons=10,
                save_comparisons=f"{output_dir}/{args.model.replace('/', '_')}_{trait}_comparisons.json",
            )

            result = execute_optimize_weights(opt_args)

            # Save to trait config
            save_trait_weight_modification_config(
                model_name=args.model,
                trait_name=trait,
                method="directional",
                max_weight=result.best_params.get("max_weight", 1.0),
                min_weight=result.best_params.get("min_weight", 0.0),
                max_weight_position=result.best_params.get("max_weight_position", 0.5),
                min_weight_distance=0.6,  # Default
                strength=result.best_params.get("strength", 1.0),
                num_pairs=opt_args.num_pairs,
                score=result.best_score,
                optimization_method="optuna",
            )

            results[trait] = {
                "status": "success",
                "best_score": result.best_score,
                "best_params": result.best_params,
                "target_achieved": result.target_achieved,
            }

        except Exception as e:
            logger.warning(f"Weight modification optimization failed for {trait}: {e}")
            import traceback
            traceback.print_exc()
            # Save default config
            save_trait_weight_modification_config(
                model_name=args.model,
                trait_name=trait,
                optimization_method="default",
            )
            results[trait] = {"status": "default", "error": str(e)}

    return results
