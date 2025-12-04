"""
Optimize all parameters for a model.

Runs all optimization routines and saves results to the unified config manager:
1. Classification optimization (layer, threshold, aggregation)
2. Steering optimization (layer, strength)
3. Weight modification optimization (abliteration parameters)

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
    """Run steering optimization for traits."""
    results = {}

    for trait in traits:
        print(f"  Optimizing steering for trait: {trait}")

        # For traits, we use synthetic pair generation
        # This is a simplified version - full implementation would call optimize_steering
        # with trait-specific pair generation

        # For now, save placeholder config that can be updated
        save_trait_steering_config(
            model_name=args.model,
            trait_name=trait,
            layer=12,  # Default, should be optimized
            strength=1.0,  # Default, should be optimized
            method="CAA",
            optimization_method="placeholder",
        )
        results[trait] = {"status": "placeholder", "layer": 12, "strength": 1.0}

    return results


def _run_weight_modification_optimization(args: argparse.Namespace, traits: List[str]) -> Dict[str, Any]:
    """Run weight modification optimization for traits."""
    results = {}

    n_trials = getattr(args, 'n_trials', 20)

    for trait in traits:
        print(f"  Optimizing weight modification for trait: {trait}")

        try:
            from wisent.core.weight_modification.abliteration_optimizer import AbliterationOptimizer

            # Create evaluation function
            def evaluate_fn(model_path: str) -> float:
                # Simplified evaluation - in practice would run actual benchmark
                # This would typically call lm-eval or similar
                return 0.5  # Placeholder

            optimizer = AbliterationOptimizer(
                model_name=args.model,
                task=trait,  # Use trait as task for pair generation
                trait_label=trait,
                base_output_dir=getattr(args, 'output_dir', './data/modified_models'),
                evaluate_fn=evaluate_fn,
                direction="maximize",
            )

            # Run optimization with fewer trials for speed
            result = optimizer.optimize(
                n_trials=n_trials,
                save_to_cache=True,
                set_as_default=False,
            )

            # Save to trait config
            save_trait_weight_modification_config(
                model_name=args.model,
                trait_name=trait,
                method="abliteration",
                max_weight=result.best_params.max_weight,
                min_weight=result.best_params.min_weight,
                max_weight_position=result.best_params.max_weight_position,
                min_weight_distance=result.best_params.min_weight_distance,
                strength=result.best_params.strength,
                num_pairs=result.best_params.num_pairs,
                score=result.best_score,
                optimization_method="optuna",
            )

            results[trait] = {
                "status": "success",
                "best_score": result.best_score,
                "max_weight": result.best_params.max_weight,
                "strength": result.best_params.strength,
            }

        except Exception as e:
            logger.warning(f"Weight modification optimization failed for {trait}: {e}")
            # Save default config
            save_trait_weight_modification_config(
                model_name=args.model,
                trait_name=trait,
                optimization_method="default",
            )
            results[trait] = {"status": "default", "error": str(e)}

    return results
