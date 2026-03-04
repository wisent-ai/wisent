"""
Full model optimization command.

Usage:
    wisent optimize meta-llama/Llama-3.1-8B-Instruct

This command runs FULL optimization by orchestrating:
1. Classification optimization (via optimize_classification)
2. Steering optimization (via optimize_steering comprehensive)
3. Weight modification optimization (via optimize_weights)

Across:
- ALL available benchmarks (339+)
- Personalization traits
- Refusal/safety
- Humanization
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)



from wisent.core.utils.config_tools.constants import (
    DISPLAY_TRUNCATION_ERROR,
    DISPLAY_TOP_N_MINI,
    SEPARATOR_WIDTH_WIDE,
    SECONDS_PER_MINUTE,
    SECONDS_PER_HOUR,
    JSON_INDENT,
)
from wisent.core.utils.cli.optimization.core.optimize_helpers import (
    get_checkpoint_path, get_s3_checkpoint_key, load_checkpoint,
    save_checkpoint, get_all_benchmarks, get_personalization_traits,
    get_safety_traits, get_humanization_traits, get_welfare_traits,
)
from wisent.core.utils.cli.optimization.core.optimize_phase2a import run_benchmark_steering
from wisent.core.utils.cli.optimization.core.optimize_phase2b import run_safety_welfare_steering


def execute_optimize(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Execute FULL model optimization by orchestrating the individual commands.
    
    This runs:
    1. Classification optimization (optimize_classification)
    2. Steering optimization (optimize_steering comprehensive)
    3. Weight modification optimization (optimize_weights)
    
    Across ALL benchmarks and traits.
    """
    from wisent.core.utils.config_tools.config import store_optimization, get_cached_optimization
    
    start_time = time.time()
    
    print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
    print(f"FULL OPTIMIZATION: {args.model}")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}")
    
    # Determine what to optimize
    all_benchmarks = get_all_benchmarks()
    personalization_traits = get_personalization_traits()
    safety_traits = get_safety_traits()
    humanization_traits = get_humanization_traits()
    welfare_traits = get_welfare_traits()

    # Filter based on args
    if args.benchmarks:
        benchmarks = args.benchmarks
    else:
        benchmarks = all_benchmarks

    if args.skip_personalization:
        personalization_traits = []
    if args.skip_safety:
        safety_traits = []
    if args.skip_humanization:
        humanization_traits = []
    if getattr(args, 'skip_welfare', False):
        welfare_traits = []

    methods = args.methods if args.methods else ["CAA"]

    print(f"\n   Benchmarks: {len(benchmarks)}")
    print(f"   Personalization traits: {len(personalization_traits)}")
    print(f"   Safety traits: {len(safety_traits)}")
    print(f"   Humanization traits: {len(humanization_traits)}")
    print(f"   Welfare traits: {len(welfare_traits)}")
    print(f"   Steering methods: {', '.join(methods)}")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}\n")
    
    # Load checkpoint if resuming
    checkpoint = None
    if getattr(args, 'resume', True) and not getattr(args, 'force', False):
        checkpoint = load_checkpoint(args.model)
        if checkpoint:
            print(f"\n   Resuming from checkpoint (saved at {checkpoint.get('_checkpoint_time', 'unknown')})")
            print(f"      Phase: {checkpoint.get('_checkpoint_phase', 'unknown')}")
            print(f"      Completed steering: {len(checkpoint.get('steering', {}))}")
    
    results = checkpoint if checkpoint else {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "classification": {},
        "steering": {},
        "weights": {},
        "errors": [],
    }
    
    # =========================================================================
    # PHASE 1: CLASSIFICATION OPTIMIZATION
    # =========================================================================
    if not args.skip_classification:
        print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
        print(f"PHASE 1: CLASSIFICATION OPTIMIZATION")
        print(f"{'=' * SEPARATOR_WIDTH_WIDE}\n")
        
        try:
            from wisent.core.utils.cli.optimize_classification import execute_optimize_classification
            
            clf_args = argparse.Namespace(
                model=args.model,
                tasks=benchmarks[:args.optimization_benchmark_limit] if len(benchmarks) > args.optimization_benchmark_limit else benchmarks,
                limit=args.limit,
                device=args.device,
                verbose=getattr(args, 'verbose', False),
                layer_range=None,
                skip_full_search=False,
                aggregation_methods=['average', 'final', 'first', 'max'],
                threshold_range=list(args.optimize_classification_thresholds),
                optimization_metric='f1',
                save_plots=False,
            )
            
            results["classification"] = execute_optimize_classification(clf_args)
            print(f"\n   Classification optimization complete")
            save_checkpoint(args.model, results, phase="classification_complete")
        except Exception as e:
            error_msg = f"Classification optimization failed: {e}"
            results["errors"].append(error_msg)
            print(f"\n   {error_msg}")
            logger.exception("Classification optimization error")
            save_checkpoint(args.model, results, phase="classification_error")
    else:
        print(f"\n   Skipping classification optimization")
    
    # =========================================================================
    # PHASE 2: STEERING OPTIMIZATION (via optimize_steering comprehensive)
    # =========================================================================

    if not args.skip_steering:
        print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
        print(f"PHASE 2: STEERING OPTIMIZATION")
        print(f"{'=' * SEPARATOR_WIDTH_WIDE}\n")
        run_benchmark_steering(args, benchmarks, results)
        run_safety_welfare_steering(args, results)

    # =========================================================================
    # PHASE 3: WEIGHT MODIFICATION OPTIMIZATION
    # =========================================================================
    if not args.skip_weights:
        print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
        print(f"PHASE 3: WEIGHT MODIFICATION OPTIMIZATION")
        print(f"{'=' * SEPARATOR_WIDTH_WIDE}\n")
        
        try:
            from wisent.core.utils.cli.optimize_weights import execute_optimize_weights
            
            # Weight optimization for key traits
            weight_tasks = []
            if personalization_traits:
                weight_tasks.extend(personalization_traits[:DISPLAY_TOP_N_MINI])  # Use first 5 for practical runtime
            if safety_traits:
                weight_tasks.extend(safety_traits)
            if humanization_traits:
                weight_tasks.extend(humanization_traits)
            if welfare_traits:
                weight_tasks.extend(welfare_traits)
            
            for task_idx, task in enumerate(weight_tasks, 1):
                print(f"\n   [{task_idx}/{len(weight_tasks)}] {task}")
                
                try:
                    weight_args = argparse.Namespace(
                        model=args.model,
                        trait=task,
                        task=None,
                        steering_vectors=None,
                        trials=args.n_trials,
                        target_metric="score",
                        target_value=args.optimize_weight_mod_target,
                        device=args.device,
                        verbose=getattr(args, 'verbose', False),
                        output_dir=f"outputs/optimize/{args.model.replace('/', '_')}/weights_{task}",
                    )
                    
                    weight_result = execute_optimize_weights(weight_args)
                    results["weights"][task] = weight_result
                    print(f"       Optimized weight modification params")
                except Exception as e:
                    error_msg = f"weights:{task}: {str(e)}"
                    results["errors"].append(error_msg)
                    print(f"       Error: {str(e)[:DISPLAY_TRUNCATION_ERROR]}")
        except ImportError as e:
            print(f"   Weight optimization not available: {e}")
    else:
        print(f"\n   Skipping weight modification optimization")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - start_time
    
    print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
    print(f"FULL OPTIMIZATION COMPLETE")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}")
    print(f"   Model: {args.model}")
    print(f"   Total time: {total_time/SECONDS_PER_MINUTE:.1f} minutes ({total_time/SECONDS_PER_HOUR:.1f} hours)")
    print(f"   Steering configs optimized: {len(results['steering'])}")
    print(f"   Weight configs optimized: {len(results['weights'])}")
    print(f"   Errors: {len(results['errors'])}")
    
    # Count method wins
    method_wins = {}
    for task, result in results["steering"].items():
        if isinstance(result, dict) and "best_method" in result:
            method = result["best_method"]
            method_wins[method] = method_wins.get(method, 0) + 1
    
    if method_wins:
        print(f"\n   Method performance:")
        for method, wins in sorted(method_wins.items(), key=lambda x: -x[1]):
            print(f"      {method}: {wins} wins")
    
    # Save final checkpoint
    save_checkpoint(args.model, results, phase="complete")
    
    # Save results
    results_dir = Path("./optimization_results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"full_optimize_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=JSON_INDENT, default=str)
    
    print(f"\n   Results saved to: {results_file}")
    print(f"   Checkpoint: {get_checkpoint_path(args.model)}")
    print(f"   Config cache: ~/.wisent/configs/")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}\n")
    
    return results
