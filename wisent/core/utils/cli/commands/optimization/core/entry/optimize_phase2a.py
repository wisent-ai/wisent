"""Phase 2a-2b: Benchmark and personalization steering optimization."""
import argparse
import json
import os

from wisent.core.control.steering_methods.configs.optimal import get_optimal
from wisent.core.utils.cli.optimization.core.optimize_helpers import save_checkpoint
from wisent.core.utils.config_tools.constants import (
    DISPLAY_TRUNCATION_ERROR,
)


def run_benchmark_steering(args, benchmarks, results):
    """Run Phase 2a: benchmark steering and 2b: personalization steering.
    
    Modifies results dict in-place.
    """
    from wisent.core.utils.cli.optimize_steering import execute_optimize_steering

    
    for bench_idx, benchmark in enumerate(benchmarks, 1):
        # Skip if already in checkpoint results
        if benchmark in results.get("steering", {}):
            print(f"\n   [{bench_idx}/{len(benchmarks)}] {benchmark} - SKIPPED (in checkpoint)")
            continue
        
        # Skip if already optimized (unless --force)
        if not getattr(args, 'force', False):
            cached = get_cached_optimization(args.model, benchmark, method="*")
            if cached:
                print(f"\n   [{bench_idx}/{len(benchmarks)}] {benchmark} - SKIPPED (cached)")
                results["steering"][benchmark] = {
                    "best_method": cached.method,
                    "best_layer": cached.layer,
                    "best_score": cached.score,
                    "from_cache": True,
                }
                continue
        
        print(f"\n   [{bench_idx}/{len(benchmarks)}] {benchmark}")
        
        try:
            steering_args = argparse.Namespace(
                model=args.model,
                steering_action="comprehensive",
                tasks=[benchmark],
                methods=methods,
                device=args.device,
                use_cached=False,
                save_as_default=True,
                compute_baseline=True,
            )
            
            steering_result = execute_optimize_steering(steering_args)
            
            if steering_result and benchmark in steering_result:
                best_result = steering_result[benchmark]
                # Get best method result
                best_method = None
                best_score = -1
                for method, method_result in best_result.items():
                    if isinstance(method_result, dict) and method_result["best_score"] > best_score:
                        best_score = method_result["best_score"]
                        best_method = method
                        best_layer = method_result.get("best_layer")
                        best_strength = method_result.get("best_strength")
                        if best_layer is None:
                            raise ValueError(f"Missing 'best_layer' in result for {method}")
                        if best_strength is None:
                            raise ValueError(f"Missing 'best_strength' in result for {method}")
                
                if best_method:
                    store_optimization(
                        model=args.model,
                        task=benchmark,
                        layer=best_layer,
                        strength=best_strength,
                        method=best_method.upper(),
                        strategy=get_optimal("steering_strategy"),
                        score=best_score,
                        metric="accuracy",
                    )
                    
                    results["steering"][benchmark] = {
                        "best_method": best_method,
                        "best_layer": best_layer,
                        "best_strength": best_strength,
                        "best_score": best_score,
                    }
                    print(f"       Best: {best_method} @ layer {best_layer} = {best_score:.3f}")
            
        except Exception as e:
            error_msg = f"{benchmark}: {str(e)}"
            results["errors"].append(error_msg)
            print(f"       Error: {str(e)[:DISPLAY_TRUNCATION_ERROR]}")
            logger.exception(f"Error optimizing {benchmark}")
        
        save_checkpoint(args.model, results, phase=f"steering_benchmark_{bench_idx}")
    
    # 2b. Personalization trait steering
    if personalization_traits:
        print(f"\n   Optimizing steering for {len(personalization_traits)} personalization traits...")
        
        for trait_idx, trait in enumerate(personalization_traits, 1):
            task_key = f"trait:{trait}"
            
            if task_key in results.get("steering", {}):
                print(f"\n   [{trait_idx}/{len(personalization_traits)}] {trait} - SKIPPED (in checkpoint)")
                continue
            
            if not getattr(args, 'force', False):
                cached = get_cached_optimization(args.model, f"personalization:{trait}", method="*")
                if cached:
                    print(f"\n   [{trait_idx}/{len(personalization_traits)}] {trait} - SKIPPED (cached)")
                    results["steering"][task_key] = {
                        "best_method": cached.method,
                        "best_layer": cached.layer,
                        "best_score": cached.score,
                        "from_cache": True,
                    }
                    continue
            
            print(f"\n   [{trait_idx}/{len(personalization_traits)}] {trait}")
            
            try:
                steering_args = argparse.Namespace(
                    model=args.model,
                    steering_action="personalization",
                    trait=trait,
                    methods=methods,
                    device=args.device,
                )
                
                steering_result = execute_optimize_steering(steering_args)
                
                if steering_result:
                    best_method = steering_result.get("best_method", "CAA")
                    best_layer = steering_result.get("best_layer")
                    if best_layer is None:
                        raise ValueError(f"Missing 'best_layer' in personalization result for {trait}")
                    best_score = steering_result["best_score"]
                    
                    store_optimization(
                        model=args.model,
                        task=f"personalization:{trait}",
                        layer=best_layer,
                        strength=steering_result["best_strength"],
                        method=best_method.upper(),
                        score=best_score,
                    )
                    
                    results["steering"][task_key] = {
                        "best_method": best_method,
                        "best_layer": best_layer,
                        "best_score": best_score,
                    }
                    print(f"       Best: {best_method} @ layer {best_layer} = {best_score:.3f}")
                
            except Exception as e:
                error_msg = f"trait:{trait}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"       Error: {str(e)[:DISPLAY_TRUNCATION_ERROR]}")

            save_checkpoint(args.model, results, phase=f"steering_personalization_{trait_idx}")
    
