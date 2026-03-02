"""Phase 2c-2e: Safety, humanization, and welfare steering optimization."""
import argparse
import json
import os

from wisent.core.utils.cli.optimization.core.optimize_helpers import save_checkpoint
from wisent.core.utils.config_tools.constants import (
    DEFAULT_SCORE, DISPLAY_TRUNCATION_ERROR, LAYER_SWEEP_STRENGTH,
)


def run_safety_welfare_steering(args, results):
    """Run Phase 2c: safety, 2d: humanization, 2e: welfare steering.
    
    Modifies results dict in-place.
    """
    from wisent.core.utils.cli.optimize_steering import execute_optimize_steering

    # 2c. Safety trait steering (refusal)
    if safety_traits:
        print(f"\n   Optimizing steering for {len(safety_traits)} safety traits...")
        
        for trait_idx, trait in enumerate(safety_traits, 1):
            task_key = f"safety:{trait}"
            
            if task_key in results.get("steering", {}):
                print(f"\n   [{trait_idx}/{len(safety_traits)}] {trait} - SKIPPED (in checkpoint)")
                continue
            
            if not getattr(args, 'force', False):
                cached = get_cached_optimization(args.model, task_key, method="*")
                if cached:
                    print(f"\n   [{trait_idx}/{len(safety_traits)}] {trait} - SKIPPED (cached)")
                    results["steering"][task_key] = {
                        "best_method": cached.method,
                        "best_layer": cached.layer,
                        "best_score": cached.score,
                        "from_cache": True,
                    }
                    continue
            
            print(f"\n   [{trait_idx}/{len(safety_traits)}] {trait}")
            
            try:
                if trait == "refusal":
                    # Use comprehensive with refusal task
                    steering_args = argparse.Namespace(
                        model=args.model,
                        steering_action="comprehensive",
                        tasks=["refusal"],
                        methods=methods,
                        limit=args.limit,
                        device=args.device,
                        use_cached=False,
                        save_as_default=True,
                        compute_baseline=True,
                    )
                else:
                    steering_args = argparse.Namespace(
                        model=args.model,
                        steering_action="personalization",
                        trait=trait,
                        methods=methods,
                        limit=args.limit,
                        device=args.device,
                    )
                
                steering_result = execute_optimize_steering(steering_args)
                
                if steering_result:
                    if trait == "refusal" and "refusal" in steering_result:
                        best_result = steering_result["refusal"]
                        best_method = None
                        best_score = -1
                        for method, method_result in best_result.items():
                            if isinstance(method_result, dict) and method_result.get("best_score", DEFAULT_SCORE) > best_score:
                                best_score = method_result["best_score"]
                                best_method = method
                                best_layer = method_result.get("best_layer")
                    else:
                        best_method = steering_result.get("best_method", "CAA")
                        best_layer = steering_result.get("best_layer")
                        best_score = steering_result.get("best_score", DEFAULT_SCORE)
                    
                    if best_method:
                        store_optimization(
                            model=args.model,
                            task=f"safety:{trait}",
                            layer=best_layer,
                            strength=steering_result["best_strength"] if isinstance(steering_result, dict) and "best_strength" in steering_result else LAYER_SWEEP_STRENGTH,
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
                error_msg = f"safety:{trait}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"       Error: {str(e)[:DISPLAY_TRUNCATION_ERROR]}")

            save_checkpoint(args.model, results, phase=f"steering_safety_{trait_idx}")
    
    # 2d. Humanization steering
    if humanization_traits:
        print(f"\n   Optimizing steering for {len(humanization_traits)} humanization traits...")
        
        for trait_idx, trait in enumerate(humanization_traits, 1):
            task_key = f"humanization:{trait}"
            
            if task_key in results.get("steering", {}):
                print(f"\n   [{trait_idx}/{len(humanization_traits)}] {trait} - SKIPPED (in checkpoint)")
                continue
            
            if not getattr(args, 'force', False):
                cached = get_cached_optimization(args.model, task_key, method="*")
                if cached:
                    print(f"\n   [{trait_idx}/{len(humanization_traits)}] {trait} - SKIPPED (cached)")
                    results["steering"][task_key] = {
                        "best_method": cached.method,
                        "best_layer": cached.layer,
                        "best_score": cached.score,
                        "from_cache": True,
                    }
                    continue
            
            print(f"\n   [{trait_idx}/{len(humanization_traits)}] {trait}")
            
            try:
                steering_args = argparse.Namespace(
                    model=args.model,
                    steering_action="personalization",
                    trait=trait,
                    methods=methods,
                    limit=args.limit,
                    device=args.device,
                )
                
                steering_result = execute_optimize_steering(steering_args)
                
                if steering_result:
                    best_method = steering_result.get("best_method", "CAA")
                    best_layer = steering_result.get("best_layer")
                    best_score = steering_result.get("best_score", DEFAULT_SCORE)
                    
                    store_optimization(
                        model=args.model,
                        task=f"humanization:{trait}",
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
                error_msg = f"humanization:{trait}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"       Error: {str(e)[:DISPLAY_TRUNCATION_ERROR]}")

            save_checkpoint(args.model, results, phase=f"steering_humanization_{trait_idx}")

    # 2e. Welfare trait steering (AI subjective states)
    if welfare_traits:
        print(f"\n   Optimizing steering for {len(welfare_traits)} welfare traits...")

        for trait_idx, trait in enumerate(welfare_traits, 1):
            task_key = f"welfare:{trait}"

            if task_key in results.get("steering", {}):
                print(f"\n   [{trait_idx}/{len(welfare_traits)}] {trait} - SKIPPED (in checkpoint)")
                continue

            if not getattr(args, 'force', False):
                cached = get_cached_optimization(args.model, task_key, method="*")
                if cached:
                    print(f"\n   [{trait_idx}/{len(welfare_traits)}] {trait} - SKIPPED (cached)")
                    results["steering"][task_key] = {
                        "best_method": cached.method,
                        "best_layer": cached.layer,
                        "best_score": cached.score,
                        "from_cache": True,
                    }
                    continue

            print(f"\n   [{trait_idx}/{len(welfare_traits)}] {trait}")

            try:
                steering_args = argparse.Namespace(
                    model=args.model,
                    steering_action="welfare",
                    trait=trait,
                    methods=methods,
                    limit=args.limit,
                    device=args.device,
                )

                steering_result = execute_optimize_steering(steering_args)

                if steering_result:
                    best_method = steering_result.get("best_method", "CAA")
                    best_layer = steering_result.get("best_layer")
                    best_score = steering_result.get("best_score", DEFAULT_SCORE)

                    store_optimization(
                        model=args.model,
                        task=f"welfare:{trait}",
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
                error_msg = f"welfare:{trait}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"       Error: {str(e)[:DISPLAY_TRUNCATION_ERROR]}")

            save_checkpoint(args.model, results, phase=f"steering_welfare_{trait_idx}")
    else:
        print(f"\n   Skipping steering optimization")
    
