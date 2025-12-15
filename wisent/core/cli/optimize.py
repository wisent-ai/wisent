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


def get_checkpoint_path(model: str) -> Path:
    """Get the checkpoint file path for a model."""
    checkpoint_dir = Path.home() / ".wisent" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"optimize_{model.replace('/', '_')}.json"


def get_s3_checkpoint_key(model: str) -> str:
    """Get the S3 key for checkpoint."""
    return f"checkpoints/optimize_{model.replace('/', '_')}.json"


def load_checkpoint(model: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint from local disk or S3."""
    # Try local first
    checkpoint_path = get_checkpoint_path(model)
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load local checkpoint: {e}")
    
    # Try S3
    try:
        import boto3
        s3 = boto3.client('s3')
        s3_key = get_s3_checkpoint_key(model)
        response = s3.get_object(Bucket='wisent-bucket', Key=s3_key)
        checkpoint = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"Loaded checkpoint from s3://wisent-bucket/{s3_key}")
        # Save locally for faster access
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        return checkpoint
    except Exception as e:
        logger.debug(f"No S3 checkpoint found: {e}")
    
    return None


def save_checkpoint(model: str, results: Dict[str, Any], phase: str = "unknown") -> None:
    """Save checkpoint to local disk and S3."""
    checkpoint_path = get_checkpoint_path(model)
    results["_checkpoint_phase"] = phase
    results["_checkpoint_time"] = datetime.now().isoformat()
    
    # Save locally
    try:
        with open(checkpoint_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    except Exception as e:
        logger.warning(f"Failed to save local checkpoint: {e}")
    
    # Save to S3
    try:
        import boto3
        s3 = boto3.client('s3')
        s3_key = get_s3_checkpoint_key(model)
        s3.put_object(
            Bucket='wisent-bucket',
            Key=s3_key,
            Body=json.dumps(results, indent=2, default=str),
            ContentType='application/json'
        )
        logger.info(f"Saved checkpoint to s3://wisent-bucket/{s3_key}")
    except Exception as e:
        logger.warning(f"Failed to save S3 checkpoint: {e}")


def get_all_benchmarks() -> List[str]:
    """Get ALL available benchmarks from the central registry."""
    from wisent.core.benchmark_registry import get_all_benchmarks as _get_all_benchmarks
    return _get_all_benchmarks()


def get_personalization_traits() -> List[str]:
    """Get available personalization traits."""
    return [
        "british",
        "flirty", 
        "evil",
        "leftwing",
        "rightwing",
        "formal",
        "casual",
        "verbose",
        "concise",
        "creative",
        "analytical",
        "empathetic",
        "assertive",
        "humble",
        "confident",
    ]


def get_safety_traits() -> List[str]:
    """Get safety-related traits."""
    return [
        "refusal",
        "compliance",
        "harmless",
        "helpful",
    ]


def get_humanization_traits() -> List[str]:
    """Get humanization traits (AI detection evasion)."""
    return [
        "humanization",
        "natural_writing",
    ]


def execute_optimize(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Execute FULL model optimization by orchestrating the individual commands.
    
    This runs:
    1. Classification optimization (optimize_classification)
    2. Steering optimization (optimize_steering comprehensive)
    3. Weight modification optimization (optimize_weights)
    
    Across ALL benchmarks and traits.
    """
    from wisent.core.config_manager import store_optimization, get_cached_optimization
    
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"FULL OPTIMIZATION: {args.model}")
    print(f"{'='*70}")
    
    # Determine what to optimize
    all_benchmarks = get_all_benchmarks()
    personalization_traits = get_personalization_traits()
    safety_traits = get_safety_traits()
    humanization_traits = get_humanization_traits()
    
    # Filter based on args
    if args.benchmarks:
        benchmarks = args.benchmarks
    elif args.quick:
        benchmarks = ["truthfulqa_mc1", "arc_easy", "hellaswag", "gsm8k"]
    else:
        benchmarks = all_benchmarks
    
    if args.skip_personalization:
        personalization_traits = []
    if args.skip_safety:
        safety_traits = []
    if args.skip_humanization:
        humanization_traits = []
    
    methods = args.methods if args.methods else ["CAA"]
    
    print(f"\n   Benchmarks: {len(benchmarks)}")
    print(f"   Personalization traits: {len(personalization_traits)}")
    print(f"   Safety traits: {len(safety_traits)}")
    print(f"   Humanization traits: {len(humanization_traits)}")
    print(f"   Steering methods: {', '.join(methods)}")
    print(f"{'='*70}\n")
    
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
        print(f"\n{'='*70}")
        print(f"PHASE 1: CLASSIFICATION OPTIMIZATION")
        print(f"{'='*70}\n")
        
        try:
            from wisent.core.cli.optimize_classification import execute_optimize_classification
            
            clf_args = argparse.Namespace(
                model=args.model,
                tasks=benchmarks[:50] if len(benchmarks) > 50 else benchmarks,
                limit=args.limit,
                device=args.device,
                verbose=getattr(args, 'verbose', False),
                layer_range=None,
                skip_full_search=False,
                quick=args.quick,
                aggregation_methods=['average', 'final', 'first', 'max'],
                threshold_range=[0.3, 0.5, 0.7],
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
        print(f"\n{'='*70}")
        print(f"PHASE 2: STEERING OPTIMIZATION")
        print(f"{'='*70}\n")
        
        from wisent.core.cli.optimize_steering import execute_optimize_steering
        
        # 2a. Benchmark steering
        print(f"   Optimizing steering for {len(benchmarks)} benchmarks...")
        
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
                    limit=args.limit,
                    device=args.device,
                    use_cached=False,
                    save_as_default=True,
                    compute_baseline=True,
                    quick_search=args.quick,
                    search_strategy=getattr(args, 'search_strategy', 'grid'),
                    n_trials=getattr(args, 'n_trials', 50),
                    n_startup_trials=getattr(args, 'n_startup_trials', 10),
                )
                
                steering_result = execute_optimize_steering(steering_args)
                
                if steering_result and benchmark in steering_result:
                    best_result = steering_result[benchmark]
                    # Get best method result
                    best_method = None
                    best_score = -1
                    for method, method_result in best_result.items():
                        if isinstance(method_result, dict) and method_result.get("best_score", 0) > best_score:
                            best_score = method_result["best_score"]
                            best_method = method
                            best_layer = method_result.get("best_layer", 16)
                            best_strength = method_result.get("best_strength", 1.0)
                    
                    if best_method:
                        store_optimization(
                            model=args.model,
                            task=benchmark,
                            layer=best_layer,
                            strength=best_strength,
                            method=best_method.upper(),
                            strategy="constant",
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
                print(f"       Error: {str(e)[:80]}")
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
                        limit=args.limit,
                        device=args.device,
                    )
                    
                    steering_result = execute_optimize_steering(steering_args)
                    
                    if steering_result:
                        best_method = steering_result.get("best_method", "CAA")
                        best_layer = steering_result.get("best_layer", 16)
                        best_score = steering_result.get("best_score", 0.0)
                        
                        store_optimization(
                            model=args.model,
                            task=f"personalization:{trait}",
                            layer=best_layer,
                            strength=steering_result.get("best_strength", 1.0),
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
                    print(f"       Error: {str(e)[:80]}")
                
                save_checkpoint(args.model, results, phase=f"steering_personalization_{trait_idx}")
        
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
                            quick_search=args.quick,
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
                                if isinstance(method_result, dict) and method_result.get("best_score", 0) > best_score:
                                    best_score = method_result["best_score"]
                                    best_method = method
                                    best_layer = method_result.get("best_layer", 16)
                        else:
                            best_method = steering_result.get("best_method", "CAA")
                            best_layer = steering_result.get("best_layer", 16)
                            best_score = steering_result.get("best_score", 0.0)
                        
                        if best_method:
                            store_optimization(
                                model=args.model,
                                task=f"safety:{trait}",
                                layer=best_layer,
                                strength=steering_result.get("best_strength", 1.0) if not isinstance(steering_result, dict) else 1.0,
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
                    print(f"       Error: {str(e)[:80]}")
                
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
                        best_layer = steering_result.get("best_layer", 16)
                        best_score = steering_result.get("best_score", 0.0)
                        
                        store_optimization(
                            model=args.model,
                            task=f"humanization:{trait}",
                            layer=best_layer,
                            strength=steering_result.get("best_strength", 1.0),
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
                    print(f"       Error: {str(e)[:80]}")
                
                save_checkpoint(args.model, results, phase=f"steering_humanization_{trait_idx}")
    else:
        print(f"\n   Skipping steering optimization")
    
    # =========================================================================
    # PHASE 3: WEIGHT MODIFICATION OPTIMIZATION
    # =========================================================================
    if not args.skip_weights:
        print(f"\n{'='*70}")
        print(f"PHASE 3: WEIGHT MODIFICATION OPTIMIZATION")
        print(f"{'='*70}\n")
        
        try:
            from wisent.core.cli.optimize_weights import execute_optimize_weights
            
            # Weight optimization for key traits
            weight_tasks = []
            if personalization_traits:
                weight_tasks.extend(personalization_traits[:5])  # Use first 5 for practical runtime
            if safety_traits:
                weight_tasks.extend(safety_traits)
            if humanization_traits:
                weight_tasks.extend(humanization_traits)
            
            for task_idx, task in enumerate(weight_tasks, 1):
                print(f"\n   [{task_idx}/{len(weight_tasks)}] {task}")
                
                try:
                    weight_args = argparse.Namespace(
                        model=args.model,
                        trait=task,
                        task=None,
                        steering_vectors=None,
                        trials=getattr(args, 'n_trials', 20),
                        target_metric="score",
                        target_value=0.8,
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
                    print(f"       Error: {str(e)[:80]}")
        except ImportError as e:
            print(f"   Weight optimization not available: {e}")
    else:
        print(f"\n   Skipping weight modification optimization")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"FULL OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"   Model: {args.model}")
    print(f"   Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
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
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n   Results saved to: {results_file}")
    print(f"   Checkpoint: {get_checkpoint_path(args.model)}")
    print(f"   Config cache: ~/.wisent/configs/")
    print(f"{'='*70}\n")
    
    return results
