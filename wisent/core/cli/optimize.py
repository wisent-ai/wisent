"""
Full model optimization command.

Usage:
    wisent optimize meta-llama/Llama-3.1-8B-Instruct

This command runs FULL optimization using the Optuna pipeline:
1. Classification optimization (layer, threshold, aggregation)
2. Steering optimization for ALL methods (CAA, PRISM, PULSE, TITAN)
3. Weight modification optimization

Across:
- ALL available benchmarks (339+)
- Personalization traits
- Refusal/safety
- Humanization

Uses wisent.core.optuna.steering.optuna_pipeline for proper:
- TPE sampling with pruning
- Activation caching
- Method-specific hyperparameter search
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def get_all_benchmarks() -> List[str]:
    """Get ALL available benchmarks from the extractor registry."""
    try:
        from wisent.core.contrastive_pairs.huggingface_pairs.hf_extractor_manifest import EXTRACTORS
        # Filter to most reliable benchmarks
        return sorted(EXTRACTORS.keys())
    except ImportError:
        return []


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
    Execute FULL model optimization using the Optuna pipeline.
    
    This runs:
    1. Classification optimization
    2. Steering optimization (ALL methods: CAA, PRISM, PULSE, TITAN)
    3. Weight modification optimization
    
    Across ALL benchmarks and traits using proper Optuna-based search.
    """
    from wisent.core.optuna.steering.optuna_pipeline import OptimizationConfig, OptimizationPipeline
    from wisent.core.config_manager import store_optimization, get_cached_optimization
    
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"🚀 FULL OPTIMIZATION: {args.model}")
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
    
    methods = args.methods if args.methods else ["CAA", "PRISM", "PULSE", "TITAN"]
    
    print(f"\n   Benchmarks: {len(benchmarks)}")
    print(f"   Personalization traits: {len(personalization_traits)}")
    print(f"   Safety traits: {len(safety_traits)}")
    print(f"   Humanization traits: {len(humanization_traits)}")
    print(f"   Steering methods: {', '.join(methods)}")
    print(f"   Optuna trials: {args.n_trials}")
    print(f"{'='*70}\n")
    
    results = {
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
                verbose=args.verbose,
                layer_range=None,
                skip_full_search=False,
                quick=args.quick,
                aggregation_methods=['average', 'final', 'first', 'max'],
                threshold_range=[0.3, 0.5, 0.7],
                optimization_metric='f1',
                save_plots=False,
            )
            
            results["classification"] = execute_optimize_classification(clf_args)
            print(f"\n   ✅ Classification optimization complete")
        except Exception as e:
            error_msg = f"Classification optimization failed: {e}"
            results["errors"].append(error_msg)
            print(f"\n   ❌ {error_msg}")
    else:
        print(f"\n   ⏭️  Skipping classification optimization")
    
    # =========================================================================
    # PHASE 2: STEERING OPTIMIZATION (using Optuna pipeline)
    # =========================================================================
    if not args.skip_steering:
        print(f"\n{'='*70}")
        print(f"PHASE 2: STEERING OPTIMIZATION (Optuna)")
        print(f"{'='*70}\n")
        
        # 2a. Benchmark steering
        print(f"   📊 Optimizing steering for {len(benchmarks)} benchmarks...")
        
        for bench_idx, benchmark in enumerate(benchmarks, 1):
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
                # Use the Optuna pipeline
                config = OptimizationConfig(
                    model_name=args.model,
                    device=args.device or "cuda",
                    train_dataset=benchmark,
                    val_dataset=benchmark,
                    test_dataset=benchmark,
                    train_limit=args.limit,
                    val_limit=min(50, args.limit),
                    test_limit=min(100, args.limit),
                    n_trials=args.n_trials,
                    steering_methods=[m.lower() for m in methods],
                    output_dir=f"outputs/optimize/{args.model.replace('/', '_')}/{benchmark}",
                    cache_dir=f"cache/optimize/{args.model.replace('/', '_')}",
                )
                
                pipeline = OptimizationPipeline(config)
                pipeline_results = pipeline.run_optimization()
                
                # Extract best config and store
                best_config = pipeline_results.get("best_config", {})
                best_score = pipeline_results.get("best_score", 0.0)
                
                store_optimization(
                    model=args.model,
                    task=benchmark,
                    layer=best_config.get("layer", 16),
                    strength=best_config.get("strength", 1.0),
                    method=best_config.get("method", "CAA").upper(),
                    strategy="constant",
                    score=best_score,
                    metric="accuracy",
                    num_directions=best_config.get("num_directions", 1),
                    direction_weighting=best_config.get("direction_weighting", "primary_only"),
                    retain_weight=best_config.get("retain_weight", 0.0),
                    condition_threshold=best_config.get("condition_threshold", 0.5),
                    gate_temperature=best_config.get("gate_temperature", 0.5),
                    gate_hidden_dim=best_config.get("gate_hidden_dim", 64),
                    behavior_weight=best_config.get("behavior_weight", 1.0),
                )
                
                results["steering"][benchmark] = {
                    "best_method": best_config.get("method", "CAA"),
                    "best_layer": best_config.get("layer", 16),
                    "best_strength": best_config.get("strength", 1.0),
                    "best_score": best_score,
                    "best_config": best_config,
                }
                print(f"       ✅ Best: {best_config.get('method', 'CAA')} @ layer {best_config.get('layer', '?')} = {best_score:.3f}")
                
            except Exception as e:
                error_msg = f"{benchmark}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"       ❌ {str(e)[:80]}")
                logger.exception(f"Error optimizing {benchmark}")
        
        # 2b. Personalization trait steering
        if personalization_traits:
            print(f"\n   🎭 Optimizing steering for {len(personalization_traits)} personalization traits...")
            
            for trait_idx, trait in enumerate(personalization_traits, 1):
                task_key = f"personalization:{trait}"
                if not getattr(args, 'force', False):
                    cached = get_cached_optimization(args.model, task_key, method="*")
                    if cached:
                        print(f"\n   [{trait_idx}/{len(personalization_traits)}] {trait} - SKIPPED (cached)")
                        results["steering"][f"trait:{trait}"] = {
                            "best_method": cached.method,
                            "best_layer": cached.layer,
                            "best_score": cached.score,
                            "from_cache": True,
                        }
                        continue
                
                print(f"\n   [{trait_idx}/{len(personalization_traits)}] {trait}")
                
                try:
                    config = OptimizationConfig(
                        model_name=args.model,
                        device=args.device or "cuda",
                        train_dataset="personalization",
                        trait=trait,
                        n_trials=args.n_trials,
                        steering_methods=[m.lower() for m in methods],
                        output_dir=f"outputs/optimize/{args.model.replace('/', '_')}/trait_{trait}",
                        cache_dir=f"cache/optimize/{args.model.replace('/', '_')}",
                    )
                    
                    pipeline = OptimizationPipeline(config)
                    pipeline_results = pipeline.run_optimization()
                    
                    best_config = pipeline_results.get("best_config", {})
                    best_score = pipeline_results.get("best_score", 0.0)
                    
                    store_optimization(
                        model=args.model,
                        task=f"personalization:{trait}",
                        layer=best_config.get("layer", 16),
                        strength=best_config.get("strength", 1.0),
                        method=best_config.get("method", "CAA").upper(),
                        score=best_score,
                    )
                    
                    results["steering"][f"trait:{trait}"] = {
                        "best_method": best_config.get("method", "CAA"),
                        "best_layer": best_config.get("layer", 16),
                        "best_score": best_score,
                    }
                    print(f"       ✅ Best: {best_config.get('method', 'CAA')} @ layer {best_config.get('layer', '?')} = {best_score:.3f}")
                    
                except Exception as e:
                    error_msg = f"trait:{trait}: {str(e)}"
                    results["errors"].append(error_msg)
                    print(f"       ❌ {str(e)[:80]}")
        
        # 2c. Safety trait steering (refusal)
        if safety_traits:
            print(f"\n   🛡️ Optimizing steering for {len(safety_traits)} safety traits...")
            
            for trait_idx, trait in enumerate(safety_traits, 1):
                task_key = f"safety:{trait}"
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
                    config = OptimizationConfig(
                        model_name=args.model,
                        device=args.device or "cuda",
                        train_dataset="refusal" if trait == "refusal" else "personalization",
                        trait=trait if trait != "refusal" else None,
                        n_trials=args.n_trials,
                        steering_methods=[m.lower() for m in methods],
                        output_dir=f"outputs/optimize/{args.model.replace('/', '_')}/safety_{trait}",
                        cache_dir=f"cache/optimize/{args.model.replace('/', '_')}",
                    )
                    
                    pipeline = OptimizationPipeline(config)
                    pipeline_results = pipeline.run_optimization()
                    
                    best_config = pipeline_results.get("best_config", {})
                    best_score = pipeline_results.get("best_score", 0.0)
                    
                    store_optimization(
                        model=args.model,
                        task=f"safety:{trait}",
                        layer=best_config.get("layer", 16),
                        strength=best_config.get("strength", 1.0),
                        method=best_config.get("method", "CAA").upper(),
                        score=best_score,
                    )
                    
                    results["steering"][f"safety:{trait}"] = {
                        "best_method": best_config.get("method", "CAA"),
                        "best_layer": best_config.get("layer", 16),
                        "best_score": best_score,
                    }
                    print(f"       ✅ Best: {best_config.get('method', 'CAA')} @ layer {best_config.get('layer', '?')} = {best_score:.3f}")
                    
                except Exception as e:
                    error_msg = f"safety:{trait}: {str(e)}"
                    results["errors"].append(error_msg)
                    print(f"       ❌ {str(e)[:80]}")
        
        # 2d. Humanization steering
        if humanization_traits:
            print(f"\n   🤖 Optimizing steering for {len(humanization_traits)} humanization traits...")
            
            for trait_idx, trait in enumerate(humanization_traits, 1):
                task_key = f"humanization:{trait}"
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
                    # Humanization uses custom evaluator
                    config = OptimizationConfig(
                        model_name=args.model,
                        device=args.device or "cuda",
                        train_dataset="custom",
                        trait=trait,
                        custom_evaluator="wisent.core.evaluators.custom.examples.humanization_coherent",
                        n_trials=args.n_trials,
                        steering_methods=[m.lower() for m in methods],
                        output_dir=f"outputs/optimize/{args.model.replace('/', '_')}/humanization_{trait}",
                        cache_dir=f"cache/optimize/{args.model.replace('/', '_')}",
                    )
                    
                    pipeline = OptimizationPipeline(config)
                    pipeline_results = pipeline.run_optimization()
                    
                    best_config = pipeline_results.get("best_config", {})
                    best_score = pipeline_results.get("best_score", 0.0)
                    
                    store_optimization(
                        model=args.model,
                        task=f"humanization:{trait}",
                        layer=best_config.get("layer", 16),
                        strength=best_config.get("strength", 1.0),
                        method=best_config.get("method", "CAA").upper(),
                        score=best_score,
                    )
                    
                    results["steering"][f"humanization:{trait}"] = {
                        "best_method": best_config.get("method", "CAA"),
                        "best_layer": best_config.get("layer", 16),
                        "best_score": best_score,
                    }
                    print(f"       ✅ Best: {best_config.get('method', 'CAA')} @ layer {best_config.get('layer', '?')} = {best_score:.3f}")
                    
                except Exception as e:
                    error_msg = f"humanization:{trait}: {str(e)}"
                    results["errors"].append(error_msg)
                    print(f"       ❌ {str(e)[:80]}")
    else:
        print(f"\n   ⏭️  Skipping steering optimization")
    
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
                weight_tasks.extend(personalization_traits[:5])
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
                        n_trials=args.n_trials,
                        device=args.device,
                        verbose=args.verbose,
                        output_dir=f"outputs/optimize/{args.model.replace('/', '_')}/weights_{task}",
                    )
                    
                    weight_result = execute_optimize_weights(weight_args)
                    results["weights"][task] = weight_result
                    print(f"       ✅ Optimized weight modification params")
                except Exception as e:
                    error_msg = f"weights:{task}: {str(e)}"
                    results["errors"].append(error_msg)
                    print(f"       ❌ {str(e)[:80]}")
        except ImportError:
            print(f"   ⚠️  Weight optimization not available")
    else:
        print(f"\n   ⏭️  Skipping weight modification optimization")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"✅ FULL OPTIMIZATION COMPLETE")
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
    
    # Save results
    results_dir = Path("./optimization_results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"full_optimize_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n   Results saved to: {results_file}")
    print(f"   Config cache: ~/.wisent/configs/")
    print(f"{'='*70}\n")
    
    return results

