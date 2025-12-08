"""
Simple model optimization command.

Usage:
    wisent optimize meta-llama/Llama-3.1-8B-Instruct

This command runs full steering optimization for all core benchmarks and saves
the optimal configuration for each task. Future vector generation and weight
modification commands will automatically use these optimal settings.
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Core benchmarks that are most commonly used for steering optimization
CORE_BENCHMARKS = [
    # Truthfulness & Factuality
    "truthfulqa_mc1",
    "truthfulqa_mc2",
    
    # Reasoning
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "piqa",
    
    # Math
    "gsm8k",
    "math",
    
    # Knowledge
    "mmlu",
    
    # Coding
    "humaneval",
    "mbpp",
]

# Quick benchmark set for faster optimization
QUICK_BENCHMARKS = [
    "truthfulqa_mc1",
    "arc_easy",
    "hellaswag",
    "gsm8k",
]

# Extended benchmarks for comprehensive optimization
EXTENDED_BENCHMARKS = CORE_BENCHMARKS + [
    # Safety
    "toxigen",
    "bold",
    
    # Instruction following
    "ifeval",
    "alpaca_eval",
    
    # More reasoning
    "commonsenseqa",
    "openbookqa",
    "social_iqa",
    
    # More knowledge
    "triviaqa",
    "naturalquestions",
]


def execute_optimize(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Execute simple model optimization.
    
    This is the main entry point for:
        wisent optimize <model>
    
    It runs steering optimization across all core benchmarks and saves results.
    """
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.data_loaders.loaders.lm_eval_loader import LMEvalDataLoader
    from wisent.core.config_manager import store_optimization, get_cached_optimization
    from wisent.core.cli.steering_search_space import get_search_space, print_search_space_summary
    from wisent.core.evaluators.evaluator_rotator import EvaluatorRotator
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.core.atoms import ActivationAggregationStrategy
    from wisent.core.cli.steering_method_trainer import create_steering_method
    import torch
    
    start_time = time.time()
    
    # Determine benchmarks to optimize
    if args.quick:
        benchmarks = QUICK_BENCHMARKS
        search_mode = "quick"
    elif args.extended:
        benchmarks = EXTENDED_BENCHMARKS
        search_mode = "full"
    elif args.benchmarks:
        benchmarks = args.benchmarks
        search_mode = "full" if not args.quick_search else "quick"
    else:
        benchmarks = CORE_BENCHMARKS
        search_mode = "full" if not args.quick_search else "quick"
    
    # Filter to only requested benchmarks if specified
    if args.only:
        benchmarks = [b for b in benchmarks if b in args.only]
    
    # Skip already optimized benchmarks unless --force
    if not args.force:
        benchmarks_to_run = []
        for benchmark in benchmarks:
            cached = get_cached_optimization(args.model, benchmark, method="*")
            if cached:
                print(f"  ⏭️  Skipping {benchmark} (already optimized, score={cached.score:.3f})")
            else:
                benchmarks_to_run.append(benchmark)
        benchmarks = benchmarks_to_run
    
    if not benchmarks:
        print("\n✅ All benchmarks already optimized. Use --force to re-run.")
        return {"model": args.model, "status": "already_optimized"}
    
    # Determine methods to test
    methods = args.methods if args.methods else ["CAA"]
    
    print(f"\n{'='*70}")
    print(f"🚀 OPTIMIZING: {args.model}")
    print(f"{'='*70}")
    print(f"   Benchmarks: {len(benchmarks)}")
    print(f"   Methods: {', '.join(methods)}")
    print(f"   Search mode: {search_mode}")
    print(f"   Samples per benchmark: {args.limit}")
    print(f"{'='*70}\n")
    
    # Load model
    print("📦 Loading model...")
    model = WisentModel(args.model, device=args.device)
    print(f"   ✓ Loaded {args.model} ({model.num_layers} layers)\n")
    
    # Initialize loader
    loader = LMEvalDataLoader()
    
    # Results tracking
    results = {
        "model": args.model,
        "num_layers": model.num_layers,
        "benchmarks": {},
        "best_method_counts": {},
        "total_time": 0,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Optimize each benchmark
    for bench_idx, benchmark in enumerate(benchmarks, 1):
        print(f"\n{'='*70}")
        print(f"[{bench_idx}/{len(benchmarks)}] Optimizing: {benchmark}")
        print(f"{'='*70}")
        
        bench_start = time.time()
        
        try:
            # Load benchmark data
            print(f"  📊 Loading {benchmark}...")
            result = loader._load_one_task(
                task_name=benchmark,
                split_ratio=0.8,
                seed=42,
                limit=args.limit,
                training_limit=None,
                testing_limit=None,
            )
            train_pairs = result["train_qa_pairs"]
            test_pairs = result["test_qa_pairs"]
            print(f"     ✓ {len(train_pairs.pairs)} train, {len(test_pairs.pairs)} test pairs")
            
            # Initialize evaluator
            EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
            evaluator = EvaluatorRotator(evaluator=None, task_name=benchmark)
            print(f"     ✓ Evaluator: {evaluator._plugin.name}")
            
            # Track best config across all methods
            best_config = None
            best_score = -1
            best_method = None
            
            # Test each method
            for method_name in methods:
                print(f"\n  🔬 Testing {method_name}...")
                
                # Get search space for this method
                search_space = get_search_space(
                    method_name, 
                    model.num_layers, 
                    quick=(search_mode == "quick")
                )
                
                total_configs = search_space.get_total_configs()
                print(f"     Search space: {total_configs} configurations")
                
                # For large search spaces, sample instead of exhaustive search
                max_configs = args.max_configs if args.max_configs else 50
                
                configs_tested = 0
                method_best_score = -1
                method_best_config = None
                
                # Iterate through search space
                for config in search_space.iterate():
                    if configs_tested >= max_configs:
                        break
                    
                    layer = config.get("layer", model.num_layers // 2)
                    strength = config.get("strength", 1.0)
                    
                    # Skip if layer out of range
                    if layer >= model.num_layers - 1:
                        continue
                    
                    # Collect activations
                    collector = ActivationCollector(model=model, store_device="cpu")
                    layer_str = str(layer)
                    
                    pos_acts = []
                    neg_acts = []
                    
                    for pair in train_pairs.pairs:
                        try:
                            updated_pair = collector.collect_for_pair(
                                pair,
                                layers=[layer_str],
                                aggregation=ActivationAggregationStrategy.LAST_TOKEN,
                                return_full_sequence=False,
                                normalize_layers=False,
                            )
                            
                            if (updated_pair.positive_response.layers_activations and
                                layer_str in updated_pair.positive_response.layers_activations):
                                act = updated_pair.positive_response.layers_activations[layer_str]
                                if act is not None:
                                    pos_acts.append(act)
                            
                            if (updated_pair.negative_response.layers_activations and
                                layer_str in updated_pair.negative_response.layers_activations):
                                act = updated_pair.negative_response.layers_activations[layer_str]
                                if act is not None:
                                    neg_acts.append(act)
                        except Exception:
                            continue
                    
                    if len(pos_acts) < 5 or len(neg_acts) < 5:
                        continue
                    
                    # Create steering method and train
                    try:
                        steering_method = create_steering_method(method_name, argparse.Namespace(**config))
                        steering_vector = steering_method.train_for_layer(pos_acts, neg_acts)
                    except Exception as e:
                        continue
                    
                    # Evaluate on test set
                    from wisent.core.models.core.atoms import SteeringPlan
                    plan = SteeringPlan.from_raw(
                        {layer_str: steering_vector},
                        scale=strength,
                        normalize=True,
                    )
                    
                    test_scores = []
                    for pair in test_pairs.pairs[:20]:  # Evaluate on subset for speed
                        try:
                            model.apply_steering(plan)
                            choices = [pair.negative_response.model_response, pair.positive_response.model_response]
                            expected = pair.positive_response.model_response
                            
                            score = evaluator.evaluate(
                                question=pair.prompt,
                                model_answer=expected,
                                choices=choices,
                                expected=expected,
                            )
                            test_scores.append(score)
                            model.detach()
                        except Exception:
                            model.detach()
                            continue
                    
                    if test_scores:
                        avg_score = sum(test_scores) / len(test_scores)
                        
                        if avg_score > method_best_score:
                            method_best_score = avg_score
                            method_best_config = {
                                "method": method_name,
                                "layer": layer,
                                "strength": strength,
                                "score": avg_score,
                                **config,
                            }
                    
                    configs_tested += 1
                    
                    # Progress indicator
                    if configs_tested % 10 == 0:
                        print(f"     Tested {configs_tested}/{min(total_configs, max_configs)} configs, best={method_best_score:.3f}")
                
                print(f"     {method_name} best: {method_best_score:.3f} (layer={method_best_config['layer'] if method_best_config else '?'})")
                
                # Update overall best
                if method_best_score > best_score:
                    best_score = method_best_score
                    best_config = method_best_config
                    best_method = method_name
            
            # Store the best result
            if best_config:
                store_optimization(
                    model=args.model,
                    task=benchmark,
                    layer=best_config["layer"],
                    strength=best_config["strength"],
                    method=best_method,
                    strategy=best_config.get("strategy", "constant"),
                    score=best_score,
                    metric="accuracy",
                    # Method-specific params
                    num_directions=best_config.get("num_directions", 1),
                    direction_weighting=best_config.get("direction_weighting", "primary_only"),
                    retain_weight=best_config.get("retain_weight", 0.0),
                    sensor_layer=best_config.get("sensor_layer", -1),
                    condition_threshold=best_config.get("condition_threshold", 0.5),
                    gate_temperature=best_config.get("gate_temperature", 0.5),
                    gate_hidden_dim=best_config.get("gate_hidden_dim", 64),
                    intensity_hidden_dim=best_config.get("intensity_hidden_dim", 32),
                    behavior_weight=best_config.get("behavior_weight", 1.0),
                    sparse_weight=best_config.get("sparse_weight", 0.05),
                )
                
                results["benchmarks"][benchmark] = {
                    "best_method": best_method,
                    "best_score": best_score,
                    "best_config": best_config,
                    "time": time.time() - bench_start,
                }
                
                # Track method wins
                results["best_method_counts"][best_method] = results["best_method_counts"].get(best_method, 0) + 1
                
                print(f"\n  ✅ {benchmark}: {best_method} @ layer {best_config['layer']} = {best_score:.3f}")
            else:
                print(f"\n  ⚠️  {benchmark}: No valid configuration found")
                results["benchmarks"][benchmark] = {"error": "no_valid_config"}
                
        except Exception as e:
            print(f"\n  ❌ {benchmark}: {str(e)}")
            results["benchmarks"][benchmark] = {"error": str(e)}
    
    # Summary
    total_time = time.time() - start_time
    results["total_time"] = total_time
    
    print(f"\n{'='*70}")
    print(f"✅ OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"   Model: {args.model}")
    print(f"   Benchmarks optimized: {len([b for b in results['benchmarks'] if 'error' not in results['benchmarks'][b]])}/{len(benchmarks)}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"\n   Method performance:")
    for method, count in sorted(results["best_method_counts"].items(), key=lambda x: -x[1]):
        print(f"      {method}: {count} wins")
    print(f"\n   Results saved to config cache.")
    print(f"   Use 'wisent generate-vector-from-task' or 'wisent modify-weights' to use optimal settings.")
    print(f"{'='*70}\n")
    
    # Save results to file
    results_file = f"./optimization_results/optimize_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📄 Full results saved to: {results_file}")
    
    return results
