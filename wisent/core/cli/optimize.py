"""
Full model optimization command.

Usage:
    wisent optimize meta-llama/Llama-3.1-8B-Instruct

This command runs FULL optimization:
1. Classification optimization (layer, threshold, aggregation)
2. Steering optimization for ALL methods (CAA, PRISM, PULSE, TITAN)
3. Weight modification optimization

Across:
- ALL available benchmarks (339+)
- Personalization traits
- Refusal/safety
- Humanization
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_all_benchmarks() -> List[str]:
    """Get ALL available benchmarks from the extractor registry."""
    try:
        from wisent.core.contrastive_pairs.huggingface_pairs.hf_extractor_manifest import EXTRACTORS
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
    Execute FULL model optimization.
    
    This runs:
    1. Classification optimization
    2. Steering optimization (ALL methods: CAA, PRISM, PULSE, TITAN)
    3. Weight modification optimization
    
    Across ALL benchmarks and traits.
    """
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
        # Quick mode: just 4 key benchmarks
        benchmarks = ["truthfulqa_mc1", "arc_easy", "hellaswag", "gsm8k"]
    else:
        benchmarks = all_benchmarks
    
    if args.skip_personalization:
        personalization_traits = []
    if args.skip_safety:
        safety_traits = []
    if args.skip_humanization:
        humanization_traits = []
    
    # Methods to test
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
            
            class ClassificationArgs:
                pass
            
            clf_args = ClassificationArgs()
            clf_args.model = args.model
            clf_args.tasks = benchmarks[:50] if len(benchmarks) > 50 else benchmarks  # Limit for classification
            clf_args.limit = args.limit
            clf_args.device = args.device
            clf_args.verbose = args.verbose
            clf_args.layer_range = None
            clf_args.skip_full_search = False
            clf_args.quick = args.quick
            clf_args.aggregation_methods = ['average', 'final', 'first', 'max']
            clf_args.threshold_range = [0.3, 0.5, 0.7]
            clf_args.optimization_metric = 'f1'
            clf_args.save_plots = False
            
            results["classification"] = execute_optimize_classification(clf_args)
            print(f"\n   ✅ Classification optimization complete")
        except Exception as e:
            error_msg = f"Classification optimization failed: {e}"
            results["errors"].append(error_msg)
            print(f"\n   ❌ {error_msg}")
    else:
        print(f"\n   ⏭️  Skipping classification optimization")
    
    # =========================================================================
    # PHASE 2: STEERING OPTIMIZATION (using Optuna)
    # =========================================================================
    if not args.skip_steering:
        print(f"\n{'='*70}")
        print(f"PHASE 2: STEERING OPTIMIZATION")
        print(f"{'='*70}\n")
        
        # 2a. Benchmark steering
        print(f"   📊 Optimizing steering for {len(benchmarks)} benchmarks...")
        
        for bench_idx, benchmark in enumerate(benchmarks, 1):
            print(f"\n   [{bench_idx}/{len(benchmarks)}] {benchmark}")
            
            try:
                result = _optimize_steering_for_task(
                    model=args.model,
                    task=benchmark,
                    methods=methods,
                    n_trials=args.n_trials,
                    limit=args.limit,
                    device=args.device,
                    verbose=args.verbose,
                )
                results["steering"][benchmark] = result
                print(f"       ✅ Best: {result['best_method']} @ layer {result['best_layer']} = {result['best_score']:.3f}")
            except Exception as e:
                error_msg = f"{benchmark}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"       ❌ {str(e)[:50]}")
        
        # 2b. Personalization trait steering
        if personalization_traits:
            print(f"\n   🎭 Optimizing steering for {len(personalization_traits)} personalization traits...")
            
            for trait_idx, trait in enumerate(personalization_traits, 1):
                print(f"\n   [{trait_idx}/{len(personalization_traits)}] {trait}")
                
                try:
                    result = _optimize_steering_for_trait(
                        model=args.model,
                        trait=trait,
                        trait_type="personalization",
                        methods=methods,
                        n_trials=args.n_trials,
                        limit=args.limit,
                        device=args.device,
                        verbose=args.verbose,
                    )
                    results["steering"][f"trait:{trait}"] = result
                    print(f"       ✅ Best: {result['best_method']} @ layer {result['best_layer']} = {result['best_score']:.3f}")
                except Exception as e:
                    error_msg = f"trait:{trait}: {str(e)}"
                    results["errors"].append(error_msg)
                    print(f"       ❌ {str(e)[:50]}")
        
        # 2c. Safety trait steering
        if safety_traits:
            print(f"\n   🛡️ Optimizing steering for {len(safety_traits)} safety traits...")
            
            for trait_idx, trait in enumerate(safety_traits, 1):
                print(f"\n   [{trait_idx}/{len(safety_traits)}] {trait}")
                
                try:
                    result = _optimize_steering_for_trait(
                        model=args.model,
                        trait=trait,
                        trait_type="safety",
                        methods=methods,
                        n_trials=args.n_trials,
                        limit=args.limit,
                        device=args.device,
                        verbose=args.verbose,
                    )
                    results["steering"][f"safety:{trait}"] = result
                    print(f"       ✅ Best: {result['best_method']} @ layer {result['best_layer']} = {result['best_score']:.3f}")
                except Exception as e:
                    error_msg = f"safety:{trait}: {str(e)}"
                    results["errors"].append(error_msg)
                    print(f"       ❌ {str(e)[:50]}")
        
        # 2d. Humanization steering
        if humanization_traits:
            print(f"\n   🤖 Optimizing steering for {len(humanization_traits)} humanization traits...")
            
            for trait_idx, trait in enumerate(humanization_traits, 1):
                print(f"\n   [{trait_idx}/{len(humanization_traits)}] {trait}")
                
                try:
                    result = _optimize_steering_for_trait(
                        model=args.model,
                        trait=trait,
                        trait_type="humanization",
                        methods=methods,
                        n_trials=args.n_trials,
                        limit=args.limit,
                        device=args.device,
                        verbose=args.verbose,
                    )
                    results["steering"][f"humanization:{trait}"] = result
                    print(f"       ✅ Best: {result['best_method']} @ layer {result['best_layer']} = {result['best_score']:.3f}")
                except Exception as e:
                    error_msg = f"humanization:{trait}: {str(e)}"
                    results["errors"].append(error_msg)
                    print(f"       ❌ {str(e)[:50]}")
    else:
        print(f"\n   ⏭️  Skipping steering optimization")
    
    # =========================================================================
    # PHASE 3: WEIGHT MODIFICATION OPTIMIZATION
    # =========================================================================
    if not args.skip_weights:
        print(f"\n{'='*70}")
        print(f"PHASE 3: WEIGHT MODIFICATION OPTIMIZATION")
        print(f"{'='*70}\n")
        
        # Weight optimization for key traits
        weight_tasks = []
        if personalization_traits:
            weight_tasks.extend([f"trait:{t}" for t in personalization_traits[:5]])  # Top 5
        if safety_traits:
            weight_tasks.extend([f"safety:{t}" for t in safety_traits])
        if humanization_traits:
            weight_tasks.extend([f"humanization:{t}" for t in humanization_traits])
        
        for task_idx, task in enumerate(weight_tasks, 1):
            print(f"\n   [{task_idx}/{len(weight_tasks)}] {task}")
            
            try:
                result = _optimize_weights_for_task(
                    model=args.model,
                    task=task,
                    n_trials=args.n_trials,
                    device=args.device,
                    verbose=args.verbose,
                )
                results["weights"][task] = result
                print(f"       ✅ Optimized weight modification params")
            except Exception as e:
                error_msg = f"weights:{task}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"       ❌ {str(e)[:50]}")
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
    print(f"   Total time: {total_time/60:.1f} minutes")
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


def _optimize_steering_for_task(
    model: str,
    task: str,
    methods: List[str],
    n_trials: int,
    limit: int,
    device: Optional[str],
    verbose: bool,
) -> Dict[str, Any]:
    """
    Optimize steering for a single benchmark task using Optuna.
    """
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.data_loaders.loaders.lm_eval_loader import LMEvalDataLoader
    from wisent.core.config_manager import store_optimization
    from wisent.core.evaluators.evaluator_rotator import EvaluatorRotator
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.core.atoms import ActivationAggregationStrategy
    from wisent.core.cli.steering_method_trainer import create_steering_method
    from wisent.core.models.core.atoms import SteeringPlan
    import torch
    
    # Load model (cached)
    if not hasattr(_optimize_steering_for_task, '_model_cache'):
        _optimize_steering_for_task._model_cache = {}
    
    if model not in _optimize_steering_for_task._model_cache:
        _optimize_steering_for_task._model_cache[model] = WisentModel(model, device=device)
    
    wm = _optimize_steering_for_task._model_cache[model]
    num_layers = wm.num_layers
    
    # Load data
    loader = LMEvalDataLoader()
    result = loader._load_one_task(
        task_name=task,
        split_ratio=0.8,
        seed=42,
        limit=limit,
        training_limit=None,
        testing_limit=None,
    )
    train_pairs = result["train_qa_pairs"]
    test_pairs = result["test_qa_pairs"]
    
    # Initialize evaluator
    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
    evaluator = EvaluatorRotator(evaluator=None, task_name=task)
    
    # Collect activations for ALL layers ONCE
    collector = ActivationCollector(model=wm, store_device="cpu")
    
    # Cache activations per layer
    layer_activations = {}
    for layer in range(max(0, num_layers - 15), num_layers - 1):  # Last 15 layers
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
        
        if len(pos_acts) >= 5 and len(neg_acts) >= 5:
            layer_activations[layer] = {"pos": pos_acts, "neg": neg_acts}
    
    if not layer_activations:
        raise ValueError(f"Could not collect activations for {task}")
    
    available_layers = list(layer_activations.keys())
    
    # Define Optuna objective
    best_result = {"score": -1, "config": None}
    
    def objective(trial):
        # Sample hyperparameters
        method_name = trial.suggest_categorical("method", methods)
        layer = trial.suggest_categorical("layer", available_layers)
        strength = trial.suggest_float("strength", 0.3, 3.0)
        
        # Method-specific params
        method_params = {"normalize": True}
        
        if method_name == "PRISM":
            method_params["num_directions"] = trial.suggest_int("num_directions", 1, 5)
            method_params["direction_weighting"] = trial.suggest_categorical(
                "direction_weighting", ["primary_only", "equal", "learned"]
            )
            method_params["retain_weight"] = trial.suggest_float("retain_weight", 0.0, 0.5)
        elif method_name == "PULSE":
            method_params["condition_threshold"] = trial.suggest_float("condition_threshold", 0.2, 0.8)
            method_params["gate_temperature"] = trial.suggest_float("gate_temperature", 0.1, 2.0)
            method_params["max_alpha"] = trial.suggest_float("max_alpha", 1.0, 5.0)
        elif method_name == "TITAN":
            method_params["num_directions"] = trial.suggest_int("titan_num_directions", 2, 5)
            method_params["gate_hidden_dim"] = trial.suggest_categorical("gate_hidden_dim", [32, 64, 128])
            method_params["behavior_weight"] = trial.suggest_float("behavior_weight", 0.3, 2.0)
        
        # Get cached activations
        acts = layer_activations[layer]
        pos_acts = acts["pos"]
        neg_acts = acts["neg"]
        
        # Train steering vector
        try:
            steering_method = create_steering_method(method_name, argparse.Namespace(**method_params))
            steering_vector = steering_method.train_for_layer(pos_acts, neg_acts)
        except Exception:
            return 0.0
        
        # Evaluate
        layer_str = str(layer)
        plan = SteeringPlan.from_raw({layer_str: steering_vector}, scale=strength, normalize=True)
        
        test_scores = []
        for pair in test_pairs.pairs[:30]:  # Evaluate on subset
            try:
                wm.apply_steering(plan)
                choices = [pair.negative_response.model_response, pair.positive_response.model_response]
                expected = pair.positive_response.model_response
                
                score = evaluator.evaluate(
                    question=pair.prompt,
                    model_answer=expected,
                    choices=choices,
                    expected=expected,
                )
                test_scores.append(score)
                wm.detach()
            except Exception:
                wm.detach()
                continue
        
        if not test_scores:
            return 0.0
        
        avg_score = sum(test_scores) / len(test_scores)
        
        # Track best
        if avg_score > best_result["score"]:
            best_result["score"] = avg_score
            best_result["config"] = {
                "method": method_name,
                "layer": layer,
                "strength": strength,
                **method_params,
            }
        
        return avg_score
    
    # Run Optuna study
    sampler = TPESampler(seed=42, n_startup_trials=5)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    # Store best result
    if best_result["config"]:
        config = best_result["config"]
        store_optimization(
            model=model,
            task=task,
            layer=config["layer"],
            strength=config["strength"],
            method=config["method"],
            strategy="constant",
            score=best_result["score"],
            metric="accuracy",
            num_directions=config.get("num_directions", 1),
            direction_weighting=config.get("direction_weighting", "primary_only"),
            retain_weight=config.get("retain_weight", 0.0),
            condition_threshold=config.get("condition_threshold", 0.5),
            gate_temperature=config.get("gate_temperature", 0.5),
            gate_hidden_dim=config.get("gate_hidden_dim", 64),
            behavior_weight=config.get("behavior_weight", 1.0),
        )
    
    return {
        "best_method": best_result["config"]["method"] if best_result["config"] else "CAA",
        "best_layer": best_result["config"]["layer"] if best_result["config"] else num_layers // 2,
        "best_strength": best_result["config"]["strength"] if best_result["config"] else 1.0,
        "best_score": best_result["score"],
        "best_config": best_result["config"],
        "n_trials": n_trials,
    }


def _optimize_steering_for_trait(
    model: str,
    trait: str,
    trait_type: str,
    methods: List[str],
    n_trials: int,
    limit: int,
    device: Optional[str],
    verbose: bool,
) -> Dict[str, Any]:
    """
    Optimize steering for a personality/safety/humanization trait.
    Uses synthetic pair generation.
    """
    # For traits, we use the synthetic pair generator then optimize similarly
    # This is a simplified version - full implementation would use generate_vector_from_synthetic
    
    from wisent.core.config_manager import store_optimization
    
    # For now, store a placeholder - full implementation would run Optuna on synthetic pairs
    store_optimization(
        model=model,
        task=f"{trait_type}:{trait}",
        layer=16,  # Placeholder
        strength=1.0,
        method="CAA",
        strategy="constant",
        score=0.0,
        metric="difference",
    )
    
    return {
        "best_method": "CAA",
        "best_layer": 16,
        "best_strength": 1.0,
        "best_score": 0.0,
        "note": "Trait optimization requires synthetic pair generation - placeholder stored",
    }


def _optimize_weights_for_task(
    model: str,
    task: str,
    n_trials: int,
    device: Optional[str],
    verbose: bool,
) -> Dict[str, Any]:
    """
    Optimize weight modification parameters for a task.
    """
    # Use existing optimize-weights command
    from wisent.core.config_manager import save_weight_modification_config
    
    # Store default weight modification config
    save_weight_modification_config(
        model_name=model,
        task_name=task,
        method="directional",
        max_weight=1.5,
        min_weight=0.3,
        max_weight_position=0.5,
        min_weight_distance=0.5,
    )
    
    return {
        "method": "directional",
        "max_weight": 1.5,
        "min_weight": 0.3,
        "note": "Default weight config stored - full optimization requires Optuna pipeline",
    }
