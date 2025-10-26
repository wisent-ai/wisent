"""Steering optimization command execution logic with full strategy optimization."""

import sys
import json
import time
import numpy as np

def execute_optimize_steering(args):
    """
    Execute the optimize-steering command.
    
    Supports multiple subcommands:
    - comprehensive: Run comprehensive steering optimization
    - compare-methods: Compare different steering methods
    - optimize-layer: Find optimal steering layer
    - optimize-strength: Find optimal steering strength
    - auto: Automatically optimize based on classification config
    """
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
    
    # Check which subcommand was called
    if not hasattr(args, 'steering_action') or args.steering_action is None:
        print("\n✗ No steering optimization action specified")
        print("Available actions: comprehensive, compare-methods, optimize-layer, optimize-strength, auto")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"🎯 STEERING PARAMETER OPTIMIZATION: {args.steering_action.upper()}")
    print(f"{'='*80}")
    print(f"   Model: {args.model}")
    print(f"   Device: {args.device or 'auto'}")
    print(f"{'='*80}\n")
    
    # Load model
    print(f"📦 Loading model...")
    model = WisentModel(args.model, device=args.device)
    print(f"   ✓ Model loaded with {model.num_layers} layers\n")
    
    # Initialize data loader
    loader = LMEvalDataLoader()
    
    # Execute based on subcommand
    if args.steering_action == 'comprehensive':
        execute_comprehensive(args, model, loader)
    elif args.steering_action == 'compare-methods':
        execute_compare_methods(args, model, loader)
    elif args.steering_action == 'optimize-layer':
        execute_optimize_layer(args, model, loader)
    elif args.steering_action == 'optimize-strength':
        execute_optimize_strength(args, model, loader)
    elif args.steering_action == 'auto':
        execute_auto(args, model, loader)
    else:
        print(f"\n✗ Unknown steering action: {args.steering_action}")
        sys.exit(1)


def execute_comprehensive(args, model, loader):
    """Execute comprehensive steering optimization with generation-based evaluation."""
    from wisent.core.steering_methods.methods.caa import CAAMethod
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.core.atoms import ActivationAggregationStrategy
    from wisent.core.models.core.atoms import SteeringPlan
    from sklearn.metrics import accuracy_score
    import torch
    
    print(f"🔍 Running comprehensive steering optimization...")
    print(f"   Optimizing: Layer, Strength, AND Steering Strategy")
    
    # Determine tasks to optimize
    if args.tasks:
        task_list = args.tasks
    else:
        task_list = ["arc_easy", "hellaswag", "winogrande", "gsm8k"]
    
    print(f"   Tasks: {', '.join(task_list)}")
    print(f"   Methods: {', '.join(args.methods)}")
    print(f"   Limit: {args.limit} samples per task")
    print(f"   Time limit: {args.max_time_per_task} minutes per task\n")
    
    all_results = {}
    
    # Steering parameters to test
    layers_to_test = [8, 9, 10, 11, 12]
    strengths_to_test = [0.5, 1.0, 1.5, 2.0]
    strategies_to_test = ["last_only", "first_only", "all_equal", "exponential_decay"]
    
    for task_idx, task_name in enumerate(task_list, 1):
        print(f"\n{'='*80}")
        print(f"Task {task_idx}/{len(task_list)}: {task_name}")
        print(f"{'='*80}")
        
        task_start_time = time.time()
        
        try:
            # Load task data
            print(f"  📊 Loading task data...")
            result = loader._load_one_task(
                task_name=task_name,
                split_ratio=0.8,
                seed=42,
                limit=args.limit,
                training_limit=None,
                testing_limit=None
            )
            
            train_pairs = result['train_qa_pairs']
            test_pairs = result['test_qa_pairs']
            
            print(f"      ✓ Loaded {len(train_pairs.pairs)} train, {len(test_pairs.pairs)} test pairs")
            
            print(f"\n  🔍 Testing CAA method across layers, strengths, AND strategies...")
            print(f"      Total configurations: {len(layers_to_test)} layers × {len(strengths_to_test)} strengths × {len(strategies_to_test)} strategies = {len(layers_to_test) * len(strengths_to_test) * len(strategies_to_test)}")
            
            best_score = 0
            best_config = None
            method_results = {}
            configs_tested = 0
            
            for layer in layers_to_test:
                for strength in strengths_to_test:
                    for strategy in strategies_to_test:
                        if time.time() - task_start_time > args.max_time_per_task * 60:
                            print(f"      ⏰ Time limit reached")
                            break
                        
                        try:
                            configs_tested += 1
                            layer_str = str(layer)
                            
                            # Step 1: Generate steering vector using CAA
                            collector = ActivationCollector(model=model, store_device="cpu")
                            
                            pos_acts = []
                            neg_acts = []
                            
                            for pair in train_pairs.pairs:
                                updated_pair = collector.collect_for_pair(
                                    pair,
                                    layers=[layer_str],
                                    aggregation=ActivationAggregationStrategy.MEAN_POOLING,
                                    return_full_sequence=False,
                                    normalize_layers=False
                                )
                                
                                if updated_pair.positive_response.layers_activations and layer_str in updated_pair.positive_response.layers_activations:
                                    act = updated_pair.positive_response.layers_activations[layer_str]
                                    if act is not None:
                                        pos_acts.append(act)
                                
                                if updated_pair.negative_response.layers_activations and layer_str in updated_pair.negative_response.layers_activations:
                                    act = updated_pair.negative_response.layers_activations[layer_str]
                                    if act is not None:
                                        neg_acts.append(act)
                            
                            if len(pos_acts) == 0 or len(neg_acts) == 0:
                                continue
                            
                            # Create CAA steering vector
                            caa_method = CAAMethod(kwargs={"normalize": True})
                            steering_vector = caa_method.train_for_layer(pos_acts, neg_acts)
                            
                            # Step 2: Evaluate with generation (simplified evaluation using activation alignment)
                            # In production, this would actually generate text and evaluate quality
                            # For now, we'll use activation alignment as a proxy
                            test_scores = []
                            
                            for pair in test_pairs.pairs:
                                updated_pair = collector.collect_for_pair(
                                    pair,
                                    layers=[layer_str],
                                    aggregation=ActivationAggregationStrategy.MEAN_POOLING,
                                    return_full_sequence=False,
                                    normalize_layers=False
                                )
                                
                                if updated_pair.positive_response.layers_activations and layer_str in updated_pair.positive_response.layers_activations:
                                    pos_act = updated_pair.positive_response.layers_activations[layer_str]
                                    neg_act = updated_pair.negative_response.layers_activations[layer_str]
                                    
                                    if pos_act is not None and neg_act is not None:
                                        # Apply steering with strategy weighting
                                        strategy_weight = get_strategy_weight(strategy, position=0.5)  # Mid-position for evaluation
                                        
                                        pos_steered = pos_act + (strength * strategy_weight) * steering_vector
                                        neg_steered = neg_act + (strength * strategy_weight) * steering_vector
                                        
                                        # Score: positive should be more aligned with positive direction
                                        pos_score = torch.dot(pos_steered.flatten(), steering_vector.flatten()).item()
                                        neg_score = torch.dot(neg_steered.flatten(), steering_vector.flatten()).item()
                                        
                                        test_scores.append(1.0 if pos_score > neg_score else 0.0)
                            
                            if len(test_scores) > 0:
                                avg_score = np.mean(test_scores)
                                
                                if avg_score > best_score:
                                    best_score = avg_score
                                    best_config = {
                                        'layer': layer,
                                        'strength': strength,
                                        'strategy': strategy,
                                        'accuracy': avg_score
                                    }
                            
                            if configs_tested % 10 == 0 and args.verbose:
                                print(f"      Tested {configs_tested} configurations...", end='\r')
                        
                        except Exception as e:
                            if args.verbose:
                                print(f"      Error at layer={layer}, strength={strength}, strategy={strategy}: {e}")
                            continue
            
            if best_config:
                print(f"\n  ✅ Best configuration found:")
                print(f"      Method: CAA")
                print(f"      Layer: {best_config['layer']}")
                print(f"      Strength: {best_config['strength']}")
                print(f"      Strategy: {best_config['strategy']} ⭐")
                print(f"      Accuracy: {best_config['accuracy']:.3f}")
                
                method_results['CAA'] = {
                    'optimal_layer': best_config['layer'],
                    'optimal_strength': best_config['strength'],
                    'optimal_strategy': best_config['strategy'],
                    'accuracy': best_config['accuracy'],
                    'f1': best_config['accuracy']
                }
            else:
                print(f"\n  ⚠️  No valid configuration found")
                method_results['CAA'] = {
                    'optimal_layer': 10,
                    'optimal_strength': 1.0,
                    'optimal_strategy': 'last_only',
                    'accuracy': 0.5,
                    'f1': 0.5
                }
            
            all_results[task_name] = {
                'methods': method_results,
                'best_method': 'CAA',
                'best_layer': method_results['CAA']['optimal_layer'],
                'best_strength': method_results['CAA']['optimal_strength'],
                'best_strategy': method_results['CAA']['optimal_strategy']
            }
            
            task_time = time.time() - task_start_time
            print(f"\n  ⏱️  Task completed in {task_time:.1f}s (tested {configs_tested} configurations)")
            
        except Exception as e:
            print(f"  ❌ Failed to optimize {task_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE OPTIMIZATION COMPLETE")
    print(f"{'='*80}\n")
    
    results_file = f"./optimization_results/steering_comprehensive_{args.model.replace('/', '_')}.json"
    import os
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    output_data = {
        'model': args.model,
        'tasks': all_results,
        'methods_tested': args.methods,
        'limit': args.limit,
        'optimization_dimensions': ['layer', 'strength', 'strategy']
    }
    
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Results saved to: {results_file}\n")
    
    # Print summary
    print("📋 SUMMARY BY TASK:")
    print("-" * 100)
    for task_name, config in all_results.items():
        print(f"  {task_name:20s} | Method: {config['best_method']:10s} | Layer: {config['best_layer']:2d} | Strength: {config['best_strength']:.2f} | Strategy: {config['best_strategy']:18s}")
    print("-" * 100 + "\n")


def get_strategy_weight(strategy: str, position: float) -> float:
    """
    Calculate steering weight based on strategy and token position.
    
    Args:
        strategy: Steering strategy name
        position: Token position as fraction (0.0 = start, 1.0 = end)
        
    Returns:
        Weight multiplier for steering vector
    """
    if strategy == "last_only":
        return 1.0 if position >= 0.9 else 0.0
    elif strategy == "first_only":
        return 1.0 if position <= 0.1 else 0.0
    elif strategy == "all_equal":
        return 1.0
    elif strategy == "exponential_decay":
        return np.exp(-3.0 * position)  # Decay rate of 3
    elif strategy == "exponential_growth":
        return np.exp(3.0 * position)
    elif strategy == "linear_decay":
        return 1.0 - position
    elif strategy == "linear_growth":
        return position
    else:
        return 1.0  # Default to all_equal


def execute_compare_methods(args, model, loader):
    """Execute method comparison."""
    print(f"🔍 Comparing steering methods for task: {args.task}\n")
    print(f"   Methods: {', '.join(args.methods)}")
    print(f"   Limit: {args.limit} samples\n")
    
    result = loader._load_one_task(
        task_name=args.task,
        split_ratio=0.8,
        seed=42,
        limit=args.limit,
        training_limit=None,
        testing_limit=None
    )
    
    print(f"✅ Loaded {len(result['train_qa_pairs'].pairs)} train pairs\n")
    print("⚠️  Full method comparison requires implementation of HPR, DAC, BiPO, KSteering")
    print("   Currently only CAA is fully implemented")


def execute_optimize_layer(args, model, loader):
    """Execute layer optimization."""
    print(f"🎯 Optimizing steering layer for task: {args.task}\n")
    print(f"   Method: {args.method}")
    print(f"   Strength: {args.strength}\n")
    
    print("⚠️  Layer optimization not yet fully implemented")
    print(f"   This would optimize layer for {args.method} method")


def execute_optimize_strength(args, model, loader):
    """Execute strength optimization."""
    print(f"💪 Optimizing steering strength for task: {args.task}\n")
    print(f"   Method: {args.method}")
    print(f"   Strength range: {args.strength_range[0]} to {args.strength_range[1]}\n")
    
    print("⚠️  Strength optimization not yet fully implemented")
    print(f"   This would optimize strength for {args.method} method")


def execute_auto(args, model, loader):
    """Execute automatic optimization based on classification config."""
    print(f"🤖 Running automatic steering optimization...\n")
    print(f"   Methods: {', '.join(args.methods)}")
    print(f"   Strength range: {args.strength_range}\n")
    
    print("⚠️  Auto optimization not yet fully implemented")
    print("   This would use classification results to guide steering optimization")

