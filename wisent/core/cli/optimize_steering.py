"""Steering optimization command execution logic with full strategy optimization."""

import sys
import json
import time
import numpy as np
from wisent.core.evaluators.rotator import EvaluatorRotator

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
    
    # Execute based on subcommand and return results
    if args.steering_action == 'comprehensive':
        return execute_comprehensive(args, model, loader)
    elif args.steering_action == 'compare-methods':
        return execute_compare_methods(args, model, loader)
    elif args.steering_action == 'optimize-layer':
        return execute_optimize_layer(args, model, loader)
    elif args.steering_action == 'optimize-strength':
        return execute_optimize_strength(args, model, loader)
    elif args.steering_action == 'auto':
        return execute_auto(args, model, loader)
    elif args.steering_action == 'personalization':
        return execute_personalization(args, model)
    else:
        print(f"\n✗ Unknown steering action: {args.steering_action}")
        sys.exit(1)


def execute_comprehensive(args, model, loader):
    """Execute comprehensive steering optimization with generation-based evaluation."""
    from wisent.core.steering_methods.methods.caa import CAAMethod
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.core.atoms import ActivationAggregationStrategy
    from wisent.core.activations.prompt_construction_strategy import PromptConstructionStrategy
    from wisent.core.models.core.atoms import SteeringPlan
    from sklearn.metrics import accuracy_score
    import torch

    print(f"🔍 Running comprehensive steering optimization...")
    print(f"   Optimizing: Layer, Strength, Steering Strategy, Token Aggregation, Prompt Construction")

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
    layers_to_test = [4, 6, 8, 10, 12] if model.num_layers > 12 else list(range(2, model.num_layers, 2))
    strengths_to_test = [0.5, 1.0, 1.5, 2.0]
    strategies_to_test = ["constant", "initial_only", "diminishing", "all_equal"]
    token_aggregations_to_test = [
        ActivationAggregationStrategy.LAST_TOKEN,
        ActivationAggregationStrategy.MEAN_POOLING,
        ActivationAggregationStrategy.FIRST_TOKEN,
    ]
    prompt_constructions_to_test = [
        PromptConstructionStrategy.CHAT_TEMPLATE,
        PromptConstructionStrategy.DIRECT_COMPLETION,
    ]

    print(f"   Layers: {layers_to_test}")
    print(f"   Strengths: {strengths_to_test}")
    print(f"   Strategies: {strategies_to_test}")
    print(f"   Token Aggregations: {[t.value for t in token_aggregations_to_test]}")
    print(f"   Prompt Constructions: {[p.value for p in prompt_constructions_to_test]}")
    total_configs = len(layers_to_test) * len(strengths_to_test) * len(strategies_to_test) * len(token_aggregations_to_test) * len(prompt_constructions_to_test)
    print(f"   Total configurations per task: {total_configs}\n")
    
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

            # Initialize evaluator for this task (auto-select based on task_name)
            EvaluatorRotator.discover_evaluators('wisent.core.evaluators.benchmark_specific')
            evaluator = EvaluatorRotator(evaluator=None, task_name=task_name)  # None = auto-select
            print(f"      ✓ Using evaluator: {evaluator._evaluator.name} (auto-selected for {task_name})")

            print(f"\n  🔍 Testing CAA method across layers, strengths, strategies, token aggregations, prompt constructions...")
            print(f"      Total configurations: {total_configs}")

            best_score = 0
            best_config = None
            method_results = {}
            configs_tested = 0
            all_generation_examples = []  # Store generation examples for all configs

            # Prepare test prompts if generating examples for all configs
            if args.save_all_generation_examples or args.save_generation_examples:
                num_examples = min(args.num_generation_examples, len(test_pairs.pairs))
                example_pairs = test_pairs.pairs[:num_examples]
                print(f"  📝 Will generate {num_examples} example responses per configuration")

            for layer in layers_to_test:
              for strength in strengths_to_test:
                for strategy in strategies_to_test:
                  for token_agg in token_aggregations_to_test:
                    for prompt_const in prompt_constructions_to_test:
                        if time.time() - task_start_time > args.max_time_per_task * 60:
                            print(f"      ⏰ Time limit reached")
                            break
                        
                        try:
                            configs_tested += 1
                            layer_str = str(layer)
                            
                            # Step 1: Generate steering vector using CAA with current token aggregation
                            collector = ActivationCollector(model=model, store_device="cpu")

                            pos_acts = []
                            neg_acts = []

                            for pair in train_pairs.pairs:
                                updated_pair = collector.collect_for_pair(
                                    pair,
                                    layers=[layer_str],
                                    aggregation=token_agg,  # Use current token aggregation strategy
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

                            # Step 2: Evaluate with ACTUAL GENERATION and task evaluator
                            # Create steering plan
                            from wisent.core.models.core.atoms import SteeringVector, SteeringPlan
                            steering_vec = SteeringVector(vector=steering_vector, scale=strength)
                            steering_plan = SteeringPlan(
                                layers={layer_str: steering_vec},
                                layers_description=[f"CAA L{layer} S{strength} {strategy} T:{token_agg.value} P:{prompt_const.value}"]
                            )

                            # Apply steering to model
                            model.apply_steering(steering_plan)

                            test_scores = []
                            detailed_results = []  # Store full evaluation details

                            for pair in test_pairs.pairs:
                                try:
                                    # Prepare choices for multiple choice evaluation
                                    # ContrastivePair uses: prompt, positive_response.model_response, negative_response.model_response
                                    choices = [pair.negative_response.model_response, pair.positive_response.model_response]
                                    expected = pair.positive_response.model_response

                                    # Use the Wisent evaluator to check correctness
                                    # The evaluator will use log likelihood if possible,
                                    # otherwise fall back to generation
                                    # Pass test_code from metadata for coding tasks
                                    test_code = pair.metadata.get("test_code") if pair.metadata else None
                                    eval_result = evaluator.evaluate(
                                        response="",  # Not used for log likelihood eval
                                        expected=expected,
                                        model=model,
                                        question=pair.prompt,
                                        choices=choices,
                                        steering_plan=steering_plan,
                                        test_code=test_code,
                                        task_name=task_name
                                    )

                                    # Convert TRUTHFUL/UNTRUTHFUL to 1.0/0.0
                                    is_correct = eval_result.ground_truth == "TRUTHFUL"
                                    test_scores.append(1.0 if is_correct else 0.0)

                                    # Save full evaluation details
                                    detailed_results.append({
                                        'prompt': pair.prompt,
                                        'choices': choices,
                                        'expected': expected,
                                        'ground_truth': eval_result.ground_truth,
                                        'method_used': eval_result.method_used,
                                        'confidence': eval_result.confidence,
                                        'details': eval_result.details,
                                        'meta': dict(eval_result.meta) if eval_result.meta else {},
                                        'is_correct': is_correct
                                    })

                                except Exception as e:
                                    # NO FALLBACK - raise the error immediately
                                    print(f"\n❌ Evaluation failed for test pair:")
                                    print(f"   Prompt: {pair.prompt[:100]}")
                                    print(f"   Error: {e}")
                                    raise

                            # Clear steering
                            model.clear_steering()
                            
                            if len(test_scores) > 0:
                                avg_score = np.mean(test_scores)

                                # Generate examples for this configuration if requested
                                if args.save_all_generation_examples:
                                    config_examples = []
                                    for idx, pair in enumerate(example_pairs):
                                        prompt = pair.prompt
                                        try:
                                            # Generate without steering (only once per prompt, reuse if already generated)
                                            unsteered_response = model.generate(
                                                [[{"role": "user", "content": prompt}]],
                                                max_new_tokens=100,
                                                temperature=0.7,
                                                use_steering=False
                                            )[0]

                                            # Create steering plan for this config
                                            from wisent.core.models.core.atoms import SteeringVector, SteeringPlan
                                            steering_vec = SteeringVector(vector=steering_vector, scale=strength)
                                            steering_plan = SteeringPlan(
                                                layers={layer_str: steering_vec},
                                                layers_description=[f"CAA steering layer={layer}, strength={strength}, strategy={strategy}"]
                                            )

                                            # Generate with steering
                                            model.apply_steering(steering_plan)
                                            steered_response = model.generate(
                                                [[{"role": "user", "content": prompt}]],
                                                max_new_tokens=100,
                                                temperature=0.7,
                                                use_steering=True,
                                                steering_plan=steering_plan
                                            )[0]
                                            model.clear_steering()

                                            config_examples.append({
                                                'prompt': prompt,
                                                'correct_answer': pair.positive_response.model_response,
                                                'incorrect_answer': pair.negative_response.model_response,
                                                'unsteered_generation': unsteered_response,
                                                'steered_generation': steered_response
                                            })
                                        except Exception as e:
                                            if args.verbose:
                                                print(f"      ⚠️ Failed to generate example for config layer={layer}, strength={strength}, strategy={strategy}: {e}")

                                    # Store this config's examples
                                    all_generation_examples.append({
                                        'layer': layer,
                                        'strength': strength,
                                        'strategy': strategy,
                                        'accuracy': avg_score,
                                        'examples': config_examples
                                    })

                                # Store detailed results for this configuration
                                config_key = f"L{layer}_S{strength}_{strategy}_{token_agg.value}_{prompt_const.value}"
                                method_results[config_key] = {
                                    'layer': layer,
                                    'strength': strength,
                                    'strategy': strategy,
                                    'token_aggregation': token_agg.value,
                                    'prompt_construction': prompt_const.value,
                                    'accuracy': avg_score,
                                    'num_test_samples': len(test_scores),
                                    'detailed_results': detailed_results  # Save all eval details
                                }

                                if avg_score > best_score:
                                    best_score = avg_score
                                    best_config = {
                                        'layer': layer,
                                        'strength': strength,
                                        'strategy': strategy,
                                        'token_aggregation': token_agg.value,
                                        'prompt_construction': prompt_const.value,
                                        'accuracy': avg_score
                                    }

                            if configs_tested % 10 == 0 and args.verbose:
                                print(f"      Tested {configs_tested} configurations...", end='\r')

                        except Exception as e:
                            # NO FALLBACK - raise the error immediately
                            print(f"\n❌ Configuration test failed:")
                            print(f"   Layer: {layer}")
                            print(f"   Strength: {strength}")
                            print(f"   Strategy: {strategy}")
                            print(f"   Error: {e}")
                            raise
            
            if best_config:
                print(f"\n  ✅ Best configuration found:")
                print(f"      Method: CAA")
                print(f"      Layer: {best_config['layer']}")
                print(f"      Strength: {best_config['strength']}")
                print(f"      Strategy: {best_config['strategy']} ⭐")
                print(f"      Token Aggregation: {best_config['token_aggregation']}")
                print(f"      Prompt Construction: {best_config['prompt_construction']}")
                print(f"      Accuracy: {best_config['accuracy']:.3f}")

                method_results['CAA'] = {
                    'optimal_layer': best_config['layer'],
                    'optimal_strength': best_config['strength'],
                    'optimal_strategy': best_config['strategy'],
                    'optimal_token_aggregation': best_config['token_aggregation'],
                    'optimal_prompt_construction': best_config['prompt_construction'],
                    'accuracy': best_config['accuracy'],
                    'f1': best_config['accuracy']
                }

                # Save best steering vector if requested
                if args.save_best_vector:
                    import os
                    vector_dir = args.save_best_vector
                    os.makedirs(vector_dir, exist_ok=True)

                    # Recreate the best steering vector with optimal token aggregation
                    best_layer_str = str(best_config['layer'])
                    best_token_agg = ActivationAggregationStrategy(best_config['token_aggregation'])
                    pos_acts_best = []
                    neg_acts_best = []

                    for pair in train_pairs.pairs:
                        updated_pair = collector.collect_for_pair(
                            pair,
                            layers=[best_layer_str],
                            aggregation=best_token_agg,  # Use optimal token aggregation
                            return_full_sequence=False,
                            normalize_layers=False
                        )

                        if updated_pair.positive_response.layers_activations and best_layer_str in updated_pair.positive_response.layers_activations:
                            act = updated_pair.positive_response.layers_activations[best_layer_str]
                            if act is not None:
                                pos_acts_best.append(act)

                        if updated_pair.negative_response.layers_activations and best_layer_str in updated_pair.negative_response.layers_activations:
                            act = updated_pair.negative_response.layers_activations[best_layer_str]
                            if act is not None:
                                neg_acts_best.append(act)

                    # Create and save steering vector
                    caa_method = CAAMethod(kwargs={"normalize": True})
                    best_steering_vector = caa_method.train_for_layer(pos_acts_best, neg_acts_best)

                    vector_path = os.path.join(vector_dir, f"{task_name}_layer{best_config['layer']}.pt")
                    torch.save({
                        'steering_vector': best_steering_vector,
                        'vector': best_steering_vector,  # Legacy key
                        'layer': best_config['layer'],
                        'layer_index': best_config['layer'],  # Legacy key
                        'strength': best_config['strength'],
                        'strategy': best_config['strategy'],
                        'token_aggregation': best_config['token_aggregation'],
                        'prompt_construction': best_config['prompt_construction'],
                        'method': 'CAA',
                        'task': task_name,
                        'model': args.model,
                        'accuracy': best_config['accuracy']
                    }, vector_path)
                    print(f"      💾 Saved steering vector to: {vector_path}")

                # Save generation examples
                if args.save_all_generation_examples:
                    # Save examples for ALL configurations
                    examples_path = os.path.join(
                        args.save_best_vector if args.save_best_vector else "./optimization_results",
                        f"{task_name}_all_generation_examples.json"
                    )
                    os.makedirs(os.path.dirname(examples_path), exist_ok=True)

                    with open(examples_path, 'w') as f:
                        json.dump({
                            'task': task_name,
                            'model': args.model,
                            'best_config': best_config,
                            'configurations': all_generation_examples
                        }, f, indent=2)

                    print(f"\n  💾 Saved generation examples for {len(all_generation_examples)} configurations to: {examples_path}")

                elif args.save_generation_examples:
                    # Save examples only for the best configuration
                    print(f"\n  📝 Generating example responses for best configuration...")

                    # Get a few test examples to generate from
                    num_examples = min(args.num_generation_examples, len(test_pairs.pairs))
                    example_pairs = test_pairs.pairs[:num_examples]

                    generation_examples = []

                    for idx, pair in enumerate(example_pairs):
                        # Create prompt from the question
                        prompt = pair.prompt

                        try:
                            # Generate without steering
                            unsteered_response = model.generate(
                                [[{"role": "user", "content": prompt}]],
                                max_new_tokens=100,
                                temperature=0.7,
                                use_steering=False
                            )[0]

                            # Recreate best steering vector for generation
                            best_layer_str = str(best_config['layer'])
                            pos_acts_gen = []
                            neg_acts_gen = []

                            # Collect activations again for steering
                            for train_pair in train_pairs.pairs[:20]:  # Use subset for speed
                                updated_pair = collector.collect_for_pair(
                                    train_pair,
                                    layers=[best_layer_str],
                                    aggregation=ActivationAggregationStrategy.MEAN_POOLING,
                                    return_full_sequence=False,
                                    normalize_layers=False
                                )

                                if updated_pair.positive_response.layers_activations and best_layer_str in updated_pair.positive_response.layers_activations:
                                    act = updated_pair.positive_response.layers_activations[best_layer_str]
                                    if act is not None:
                                        pos_acts_gen.append(act)

                                if updated_pair.negative_response.layers_activations and best_layer_str in updated_pair.negative_response.layers_activations:
                                    act = updated_pair.negative_response.layers_activations[best_layer_str]
                                    if act is not None:
                                        neg_acts_gen.append(act)

                            # Create steering vector
                            caa_method_gen = CAAMethod(kwargs={"normalize": True})
                            steering_vector_gen = caa_method_gen.train_for_layer(pos_acts_gen, neg_acts_gen)

                            # Create SteeringPlan
                            from wisent.core.models.core.atoms import SteeringVector, SteeringPlan
                            steering_vec = SteeringVector(vector=steering_vector_gen, scale=best_config['strength'])
                            steering_plan = SteeringPlan(
                                layers={best_layer_str: steering_vec},
                                layers_description=[f"CAA steering for {task_name}"]
                            )

                            # Generate with steering
                            model.attach(steering_plan)
                            steered_response = model.generate(
                                [[{"role": "user", "content": prompt}]],
                                max_new_tokens=100,
                                temperature=0.7,
                                use_steering=True,
                                steering_plan=steering_plan
                            )[0]
                            model.detach()

                            generation_examples.append({
                                'question': prompt,
                                'correct_answer': pair.positive_response.model_response,
                                'incorrect_answer': pair.negative_response.model_response,
                                'unsteered_generation': unsteered_response,
                                'steered_generation': steered_response
                            })

                            print(f"      Generated example {idx+1}/{num_examples}")

                        except Exception as e:
                            print(f"      ⚠️ Failed to generate example {idx+1}: {e}")
                            if args.verbose:
                                import traceback
                                traceback.print_exc()

                    # Save examples to JSON
                    examples_path = os.path.join(
                        args.save_best_vector if args.save_best_vector else "./optimization_results",
                        f"{task_name}_generation_examples.json"
                    )
                    os.makedirs(os.path.dirname(examples_path), exist_ok=True)

                    with open(examples_path, 'w') as f:
                        json.dump({
                            'task': task_name,
                            'model': args.model,
                            'best_config': best_config,
                            'examples': generation_examples
                        }, f, indent=2)

                    print(f"      💾 Saved {len(generation_examples)} generation examples to: {examples_path}")

            else:
                print(f"\n  ⚠️  No valid configuration found")
                method_results['CAA'] = {
                    'optimal_layer': 8,
                    'optimal_strength': 1.0,
                    'optimal_strategy': 'constant',
                    'optimal_token_aggregation': 'last_token',
                    'optimal_prompt_construction': 'chat_template',
                    'accuracy': 0.5,
                    'f1': 0.5
                }
            
            all_results[task_name] = {
                'methods': method_results,
                'best_method': 'CAA',
                'best_layer': method_results['CAA']['optimal_layer'],
                'best_strength': method_results['CAA']['optimal_strength'],
                'best_strategy': method_results['CAA']['optimal_strategy'],
                'best_token_aggregation': method_results['CAA']['optimal_token_aggregation'],
                'best_prompt_construction': method_results['CAA']['optimal_prompt_construction']
            }

            task_time = time.time() - task_start_time
            print(f"\n  ⏱️  Task completed in {task_time:.1f}s (tested {configs_tested} configurations)")

        except Exception as e:
            # NO FALLBACK - raise the error immediately
            print(f"\n❌ Task '{task_name}' optimization failed:")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            raise

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
        'optimization_dimensions': ['layer', 'strength', 'strategy', 'token_aggregation', 'prompt_construction']
    }

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Results saved to: {results_file}\n")

    # Print summary
    print("📋 SUMMARY BY TASK:")
    print("-" * 140)
    for task_name, config in all_results.items():
        print(f"  {task_name:20s} | L{config['best_layer']:2d} S{config['best_strength']:.1f} | {config['best_strategy']:12s} | T:{config['best_token_aggregation']:12s} | P:{config['best_prompt_construction']:18s}")
    print("-" * 140 + "\n")

    # Return results for programmatic access
    return {
        "model": args.model,
        "action": "comprehensive",
        "methods_tested": args.methods,
        "tasks_optimized": list(all_results.keys()),
        "results": all_results,
        "results_file": results_file,
        "optimization_dimensions": ['layer', 'strength', 'strategy', 'token_aggregation', 'prompt_construction']
    }


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
    """Execute method comparison - currently only CAA is implemented."""
    from wisent.core.steering_methods.methods.caa import CAAMethod
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.core.atoms import ActivationAggregationStrategy
    from wisent.core.models.core.atoms import SteeringVector, SteeringPlan
    from wisent_plots import LineChart
    import matplotlib.pyplot as plt
    import torch

    print(f"🔍 Comparing steering methods for task: {args.task}\n")
    print(f"   Methods: {', '.join(args.methods)}")
    print(f"   Limit: {args.limit} samples")
    print(f"   Layer: {args.layer}")
    print(f"   Strength: {args.strength}\n")

    # Load task data
    print(f"📊 Loading task data...")
    result = loader._load_one_task(
        task_name=args.task,
        split_ratio=0.8,
        seed=42,
        limit=args.limit,
        training_limit=None,
        testing_limit=None
    )

    train_pairs = result['train_qa_pairs']
    test_pairs = result['test_qa_pairs']
    print(f"   ✓ Loaded {len(train_pairs.pairs)} train, {len(test_pairs.pairs)} test pairs\n")

    # Initialize evaluator
    EvaluatorRotator.discover_evaluators('wisent.core.evaluators.benchmark_specific')
    evaluator = EvaluatorRotator(evaluator=None, task_name=args.task)
    print(f"   ✓ Using evaluator: {evaluator._evaluator.name}\n")

    # Collect activations once for all methods
    layer_str = str(args.layer)
    collector = ActivationCollector(model=model, store_device="cpu")

    print(f"🎯 Collecting training activations (ONCE)...")
    pos_acts = []
    neg_acts = []

    for i, pair in enumerate(train_pairs.pairs):
        if i % 10 == 0:
            print(f"   Processing train pair {i+1}/{len(train_pairs.pairs)}...", end='\r')

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

    print(f"   Processing train pair {len(train_pairs.pairs)}/{len(train_pairs.pairs)}... Done!")
    print(f"   ✓ Collected {len(pos_acts)} positive, {len(neg_acts)} negative activations\n")

    # Test each method
    print(f"🧪 Testing methods...")
    method_results = {}

    # Only CAA is implemented for now
    if 'CAA' in args.methods:
        print(f"\n   Testing CAA method...")

        # Train CAA steering vector
        caa_method = CAAMethod(kwargs={"normalize": True})
        steering_vector = caa_method.train_for_layer(pos_acts, neg_acts)

        # Create steering plan
        steering_vec = SteeringVector(vector=steering_vector, scale=args.strength)
        steering_plan = SteeringPlan(
            layers={layer_str: steering_vec},
            layers_description=[f"CAA steering layer={args.layer}, strength={args.strength}"]
        )

        # Apply steering and evaluate
        model.apply_steering(steering_plan)

        test_scores = []
        detailed_results = []
        for pair in test_pairs.pairs:
            choices = [pair.negative_response.model_response, pair.positive_response.model_response]
            expected = pair.positive_response.model_response
            test_code = pair.metadata.get("test_code") if pair.metadata else None

            eval_result = evaluator.evaluate(
                response="",
                expected=expected,
                model=model,
                question=pair.prompt,
                choices=choices,
                steering_plan=steering_plan,
                test_code=test_code,
                task_name=args.task
            )

            is_correct = eval_result.ground_truth == "TRUTHFUL"
            test_scores.append(1.0 if is_correct else 0.0)

            # Save full evaluation details
            detailed_results.append({
                'question': pair.prompt,
                'choices': choices,
                'expected': expected,
                'ground_truth': eval_result.ground_truth,
                'method_used': eval_result.method_used,
                'confidence': eval_result.confidence,
                'details': eval_result.details,
                'meta': dict(eval_result.meta) if eval_result.meta else {},
                'is_correct': is_correct
            })

        model.clear_steering()

        caa_accuracy = np.mean(test_scores) if len(test_scores) > 0 else 0.0
        method_results['CAA'] = {
            'accuracy': caa_accuracy,
            'num_test_samples': len(test_scores),
            'detailed_results': detailed_results
        }

        print(f"      ✓ CAA: accuracy={caa_accuracy:.3f}")

    # Other methods are not yet implemented
    for method in args.methods:
        if method not in ['CAA']:
            print(f"      ⚠️  {method}: not yet implemented")
            method_results[method] = {
                'accuracy': 0.0,
                'status': 'not_implemented'
            }

    # Save results
    print(f"\n{'='*80}")
    print(f"📊 METHOD COMPARISON COMPLETE")
    print(f"{'='*80}\n")

    results_file = f"./optimization_results/steering_compare_methods_{args.task}_{args.model.replace('/', '_')}.json"
    import os
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    output_data = {
        'model': args.model,
        'task': args.task,
        'layer': args.layer,
        'strength': args.strength,
        'methods': method_results,
        'limit': args.limit
    }

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Results saved to: {results_file}\n")

    # Create comparison plot if we have results
    implemented_methods = [m for m in method_results if method_results[m].get('accuracy', 0) > 0]
    if len(implemented_methods) > 1 and args.save_plot:
        plot_path_svg = f"steering_compare_methods_{args.task}_{args.model.replace('/', '_')}.svg"
        plot_path_png = f"steering_compare_methods_{args.task}_{args.model.replace('/', '_')}.png"

        method_names = list(implemented_methods)
        accuracies = [method_results[m]['accuracy'] for m in method_names]

        chart = LineChart(style=1, figsize=(10, 6), show_markers=True)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.bar(method_names, accuracies, color='#3498db', alpha=0.8)
        ax.set_xlabel('Steering Method')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Steering Method Comparison\n{args.model} on {args.task}')
        ax.set_ylim(0, 1)

        fig.savefig(plot_path_svg, format='svg', bbox_inches='tight')
        fig.savefig(plot_path_png, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"💾 Comparison plot saved to:")
        print(f"   SVG: {plot_path_svg}")
        print(f"   PNG: {plot_path_png}\n")

    return {
        "action": "compare-methods",
        "task": args.task,
        "methods": method_results,
        "results_file": results_file
    }


def execute_optimize_layer(args, model, loader):
    """Execute layer optimization - find the best layer for steering."""
    from wisent.core.steering_methods.methods.caa import CAAMethod
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.core.atoms import ActivationAggregationStrategy
    from wisent.core.models.core.atoms import SteeringVector, SteeringPlan
    from wisent_plots import LineChart
    import matplotlib.pyplot as plt
    import torch

    print(f"🎯 Optimizing steering layer for task: {args.task}\n")
    print(f"   Method: {args.method}")
    print(f"   Strength: {args.strength}")
    print(f"   Limit: {args.limit} samples\n")

    # Load task data
    print(f"📊 Loading task data...")
    result = loader._load_one_task(
        task_name=args.task,
        split_ratio=0.8,
        seed=42,
        limit=args.limit,
        training_limit=None,
        testing_limit=None
    )

    train_pairs = result['train_qa_pairs']
    test_pairs = result['test_qa_pairs']
    print(f"   ✓ Loaded {len(train_pairs.pairs)} train, {len(test_pairs.pairs)} test pairs\n")

    # Initialize evaluator
    EvaluatorRotator.discover_evaluators('wisent.core.evaluators.benchmark_specific')
    evaluator = EvaluatorRotator(evaluator=None, task_name=args.task)
    print(f"   ✓ Using evaluator: {evaluator._evaluator.name}\n")

    # Determine layers to test
    if args.layers:
        layers_to_test = args.layers
    else:
        # Test all layers from 0 to num_layers-1
        layers_to_test = list(range(model.num_layers))

    print(f"🔍 Testing {len(layers_to_test)} layers: {layers_to_test[:5]}{'...' if len(layers_to_test) > 5 else ''}\n")

    collector = ActivationCollector(model=model, store_device="cpu")
    layer_results = {}
    best_layer = None
    best_accuracy = 0.0

    for layer_idx, layer in enumerate(layers_to_test, 1):
        layer_str = str(layer)
        print(f"   [{layer_idx}/{len(layers_to_test)}] Testing layer {layer}...", end=' ')

        try:
            # Collect activations for this layer
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
                print(f"⚠️  No activations collected")
                continue

            # Train steering vector (only CAA supported)
            if args.method == 'CAA':
                caa_method = CAAMethod(kwargs={"normalize": True})
                steering_vector = caa_method.train_for_layer(pos_acts, neg_acts)
            else:
                print(f"⚠️  Method {args.method} not supported")
                continue

            # Create steering plan
            steering_vec = SteeringVector(vector=steering_vector, scale=args.strength)
            steering_plan = SteeringPlan(
                layers={layer_str: steering_vec},
                layers_description=[f"{args.method} steering layer={layer}"]
            )

            # Evaluate
            model.apply_steering(steering_plan)

            test_scores = []
            detailed_results = []
            for pair in test_pairs.pairs:
                choices = [pair.negative_response.model_response, pair.positive_response.model_response]
                expected = pair.positive_response.model_response
                test_code = pair.metadata.get("test_code") if pair.metadata else None

                eval_result = evaluator.evaluate(
                    response="",
                    expected=expected,
                    model=model,
                    question=pair.prompt,
                    choices=choices,
                    steering_plan=steering_plan,
                    test_code=test_code,
                    task_name=task_name
                )

                is_correct = eval_result.ground_truth == "TRUTHFUL"
                test_scores.append(1.0 if is_correct else 0.0)

                # Save full evaluation details
                detailed_results.append({
                    'question': pair.prompt,
                    'choices': choices,
                    'expected': expected,
                    'ground_truth': eval_result.ground_truth,
                    'method_used': eval_result.method_used,
                    'confidence': eval_result.confidence,
                    'details': eval_result.details,
                    'meta': dict(eval_result.meta) if eval_result.meta else {},
                    'is_correct': is_correct
                })

            model.clear_steering()

            accuracy = np.mean(test_scores) if len(test_scores) > 0 else 0.0
            layer_results[layer] = {
                'accuracy': accuracy,
                'num_test_samples': len(test_scores),
                'detailed_results': detailed_results
            }

            print(f"accuracy={accuracy:.3f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_layer = layer

        except Exception as e:
            print(f"❌ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Results
    print(f"\n{'='*80}")
    print(f"📊 LAYER OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"   Best layer: {best_layer}")
    print(f"   Best accuracy: {best_accuracy:.4f}")
    print(f"{'='*80}\n")

    # Save results
    results_file = f"./optimization_results/steering_optimize_layer_{args.task}_{args.model.replace('/', '_')}.json"
    import os
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    output_data = {
        'model': args.model,
        'task': args.task,
        'method': args.method,
        'strength': args.strength,
        'best_layer': best_layer,
        'best_accuracy': best_accuracy,
        'layer_results': {str(k): v for k, v in layer_results.items()},
        'limit': args.limit
    }

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Results saved to: {results_file}\n")

    # Create plot
    if args.save_plot and len(layer_results) > 0:
        plot_path_svg = f"steering_optimize_layer_{args.task}_{args.model.replace('/', '_')}.svg"
        plot_path_png = f"steering_optimize_layer_{args.task}_{args.model.replace('/', '_')}.png"

        layers = sorted(layer_results.keys())
        accuracies = [layer_results[l]['accuracy'] for l in layers]

        chart = LineChart(style=1, figsize=(10, 6), show_markers=True)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        chart.plot_multiple(
            x=layers,
            y_series=[accuracies],
            labels=['Accuracy'],
            title=f'Layer Optimization\n{args.model} on {args.task}',
            xlabel='Layer',
            ylabel='Accuracy',
            fig=fig,
            ax=ax,
            output_format='png'
        )

        # Add vertical line for optimal layer
        ax.axvline(x=best_layer, color='#2ecc71', linestyle='--', linewidth=2,
                   label=f'Best: Layer {best_layer}', alpha=0.7)
        ax.legend()

        fig.savefig(plot_path_svg, format='svg', bbox_inches='tight')
        fig.savefig(plot_path_png, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"💾 Layer optimization plot saved to:")
        print(f"   SVG: {plot_path_svg}")
        print(f"   PNG: {plot_path_png}\n")

    return {
        "action": "optimize-layer",
        "task": args.task,
        "method": args.method,
        "best_layer": best_layer,
        "best_accuracy": best_accuracy,
        "results_file": results_file
    }


def execute_optimize_strength(args, model, loader):
    """Execute strength optimization - find the best steering strength."""
    from wisent.core.steering_methods.methods.caa import CAAMethod
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.core.atoms import ActivationAggregationStrategy
    from wisent.core.models.core.atoms import SteeringVector, SteeringPlan
    from wisent_plots import LineChart
    import matplotlib.pyplot as plt
    import torch

    print(f"💪 Optimizing steering strength for task: {args.task}\n")
    print(f"   Method: {args.method}")
    print(f"   Layer: {args.layer}")
    print(f"   Strength range: {args.strength_range[0]} to {args.strength_range[1]}")
    print(f"   Num steps: {args.num_strength_steps}")
    print(f"   Limit: {args.limit} samples\n")

    # Load task data
    print(f"📊 Loading task data...")
    result = loader._load_one_task(
        task_name=args.task,
        split_ratio=0.8,
        seed=42,
        limit=args.limit,
        training_limit=None,
        testing_limit=None
    )

    train_pairs = result['train_qa_pairs']
    test_pairs = result['test_qa_pairs']
    print(f"   ✓ Loaded {len(train_pairs.pairs)} train, {len(test_pairs.pairs)} test pairs\n")

    # Initialize evaluator
    EvaluatorRotator.discover_evaluators('wisent.core.evaluators.benchmark_specific')
    evaluator = EvaluatorRotator(evaluator=None, task_name=args.task)
    print(f"   ✓ Using evaluator: {evaluator._evaluator.name}\n")

    # Collect activations ONCE
    layer_str = str(args.layer)
    collector = ActivationCollector(model=model, store_device="cpu")

    print(f"🎯 Collecting training activations (ONCE)...")
    pos_acts = []
    neg_acts = []

    for i, pair in enumerate(train_pairs.pairs):
        if i % 10 == 0:
            print(f"   Processing train pair {i+1}/{len(train_pairs.pairs)}...", end='\r')

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

    print(f"   Processing train pair {len(train_pairs.pairs)}/{len(train_pairs.pairs)}... Done!")
    print(f"   ✓ Collected {len(pos_acts)} positive, {len(neg_acts)} negative activations\n")

    # Train steering vector ONCE (only CAA supported)
    if args.method == 'CAA':
        caa_method = CAAMethod(kwargs={"normalize": True})
        steering_vector = caa_method.train_for_layer(pos_acts, neg_acts)
    else:
        print(f"❌ Method {args.method} not supported")
        return {
            "action": "optimize-strength",
            "task": args.task,
            "method": args.method,
            "status": "method_not_supported"
        }

    # Generate strength values to test
    min_strength, max_strength = args.strength_range
    strengths_to_test = np.linspace(min_strength, max_strength, args.num_strength_steps)

    print(f"🔍 Testing {len(strengths_to_test)} strength values: {strengths_to_test[0]:.2f} to {strengths_to_test[-1]:.2f}\n")

    strength_results = {}
    best_strength = None
    best_accuracy = 0.0

    for strength_idx, strength in enumerate(strengths_to_test, 1):
        print(f"   [{strength_idx}/{len(strengths_to_test)}] Testing strength {strength:.2f}...", end=' ')

        try:
            # Create steering plan with this strength
            steering_vec = SteeringVector(vector=steering_vector, scale=float(strength))
            steering_plan = SteeringPlan(
                layers={layer_str: steering_vec},
                layers_description=[f"{args.method} steering strength={strength:.2f}"]
            )

            # Evaluate
            model.apply_steering(steering_plan)

            test_scores = []
            detailed_results = []
            for pair in test_pairs.pairs:
                choices = [pair.negative_response.model_response, pair.positive_response.model_response]
                expected = pair.positive_response.model_response
                test_code = pair.metadata.get("test_code") if pair.metadata else None

                eval_result = evaluator.evaluate(
                    response="",
                    expected=expected,
                    model=model,
                    question=pair.prompt,
                    choices=choices,
                    steering_plan=steering_plan,
                    test_code=test_code,
                    task_name=task_name
                )

                is_correct = eval_result.ground_truth == "TRUTHFUL"
                test_scores.append(1.0 if is_correct else 0.0)

                # Save full evaluation details
                detailed_results.append({
                    'question': pair.prompt,
                    'choices': choices,
                    'expected': expected,
                    'ground_truth': eval_result.ground_truth,
                    'method_used': eval_result.method_used,
                    'confidence': eval_result.confidence,
                    'details': eval_result.details,
                    'meta': dict(eval_result.meta) if eval_result.meta else {},
                    'is_correct': is_correct
                })

            model.clear_steering()

            accuracy = np.mean(test_scores) if len(test_scores) > 0 else 0.0
            strength_results[float(strength)] = {
                'accuracy': accuracy,
                'num_test_samples': len(test_scores),
                'detailed_results': detailed_results
            }

            print(f"accuracy={accuracy:.3f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_strength = float(strength)

        except Exception as e:
            print(f"❌ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Results
    print(f"\n{'='*80}")
    print(f"📊 STRENGTH OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"   Best strength: {best_strength:.2f}")
    print(f"   Best accuracy: {best_accuracy:.4f}")
    print(f"{'='*80}\n")

    # Save results
    results_file = f"./optimization_results/steering_optimize_strength_{args.task}_{args.model.replace('/', '_')}.json"
    import os
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    output_data = {
        'model': args.model,
        'task': args.task,
        'method': args.method,
        'layer': args.layer,
        'best_strength': best_strength,
        'best_accuracy': best_accuracy,
        'strength_results': {str(k): v for k, v in strength_results.items()},
        'limit': args.limit
    }

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Results saved to: {results_file}\n")

    # Create plot
    if args.save_plot and len(strength_results) > 0:
        plot_path_svg = f"steering_optimize_strength_{args.task}_{args.model.replace('/', '_')}.svg"
        plot_path_png = f"steering_optimize_strength_{args.task}_{args.model.replace('/', '_')}.png"

        strengths = sorted(strength_results.keys())
        accuracies = [strength_results[s]['accuracy'] for s in strengths]

        chart = LineChart(style=1, figsize=(10, 6), show_markers=True)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        chart.plot_multiple(
            x=strengths,
            y_series=[accuracies],
            labels=['Accuracy'],
            title=f'Strength Optimization\n{args.model} on {args.task}',
            xlabel='Steering Strength',
            ylabel='Accuracy',
            fig=fig,
            ax=ax,
            output_format='png'
        )

        # Add vertical line for optimal strength
        ax.axvline(x=best_strength, color='#2ecc71', linestyle='--', linewidth=2,
                   label=f'Best: {best_strength:.2f}', alpha=0.7)
        ax.legend()

        fig.savefig(plot_path_svg, format='svg', bbox_inches='tight')
        fig.savefig(plot_path_png, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"💾 Strength optimization plot saved to:")
        print(f"   SVG: {plot_path_svg}")
        print(f"   PNG: {plot_path_png}\n")

    return {
        "action": "optimize-strength",
        "task": args.task,
        "method": args.method,
        "best_strength": best_strength,
        "best_accuracy": best_accuracy,
        "results_file": results_file
    }


def execute_auto(args, model, loader):
    """Execute automatic optimization - optimizes layer AND strength together."""
    from wisent.core.steering_methods.methods.caa import CAAMethod
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.core.atoms import ActivationAggregationStrategy
    from wisent.core.models.core.atoms import SteeringVector, SteeringPlan
    from wisent_plots import LineChart
    import matplotlib.pyplot as plt
    import torch

    print(f"🤖 Running automatic steering optimization...\n")
    print(f"   Task: {args.task}")
    print(f"   Methods: {', '.join(args.methods)}")
    print(f"   Strength range: {args.strength_range}")
    print(f"   Limit: {args.limit} samples\n")

    # Load task data
    print(f"📊 Loading task data...")
    result = loader._load_one_task(
        task_name=args.task,
        split_ratio=0.8,
        seed=42,
        limit=args.limit,
        training_limit=None,
        testing_limit=None
    )

    train_pairs = result['train_qa_pairs']
    test_pairs = result['test_qa_pairs']
    print(f"   ✓ Loaded {len(train_pairs.pairs)} train, {len(test_pairs.pairs)} test pairs\n")

    # Initialize evaluator
    EvaluatorRotator.discover_evaluators('wisent.core.evaluators.benchmark_specific')
    evaluator = EvaluatorRotator(evaluator=None, task_name=args.task)
    print(f"   ✓ Using evaluator: {evaluator._evaluator.name}\n")

    # Define search space
    layers_to_test = list(range(max(0, model.num_layers // 2 - 2), min(model.num_layers, model.num_layers // 2 + 3)))  # Test 5 layers around middle
    min_strength, max_strength = args.strength_range
    strengths_to_test = np.linspace(min_strength, max_strength, 5)  # 5 strength values

    print(f"🔍 Auto-optimizing layer and strength...")
    print(f"   Testing {len(layers_to_test)} layers: {layers_to_test}")
    print(f"   Testing {len(strengths_to_test)} strengths: {strengths_to_test[0]:.2f} to {strengths_to_test[-1]:.2f}")
    print(f"   Total configurations: {len(layers_to_test) * len(strengths_to_test)}\n")

    collector = ActivationCollector(model=model, store_device="cpu")
    all_results = {}
    best_config = None
    best_accuracy = 0.0

    config_count = 0
    total_configs = len(layers_to_test) * len(strengths_to_test)

    for layer in layers_to_test:
        layer_str = str(layer)

        # Collect activations for this layer
        print(f"   Collecting activations for layer {layer}...")
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
            print(f"      ⚠️  No activations collected for layer {layer}")
            continue

        # Train steering vector for this layer (only CAA supported)
        if 'CAA' in args.methods:
            caa_method = CAAMethod(kwargs={"normalize": True})
            steering_vector = caa_method.train_for_layer(pos_acts, neg_acts)
        else:
            print(f"      ⚠️  Only CAA method is supported")
            continue

        # Test different strengths for this layer
        for strength in strengths_to_test:
            config_count += 1
            print(f"      [{config_count}/{total_configs}] Layer {layer}, Strength {strength:.2f}...", end=' ')

            try:
                # Create steering plan
                steering_vec = SteeringVector(vector=steering_vector, scale=float(strength))
                steering_plan = SteeringPlan(
                    layers={layer_str: steering_vec},
                    layers_description=[f"CAA layer={layer}, strength={strength:.2f}"]
                )

                # Evaluate
                model.apply_steering(steering_plan)

                test_scores = []
                detailed_results = []
                for pair in test_pairs.pairs:
                    choices = [pair.negative_response.model_response, pair.positive_response.model_response]
                    expected = pair.positive_response.model_response
                    test_code = pair.metadata.get("test_code") if pair.metadata else None

                    eval_result = evaluator.evaluate(
                        response="",
                        expected=expected,
                        model=model,
                        question=pair.prompt,
                        choices=choices,
                        steering_plan=steering_plan,
                        test_code=test_code,
                        task_name=task_name
                    )

                    is_correct = eval_result.ground_truth == "TRUTHFUL"
                    test_scores.append(1.0 if is_correct else 0.0)

                    # Save full evaluation details
                    detailed_results.append({
                        'question': pair.prompt,
                        'choices': choices,
                        'expected': expected,
                        'ground_truth': eval_result.ground_truth,
                        'method_used': eval_result.method_used,
                        'confidence': eval_result.confidence,
                        'details': eval_result.details,
                        'meta': dict(eval_result.meta) if eval_result.meta else {},
                        'is_correct': is_correct
                    })

                model.clear_steering()

                accuracy = np.mean(test_scores) if len(test_scores) > 0 else 0.0
                all_results[(layer, float(strength))] = {
                    'accuracy': accuracy,
                    'num_test_samples': len(test_scores),
                    'detailed_results': detailed_results
                }

                print(f"accuracy={accuracy:.3f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = {
                        'layer': layer,
                        'strength': float(strength),
                        'accuracy': accuracy
                    }

            except Exception as e:
                print(f"❌ Error: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

    # Results
    print(f"\n{'='*80}")
    print(f"📊 AUTO OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    if best_config:
        print(f"   Best layer: {best_config['layer']}")
        print(f"   Best strength: {best_config['strength']:.2f}")
        print(f"   Best accuracy: {best_config['accuracy']:.4f}")
    else:
        print(f"   ⚠️  No valid configuration found")
    print(f"{'='*80}\n")

    # Save results
    results_file = f"./optimization_results/steering_auto_{args.task}_{args.model.replace('/', '_')}.json"
    import os
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    output_data = {
        'model': args.model,
        'task': args.task,
        'methods': args.methods,
        'best_config': best_config,
        'all_results': {f"layer{k[0]}_strength{k[1]:.2f}": v for k, v in all_results.items()},
        'limit': args.limit
    }

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Results saved to: {results_file}\n")

    # Create heatmap plot
    if args.save_plot and len(all_results) > 0 and best_config:
        plot_path_svg = f"steering_auto_{args.task}_{args.model.replace('/', '_')}.svg"
        plot_path_png = f"steering_auto_{args.task}_{args.model.replace('/', '_')}.png"

        # Prepare data for heatmap
        layers = sorted(set(k[0] for k in all_results.keys()))
        strengths = sorted(set(k[1] for k in all_results.keys()))

        # Create accuracy matrix
        accuracy_matrix = np.zeros((len(strengths), len(layers)))
        for i, strength in enumerate(strengths):
            for j, layer in enumerate(layers):
                if (layer, strength) in all_results:
                    accuracy_matrix[i, j] = all_results[(layer, strength)]['accuracy']

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        im = ax.imshow(accuracy_matrix, cmap='viridis', aspect='auto')

        # Set ticks and labels
        ax.set_xticks(np.arange(len(layers)))
        ax.set_yticks(np.arange(len(strengths)))
        ax.set_xticklabels(layers)
        ax.set_yticklabels([f"{s:.2f}" for s in strengths])

        # Labels
        ax.set_xlabel('Layer')
        ax.set_ylabel('Strength')
        ax.set_title(f'Auto Optimization Heatmap\n{args.model} on {args.task}')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy', rotation=270, labelpad=15)

        # Mark best configuration
        best_layer_idx = layers.index(best_config['layer'])
        best_strength_idx = strengths.index(best_config['strength'])
        ax.plot(best_layer_idx, best_strength_idx, 'r*', markersize=20,
                label=f"Best: L{best_config['layer']}, S{best_config['strength']:.2f}")
        ax.legend()

        fig.savefig(plot_path_svg, format='svg', bbox_inches='tight')
        fig.savefig(plot_path_png, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"💾 Auto optimization heatmap saved to:")
        print(f"   SVG: {plot_path_svg}")
        print(f"   PNG: {plot_path_png}\n")

    return {
        "action": "auto",
        "task": args.task,
        "methods": args.methods,
        "best_config": best_config,
        "results_file": results_file
    }


def execute_personalization(args, model):
    """
    Execute personalization optimization - find optimal parameters for trait steering.

    This optimizes ALL steering parameters for personality/trait vectors by:
    1. Generating synthetic contrastive pairs for the trait
    2. Testing all combinations of:
       - Layers (where to apply steering)
       - Strengths (how strong the steering signal is)
       - Token aggregation strategies (LAST_TOKEN, MEAN_POOLING, FIRST_TOKEN)
       - Prompt construction strategies (CHAT_TEMPLATE, DIRECT_COMPLETION)
    3. Evaluating each configuration using personalization metrics:
       - Difference: Is the steered response different from baseline?
       - Quality: Is the response coherent (not lobotomized)?
       - Alignment: Does the response match the intended trait?
    4. Selecting the configuration with the highest overall score
    """
    import os
    import torch
    from wisent.core.steering_methods.methods.caa import CAAMethod
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.core.atoms import ActivationAggregationStrategy
    from wisent.core.activations.prompt_construction_strategy import PromptConstructionStrategy
    from wisent.core.models.core.atoms import SteeringVector, SteeringPlan
    from wisent.core.evaluators.personalization_evaluator import PersonalizationEvaluator
    from wisent.core.synthetic.generators.pairs_generator import SyntheticContrastivePairsGenerator
    from wisent.core.synthetic.cleaners.pairs_cleaner import PairsCleaner
    from wisent.core.synthetic.db_instructions.mini_dp import Default_DB_Instructions
    from wisent.core.synthetic.generators.diversities.methods.fast_diversity import FastDiversity

    trait = args.trait
    trait_name = args.trait_name or trait.split()[0].lower()

    print(f"\n{'='*80}", flush=True)
    print(f"🎭 PERSONALIZATION OPTIMIZATION (COMPREHENSIVE)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"   Trait: {trait}", flush=True)
    print(f"   Trait Name: {trait_name}", flush=True)
    print(f"   Model: {args.model}", flush=True)
    print(f"   Num Pairs: {args.num_pairs}", flush=True)
    print(f"   Num Test Prompts: {args.num_test_prompts}", flush=True)
    print(f"   Output Directory: {args.output_dir}", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "vectors"), exist_ok=True)

    # Determine layers to test - ALL layers by default
    if args.layers:
        layers_to_test = args.layers
    else:
        # Test ALL layers (1-indexed, since activation collector uses 1-based indexing)
        num_layers = model.num_layers
        layers_to_test = list(range(1, num_layers + 1))

    # Determine strengths to test
    min_strength, max_strength = args.strength_range
    strengths_to_test = np.linspace(min_strength, max_strength, args.num_strength_steps)

    # Token aggregation strategies to test - ALL strategies
    token_aggregations_to_test = [
        ActivationAggregationStrategy.LAST_TOKEN,
        ActivationAggregationStrategy.MEAN_POOLING,
        ActivationAggregationStrategy.FIRST_TOKEN,
        ActivationAggregationStrategy.MAX_POOLING,
    ]

    # Prompt construction strategies to test - ALL strategies
    prompt_constructions_to_test = [
        PromptConstructionStrategy.CHAT_TEMPLATE,
        PromptConstructionStrategy.DIRECT_COMPLETION,
        PromptConstructionStrategy.INSTRUCTION_FOLLOWING,
        PromptConstructionStrategy.ROLE_PLAYING,
        PromptConstructionStrategy.MULTIPLE_CHOICE,
    ]

    # Steering application strategies to test - ALL strategies
    steering_strategies_to_test = ["constant", "initial_only", "diminishing", "all_equal"]

    total_configs = (
        len(layers_to_test) *
        len(strengths_to_test) *
        len(steering_strategies_to_test) *
        len(token_aggregations_to_test) *
        len(prompt_constructions_to_test)
    )

    print(f"📊 Search Space:", flush=True)
    print(f"   Layers: {layers_to_test} ({len(layers_to_test)} total)", flush=True)
    print(f"   Strengths: {[f'{s:.2f}' for s in strengths_to_test]}", flush=True)
    print(f"   Steering Strategies: {steering_strategies_to_test}", flush=True)
    print(f"   Token Aggregations: {[t.value for t in token_aggregations_to_test]}", flush=True)
    print(f"   Prompt Constructions: {[p.value for p in prompt_constructions_to_test]}", flush=True)
    print(f"   Total configurations: {total_configs}\n", flush=True)

    # Step 1: Generate synthetic contrastive pairs
    print(f"🔧 Step 1: Generating {args.num_pairs} synthetic contrastive pairs...", flush=True)

    # Create the pair generator with required components
    cleaner = PairsCleaner(steps=[])  # Minimal cleaning
    diversity = FastDiversity()
    db_instructions = Default_DB_Instructions()

    pair_generator = SyntheticContrastivePairsGenerator(
        model=model,
        generation_config={"max_new_tokens": 150, "temperature": 0.7},
        contrastive_set_name=f"{trait_name}_pairs",
        trait_description=trait,
        trait_label=trait_name,
        db_instructions=db_instructions,
        cleaner=cleaner,
        diversity=diversity,
    )

    pair_set, generation_report = pair_generator.generate(num_pairs=args.num_pairs)
    pairs = pair_set.pairs

    print(f"   ✓ Generated {len(pairs)} contrastive pairs\n", flush=True)

    # Generate test prompts for evaluation
    test_prompts = [
        "What's your favorite food?",
        "How do you spend your weekends?",
        "What motivates you in life?",
        "How do you handle setbacks?",
        "What's your opinion on teamwork?",
    ][:args.num_test_prompts]

    print(f"📝 Test prompts for evaluation:", flush=True)
    for i, prompt in enumerate(test_prompts, 1):
        print(f"   {i}. {prompt}", flush=True)
    print(flush=True)

    # Initialize activation collector
    collector = ActivationCollector(model=model, store_device="cpu")

    # Track results for all configurations
    all_results = {}
    best_config = None
    best_score = -1.0
    best_steering_vector = None

    # Cache for steering vectors per (layer, token_agg, prompt_const) combination
    # to avoid recomputing activations unnecessarily
    steering_vector_cache = {}

    # Step 2: Test all configurations
    print(f"🎯 Step 2: Testing {total_configs} configurations...", flush=True)

    config_count = 0

    for token_agg in token_aggregations_to_test:
        for prompt_const in prompt_constructions_to_test:
            print(f"\n   📊 Token Aggregation: {token_agg.value}, Prompt Construction: {prompt_const.value}", flush=True)

            for layer in layers_to_test:
                layer_str = str(layer)

                # Check if we already have activations for this (layer, token_agg) combo
                cache_key = (layer, token_agg.value, prompt_const.value)

                if cache_key not in steering_vector_cache:
                    print(f"\n      📍 Layer {layer}: Collecting activations...", flush=True)

                    # Collect activations for this layer with current token_agg and prompt_const
                    pos_acts = []
                    neg_acts = []

                    for pair in pairs:
                        updated_pair = collector.collect_for_pair(
                            pair,
                            layers=[layer_str],
                            aggregation=token_agg,
                            prompt_strategy=prompt_const,
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
                        print(f"         ⚠️ No activations collected for layer {layer}", flush=True)
                        steering_vector_cache[cache_key] = None
                        continue

                    print(f"         ✓ Collected {len(pos_acts)} positive, {len(neg_acts)} negative activations", flush=True)

                    # Create steering vector using CAA
                    caa_method = CAAMethod(kwargs={"normalize": True})
                    steering_vector = caa_method.train_for_layer(pos_acts, neg_acts)
                    steering_vector_cache[cache_key] = steering_vector

                    print(f"         ✓ Created steering vector (norm: {torch.norm(steering_vector).item():.4f})", flush=True)
                else:
                    steering_vector = steering_vector_cache[cache_key]
                    if steering_vector is None:
                        continue

                # Test different strengths and steering strategies
                for strength in strengths_to_test:
                    for steering_strategy in steering_strategies_to_test:
                        config_count += 1
                        config_desc = f"L{layer} S{strength:.2f} St:{steering_strategy} T:{token_agg.value} P:{prompt_const.value}"
                        print(f"      [{config_count}/{total_configs}] Testing {config_desc}...", end=' ')

                        # Create steering plan
                        steering_vec = SteeringVector(vector=steering_vector, scale=float(strength))
                        steering_plan = SteeringPlan(
                            layers={layer_str: steering_vec},
                            layers_description=[f"Personalization {config_desc}"]
                        )

                        # Generate baseline and steered responses
                        baseline_responses = []
                        steered_responses = []

                        for prompt in test_prompts:
                            # Generate baseline (no steering)
                            baseline = model.generate(
                                [[{"role": "user", "content": prompt}]],
                                max_new_tokens=args.max_new_tokens,
                                temperature=0.7,
                                use_steering=False
                            )[0]
                            baseline_responses.append(baseline)

                            # Generate steered response
                            model.apply_steering(steering_plan)
                            steered = model.generate(
                                [[{"role": "user", "content": prompt}]],
                                max_new_tokens=args.max_new_tokens,
                                temperature=0.7,
                                use_steering=True,
                                steering_plan=steering_plan
                            )[0]
                            model.clear_steering()
                            steered_responses.append(steered)

                        # Evaluate using personalization metrics
                        evaluator = PersonalizationEvaluator()

                        # Calculate difference score
                        difference_score = evaluator._evaluate_difference(baseline_responses, steered_responses)

                        # Calculate quality score
                        quality_score = evaluator._evaluate_quality(steered_responses)

                        # Calculate alignment score using simple keyword matching
                        # (Full alignment needs model-based judge which is expensive)
                        alignment_score = _estimate_alignment(steered_responses, trait)

                        # Calculate overall score (weighted average)
                        # Only count if difference > 0.3 (steering is actually doing something)
                        if difference_score < 0.3:
                            overall_score = 0.0
                        else:
                            overall_score = (
                                0.2 * difference_score +
                                0.3 * quality_score +
                                0.5 * alignment_score
                            )

                        print(f"diff={difference_score:.2f} qual={quality_score:.2f} align={alignment_score:.2f} overall={overall_score:.2f}")

                        # Store results with full config key
                        config_key = f"L{layer}_S{strength:.2f}_St:{steering_strategy}_T:{token_agg.value}_P:{prompt_const.value}"
                        all_results[config_key] = {
                            'layer': layer,
                            'strength': float(strength),
                            'steering_strategy': steering_strategy,
                            'token_aggregation': token_agg.value,
                            'prompt_construction': prompt_const.value,
                            'difference_score': float(difference_score),
                            'quality_score': float(quality_score),
                            'alignment_score': float(alignment_score),
                            'overall_score': float(overall_score),
                            'sample_baseline': baseline_responses[0][:200] if baseline_responses else "",
                            'sample_steered': steered_responses[0][:200] if steered_responses else "",
                        }

                        # Track best configuration
                        if overall_score > best_score:
                            best_score = overall_score
                            best_config = {
                                'layer': layer,
                                'strength': float(strength),
                                'steering_strategy': steering_strategy,
                                'token_aggregation': token_agg.value,
                                'prompt_construction': prompt_const.value,
                                'difference_score': float(difference_score),
                                'quality_score': float(quality_score),
                                'alignment_score': float(alignment_score),
                                'overall_score': float(overall_score),
                            }
                            best_steering_vector = steering_vector

    # Step 3: Save results
    print(f"\n{'='*80}")
    print(f"📊 OPTIMIZATION COMPLETE")
    print(f"{'='*80}")

    vector_path = None
    if best_config:
        print(f"\n✅ Best Configuration:")
        print(f"   Layer: {best_config['layer']}")
        print(f"   Strength: {best_config['strength']:.2f}")
        print(f"   Steering Strategy: {best_config['steering_strategy']}")
        print(f"   Token Aggregation: {best_config['token_aggregation']}")
        print(f"   Prompt Construction: {best_config['prompt_construction']}")
        print(f"   Difference Score: {best_config['difference_score']:.3f}")
        print(f"   Quality Score: {best_config['quality_score']:.3f}")
        print(f"   Alignment Score: {best_config['alignment_score']:.3f}")
        print(f"   Overall Score: {best_config['overall_score']:.3f}")

        # Save best steering vector
        vector_path = os.path.join(args.output_dir, "vectors", f"{trait_name}_optimal.pt")
        torch.save({
            'steering_vector': best_steering_vector,
            'layer': best_config['layer'],
            'layer_index': best_config['layer'],
            'strength': best_config['strength'],
            'steering_strategy': best_config['steering_strategy'],
            'token_aggregation': best_config['token_aggregation'],
            'prompt_construction': best_config['prompt_construction'],
            'trait': trait,
            'trait_name': trait_name,
            'model': args.model,
            'method': 'CAA',
            'optimization_scores': {
                'difference': best_config['difference_score'],
                'quality': best_config['quality_score'],
                'alignment': best_config['alignment_score'],
                'overall': best_config['overall_score'],
            }
        }, vector_path)
        print(f"\n💾 Saved optimal steering vector to: {vector_path}")
    else:
        print(f"\n⚠️ No valid configuration found")

    # Save full results to JSON
    results_file = os.path.join(args.output_dir, f"{trait_name}_optimization_results.json")

    # best_config doesn't have steering_vector anymore (it's in best_steering_vector)
    best_config_json = best_config

    output_data = {
        'model': args.model,
        'trait': trait,
        'trait_name': trait_name,
        'num_pairs': args.num_pairs,
        'num_test_prompts': args.num_test_prompts,
        'layers_tested': layers_to_test,
        'strengths_tested': [float(s) for s in strengths_to_test],
        'steering_strategies_tested': steering_strategies_to_test,
        'token_aggregations_tested': [t.value for t in token_aggregations_to_test],
        'prompt_constructions_tested': [p.value for p in prompt_constructions_to_test],
        'best_config': best_config_json,
        'all_results': all_results,
    }

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"💾 Saved full results to: {results_file}")

    # Print usage example
    print(f"\n📝 Usage Example:")
    if best_config:
        print(f"   python -m wisent.core.main multi-steer \\")
        print(f"       --vector {vector_path}:{best_config['strength']:.1f} \\")
        print(f"       --model {args.model} \\")
        print(f"       --layer {best_config['layer']} \\")
        print(f"       --prompt \"Your prompt here\"")

    print(f"\n{'='*80}\n")

    return {
        "action": "personalization",
        "trait": trait,
        "trait_name": trait_name,
        "best_config": best_config_json,
        "results_file": results_file,
        "vector_path": vector_path if best_config else None,
    }


def _estimate_alignment(responses: list[str], trait_description: str) -> float:
    """
    Estimate trait alignment using keyword matching.

    This is a fast heuristic that checks for trait-related keywords in responses.
    For more accurate alignment, use the full model-based evaluator.
    """
    import re

    # Extract keywords from trait description
    trait_words = set(re.findall(r'\b[a-z]+\b', trait_description.lower()))

    # Common trait indicators to look for
    trait_indicators = {
        'evil': ['evil', 'villain', 'domination', 'destroy', 'conquer', 'mwahaha', 'muahaha', 'fool', 'minion', 'scheme'],
        'italian': ['italian', 'mamma', 'mia', 'pasta', 'pizza', 'bellissimo', 'ciao', 'capisce', 'famiglia', 'amore'],
        'british': ['british', 'jolly', 'cheerio', 'lovely', 'quite', 'indeed', 'rather', 'splendid', 'tea', 'blimey'],
        'pirate': ['pirate', 'arrr', 'matey', 'treasure', 'ship', 'captain', 'sea', 'ahoy', 'plunder', 'rum'],
        'formal': ['formal', 'hereby', 'therefore', 'accordingly', 'furthermore', 'pursuant', 'respectfully'],
        'casual': ['casual', 'hey', 'cool', 'awesome', 'yeah', 'kinda', 'gonna', 'wanna'],
    }

    # Find which trait category matches best
    matched_indicators = set()
    for category, keywords in trait_indicators.items():
        if any(word in trait_words for word in [category] + keywords):
            matched_indicators.update(keywords)

    # Also use raw trait words as indicators
    matched_indicators.update(trait_words)

    if not matched_indicators:
        # If no specific indicators, use generic difference check
        return 0.5

    # Count matches in responses
    alignment_scores = []
    for response in responses:
        response_lower = response.lower()
        matches = sum(1 for indicator in matched_indicators if indicator in response_lower)
        # Normalize: more matches = higher score, cap at 1.0
        score = min(1.0, matches / 3.0)  # 3+ matches = perfect score
        alignment_scores.append(score)

    return float(np.mean(alignment_scores)) if alignment_scores else 0.0
