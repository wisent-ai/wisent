"""Classification optimization command - uses native Wisent methods.

This optimizer tests different configurations (layer, aggregation, threshold)
by calling the native execute_tasks() function for each configuration,
then evaluates using Wisent's native evaluators.
"""

import sys
import json
import time
from typing import List, Dict, Any
import os


def execute_optimize_classification(args):
    """
    Execute classification optimization using native Wisent methods.

    Tests different configurations by calling execute_tasks() for each combination:
    - Different layers
    - Different aggregation methods
    - Different detection thresholds

    Uses native Wisent evaluation (not sklearn metrics).
    """
    from wisent.core.cli.tasks import execute_tasks
    from types import SimpleNamespace

    print(f"\n{'='*80}")
    print(f"🔍 CLASSIFICATION PARAMETER OPTIMIZATION")
    print(f"{'='*80}")
    print(f"   Model: {args.model}")
    print(f"   Limit per task: {args.limit}")
    print(f"   Device: {args.device or 'auto'}")
    print(f"{'='*80}\n")

    # 1. Determine layer range
    # First need to load model to get num_layers
    from wisent.core.models.wisent_model import WisentModel
    print(f"📦 Loading model to determine layer range...")
    model = WisentModel(args.model, device=args.device)
    total_layers = model.num_layers
    print(f"   ✓ Model has {total_layers} layers\n")

    if args.layer_range:
        start, end = map(int, args.layer_range.split('-'))
        layers_to_test = list(range(start, end + 1))
    else:
        # Test middle layers by default
        start_layer = total_layers // 3
        end_layer = (2 * total_layers) // 3
        layers_to_test = list(range(start_layer, end_layer + 1))

    # Classifier types to test
    classifier_types = ['logistic', 'mlp']

    # Prompt construction strategies to test
    prompt_construction_strategies = [
        'multiple_choice',
        'role_playing',
        'direct_completion',
        'instruction_following'
    ]

    # Token targeting strategies to test
    token_targeting_strategies = [
        'choice_token',
        'continuation_token',
        'last_token',
        'first_token',
        'mean_pooling'
    ]

    print(f"🎯 Testing layers: {layers_to_test[0]} to {layers_to_test[-1]} ({len(layers_to_test)} layers)")
    print(f"🤖 Classifier types: {', '.join(classifier_types)}")
    print(f"🔄 Aggregation methods: {', '.join(args.aggregation_methods)}")
    print(f"📝 Prompt strategies: {', '.join(prompt_construction_strategies)}")
    print(f"🎯 Token targeting: {', '.join(token_targeting_strategies)}")
    print(f"📊 Thresholds: {args.threshold_range}\n")

    # 2. Get list of tasks
    if hasattr(args, 'tasks') and args.tasks:
        task_list = args.tasks if isinstance(args.tasks, list) else [args.tasks]
    else:
        task_list = [
            "arc_easy", "arc_challenge", "hellaswag",
            "winogrande", "gsm8k"
        ]

    print(f"📋 Optimizing {len(task_list)} tasks\n")

    # 3. Results storage
    all_results = {}

    # 4. Process each task
    for task_idx, task_name in enumerate(task_list, 1):
        print(f"\n{'='*80}")
        print(f"Task {task_idx}/{len(task_list)}: {task_name}")
        print(f"{'='*80}")

        task_start_time = time.time()

        try:
            best_score = -1
            best_config = None

            combinations_tested = 0
            total_combinations = (len(layers_to_test) * len(classifier_types) *
                                  len(args.aggregation_methods) * len(args.threshold_range) *
                                  len(prompt_construction_strategies) * len(token_targeting_strategies))

            print(f"  🔍 Testing {total_combinations} configurations...")
            print(f"      ({len(layers_to_test)} layers × {len(classifier_types)} classifiers × "
                  f"{len(args.aggregation_methods)} aggregations × {len(args.threshold_range)} thresholds × "
                  f"{len(prompt_construction_strategies)} prompt strategies × {len(token_targeting_strategies)} token strategies)")

            for layer in layers_to_test:
                for classifier_type in classifier_types:
                    for agg_method in args.aggregation_methods:
                        for threshold in args.threshold_range:
                            for prompt_strategy in prompt_construction_strategies:
                                for token_strategy in token_targeting_strategies:
                                    combinations_tested += 1

                                    # Create args namespace for execute_tasks
                                    task_args = SimpleNamespace(
                                        task_names=[task_name],
                                        model=args.model,
                                        layer=layer,
                                        classifier_type=classifier_type,
                                        token_aggregation=agg_method,
                                        detection_threshold=threshold,
                                        prompt_construction_strategy=prompt_strategy,
                                        token_targeting_strategy=token_strategy,
                                        split_ratio=0.8,
                                        seed=42,
                                        limit=args.limit,
                                        training_limit=None,
                                        testing_limit=None,
                                        device=args.device,
                                        save_classifier=None,  # Don't save intermediate classifiers
                                        output=None,
                                        inference_only=False,
                                        load_steering_vector=None,
                                        save_steering_vector=None,
                                        train_only=False,
                                        steering_method='caa'
                                    )

                                    try:
                                        # Call native Wisent execute_tasks
                                        result = execute_tasks(task_args)

                                        # Extract metrics from result
                                        # Map CLI argument to result key
                                        metric_map = {
                                            'f1': 'f1_score',
                                            'accuracy': 'accuracy',
                                            'precision': 'precision',
                                            'recall': 'recall'
                                        }
                                        metric_key = metric_map.get(args.optimization_metric, 'f1_score')
                                        metric_value = result.get(metric_key, 0)

                                        if metric_value > best_score:
                                            best_score = metric_value
                                            best_config = {
                                                'layer': layer,
                                                'classifier_type': classifier_type,
                                                'aggregation': agg_method,
                                                'threshold': threshold,
                                                'prompt_construction_strategy': prompt_strategy,
                                                'token_targeting_strategy': token_strategy,
                                                'accuracy': result.get('accuracy', 0),
                                                'f1_score': result.get('f1_score', 0),
                                                'precision': result.get('precision', 0),
                                                'recall': result.get('recall', 0),
                                                'generation_count': result.get('generation_count', 0),
                                            }

                                        if combinations_tested % 20 == 0:
                                            print(f"      Progress: {combinations_tested}/{total_combinations} tested, best {args.optimization_metric}: {best_score:.4f}", end='\r')

                                    except Exception as e:
                                        # NO FALLBACK - raise error
                                        print(f"\n❌ Configuration failed:")
                                        print(f"   Layer: {layer}")
                                        print(f"   Aggregation: {agg_method}")
                                        print(f"   Threshold: {threshold}")
                                        print(f"   Prompt strategy: {prompt_strategy}")
                                        print(f"   Token strategy: {token_strategy}")
                                        print(f"   Error: {e}")
                                        raise

            print(f"\n\n  ✅ Best config for {task_name}:")
            print(f"      Layer: {best_config['layer']}")
            print(f"      Classifier: {best_config['classifier_type']}")
            print(f"      Aggregation: {best_config['aggregation']}")
            print(f"      Threshold: {best_config['threshold']:.2f}")
            print(f"      Prompt Strategy: {best_config['prompt_construction_strategy']}")
            print(f"      Token Strategy: {best_config['token_targeting_strategy']}")
            print(f"      Performance metrics:")
            print(f"        • Accuracy:  {best_config['accuracy']:.4f}")
            print(f"        • F1 Score:  {best_config['f1_score']:.4f}")
            print(f"        • Precision: {best_config['precision']:.4f}")
            print(f"        • Recall:    {best_config['recall']:.4f}")
            print(f"        • Generations evaluated: {best_config['generation_count']}")

            # Train final classifier with best config and save
            if hasattr(args, 'classifiers_dir') and args.classifiers_dir:
                print(f"\n  💾 Training final classifier with best config...")

                final_args = SimpleNamespace(
                    task_names=[task_name],
                    model=args.model,
                    layer=best_config['layer'],
                    classifier_type=best_config['classifier_type'],
                    token_aggregation=best_config['aggregation'],
                    detection_threshold=best_config['threshold'],
                    prompt_construction_strategy=best_config['prompt_construction_strategy'],
                    token_targeting_strategy=best_config['token_targeting_strategy'],
                    split_ratio=0.8,
                    seed=42,
                    limit=args.limit,
                    training_limit=None,
                    testing_limit=None,
                    device=args.device,
                    save_classifier=os.path.join(args.classifiers_dir, f"{task_name}_classifier.pt"),
                    output=os.path.join(args.classifiers_dir, task_name),
                    inference_only=False,
                    load_steering_vector=None,
                    save_steering_vector=None,
                    train_only=False,
                    steering_method='caa'
                )

                execute_tasks(final_args)
                print(f"      ✓ Classifier saved to: {final_args.save_classifier}")

            # Store results
            all_results[task_name] = {
                'best_config': best_config,
                'optimization_metric': args.optimization_metric,
                'best_score': best_score,
                'combinations_tested': combinations_tested
            }

            task_time = time.time() - task_start_time
            print(f"\n  ⏱️  Task completed in {task_time:.1f}s")

        except Exception as e:
            # NO FALLBACK - raise error
            print(f"\n❌ Task '{task_name}' optimization failed:")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            raise

    # 5. Save optimization results
    results_dir = getattr(args, 'classifiers_dir', None) or './optimization_results'
    os.makedirs(results_dir, exist_ok=True)

    model_name_safe = args.model.replace('/', '_')
    results_file = os.path.join(results_dir, f'classification_optimization_{model_name_safe}.json')

    with open(results_file, 'w') as f:
        json.dump({
            'model': args.model,
            'optimization_metric': args.optimization_metric,
            'results': all_results
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"📊 OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Results saved to: {results_file}\n")

    # Print summary
    print(f"📋 SUMMARY BY TASK:")
    print(f"-" * 180)
    print(f"{'Task':<20} | {'Layer':>5} | {'Classifier':<10} | {'Agg':<12} | {'Prompt':<20} | {'Token':<15} | {'Thresh':>6} | {'F1':>6} | {'Acc':>6}")
    print(f"-" * 180)
    for task_name, result in all_results.items():
        config = result['best_config']
        print(f"{task_name:<20} | {config['layer']:>5} | {config['classifier_type']:<10} | "
              f"{config['aggregation']:<12} | {config['prompt_construction_strategy']:<20} | "
              f"{config['token_targeting_strategy']:<15} | {config['threshold']:>6.2f} | "
              f"{config['f1_score']:>6.4f} | {config['accuracy']:>6.4f}")
    print(f"-" * 180)
    print()

    # Handle --show-comparisons and --save-comparisons
    show_comparisons = getattr(args, 'show_comparisons', 0)
    save_comparisons = getattr(args, 'save_comparisons', None)

    if show_comparisons > 0 or save_comparisons:
        print("\n📊 Generating comparison data (optimized vs default config)...")

        # Build comparison data showing best config vs a baseline config
        all_comparisons = []
        for task_name, result in all_results.items():
            best_config = result['best_config']

            # Default baseline config for comparison
            default_config = {
                'layer': total_layers // 2,  # Middle layer
                'aggregation': 'average',
                'threshold': 0.5,
                'classifier_type': 'logistic',
                'prompt_construction_strategy': 'multiple_choice',
                'token_targeting_strategy': 'last_token',
            }

            all_comparisons.append({
                'task': task_name,
                'default_config': default_config,
                'optimized_config': {
                    'layer': best_config['layer'],
                    'aggregation': best_config['aggregation'],
                    'threshold': best_config['threshold'],
                    'classifier_type': best_config['classifier_type'],
                    'prompt_construction_strategy': best_config['prompt_construction_strategy'],
                    'token_targeting_strategy': best_config['token_targeting_strategy'],
                },
                'optimized_metrics': {
                    'f1': best_config['f1_score'],
                    'accuracy': best_config['accuracy'],
                    'precision': best_config['precision'],
                    'recall': best_config['recall'],
                },
                'improvements': {
                    'layer_change': best_config['layer'] - default_config['layer'],
                    'aggregation_change': default_config['aggregation'] != best_config['aggregation'],
                    'threshold_change': best_config['threshold'] - default_config['threshold'],
                },
            })

        # Save to JSON if requested
        if save_comparisons:
            os.makedirs(os.path.dirname(save_comparisons) if os.path.dirname(save_comparisons) else ".", exist_ok=True)
            with open(save_comparisons, 'w') as f:
                json.dump({
                    'model': args.model,
                    'optimization_metric': args.optimization_metric,
                    'comparisons': all_comparisons,
                }, f, indent=2)
            print(f"💾 Saved comparisons to: {save_comparisons}")

        # Display in console if requested
        if show_comparisons > 0:
            print(f"\n📊 Configuration Comparisons (showing {min(show_comparisons, len(all_comparisons))} tasks):\n")
            for i, comp in enumerate(all_comparisons[:show_comparisons]):
                print(f"{'─'*80}")
                print(f"Task: {comp['task']}")
                print(f"{'─'*80}")
                print(f"DEFAULT CONFIG:")
                print(f"  Layer: {comp['default_config']['layer']}, Agg: {comp['default_config']['aggregation']}, "
                      f"Threshold: {comp['default_config']['threshold']}")
                print(f"OPTIMIZED CONFIG:")
                print(f"  Layer: {comp['optimized_config']['layer']}, Agg: {comp['optimized_config']['aggregation']}, "
                      f"Threshold: {comp['optimized_config']['threshold']:.2f}")
                print(f"  Classifier: {comp['optimized_config']['classifier_type']}, "
                      f"Prompt: {comp['optimized_config']['prompt_construction_strategy']}, "
                      f"Token: {comp['optimized_config']['token_targeting_strategy']}")
                print(f"METRICS:")
                print(f"  F1: {comp['optimized_metrics']['f1']:.4f}, Accuracy: {comp['optimized_metrics']['accuracy']:.4f}")
                print()
