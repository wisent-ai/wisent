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

    print(f"🎯 Testing layers: {layers_to_test[0]} to {layers_to_test[-1]} ({len(layers_to_test)} layers)")
    print(f"🤖 Classifier types: {', '.join(classifier_types)}")
    print(f"🔄 Aggregation methods: {', '.join(args.aggregation_methods)}")
    print(f"📊 Thresholds: {args.threshold_range}\n")

    # 2. Get list of tasks
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
            total_combinations = len(layers_to_test) * len(classifier_types) * len(args.aggregation_methods) * len(args.threshold_range)

            print(f"  🔍 Testing {total_combinations} configurations...")

            for layer in layers_to_test:
                for classifier_type in classifier_types:
                    for agg_method in args.aggregation_methods:
                        for threshold in args.threshold_range:
                            combinations_tested += 1

                            # Create args namespace for execute_tasks
                            task_args = SimpleNamespace(
                                task_names=[task_name],
                                model=args.model,
                                layer=layer,
                                classifier_type=classifier_type,
                                token_aggregation=agg_method,
                                detection_threshold=threshold,
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
                                    'accuracy': result.get('accuracy', 0),
                                    'f1_score': result.get('f1_score', 0),
                                    'precision': result.get('precision', 0),
                                    'recall': result.get('recall', 0),
                                    'generation_count': result.get('generation_count', 0),
                                }

                            if combinations_tested % 5 == 0:
                                print(f"      Progress: {combinations_tested}/{total_combinations} tested, best {args.optimization_metric}: {best_score:.4f}", end='\r')

                        except Exception as e:
                            # NO FALLBACK - raise error
                            print(f"\n❌ Configuration failed:")
                            print(f"   Layer: {layer}")
                            print(f"   Aggregation: {agg_method}")
                            print(f"   Threshold: {threshold}")
                            print(f"   Error: {e}")
                            raise

            print(f"\n\n  ✅ Best config for {task_name}:")
            print(f"      Layer: {best_config['layer']}")
            print(f"      Classifier: {best_config['classifier_type']}")
            print(f"      Aggregation: {best_config['aggregation']}")
            print(f"      Threshold: {best_config['threshold']:.2f}")
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
    print(f"-" * 140)
    for task_name, result in all_results.items():
        config = result['best_config']
        print(f"{task_name:20} | Layer: {config['layer']:2} | Classifier: {config['classifier_type']:10} | "
              f"Agg: {config['aggregation']:8} | Thresh: {config['threshold']:.2f} | "
              f"F1: {config['f1_score']:.4f} | Acc: {config['accuracy']:.4f} | Gens: {config['generation_count']:3}")
    print(f"-" * 140)
    print()
