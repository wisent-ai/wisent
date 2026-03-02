"""Classification optimization execution loop."""
from __future__ import annotations
import os
import time

from wisent.core.constants import DEFAULT_SCORE, DEFAULT_SPLIT_RATIO, DEFAULT_RANDOM_SEED, PROGRESS_LOG_INTERVAL_20
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


def run_classification_optimization(
    args, execute_tasks_fn, steering_evaluator,
    total_layers, layers_to_test, classifier_types,
    prompt_strategies, token_strategies, task_list,
):
    """Run the classification optimization grid search loop. Returns all_results dict."""
    all_results: Dict[str, Any] = {}
    for task_idx, task_name in enumerate(task_list, 1):
        print(f"\n{'='*80}")
        print(f"Task {task_idx}/{len(task_list)}: {task_name}")
        print(f"{'='*80}")
        task_start_time = time.time()
        try:
            best_score = -1
            best_config = None
            combinations_tested = 0
            total_combinations = (
                len(layers_to_test) * len(classifier_types) *
                len(args.aggregation_methods) * len(args.threshold_range) *
                len(prompt_strategies) * len(token_strategies)
            )
            print(f"  Testing {total_combinations} configurations...")
            for layer in layers_to_test:
                for classifier_type in classifier_types:
                    for agg_method in args.aggregation_methods:
                        for threshold in args.threshold_range:
                            for prompt_strategy in prompt_strategies:
                                for token_strategy in token_strategies:
                                    combinations_tested += 1
                                    task_args = _build_task_args(
                                        task_name, args, layer, classifier_type,
                                        agg_method, threshold, prompt_strategy,
                                        token_strategy,
                                    )
                                    try:
                                        result = execute_tasks_fn(task_args)
                                        mv = _extract_metric(result, args, steering_evaluator)
                                        if mv > best_score:
                                            best_score = mv
                                            best_config = _build_best_config(
                                                layer, classifier_type, agg_method,
                                                threshold, prompt_strategy,
                                                token_strategy, result,
                                            )
                                        if combinations_tested % PROGRESS_LOG_INTERVAL_20 == 0:
                                            mn = args.optimization_metric
                                            print(f"      Progress: {combinations_tested}/{total_combinations}, best {mn}: {best_score:.4f}", end='\r')
                                    except Exception as e:
                                        print(f"\nConfig failed: layer={layer} agg={agg_method} thresh={threshold}")
                                        print(f"   prompt={prompt_strategy} token={token_strategy} err={e}")
                                        raise
            _print_task_best(task_name, best_config)
            _train_final_if_needed(args, task_name, best_config, execute_tasks_fn)
            all_results[task_name] = {
                'best_config': best_config,
                'optimization_metric': args.optimization_metric,
                'best_score': best_score,
                'combinations_tested': combinations_tested,
            }
            task_time = time.time() - task_start_time
            print(f"\n  Task completed in {task_time:.1f}s")
        except Exception as e:
            print(f"\nTask '{task_name}' optimization failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    return all_results


def _build_task_args(task_name, args, layer, classifier_type, agg_method, threshold, prompt_strategy, token_strategy):
    """Build SimpleNamespace args for execute_tasks call."""
    return SimpleNamespace(
        task_names=[task_name], model=args.model, layer=layer,
        classifier_type=classifier_type, token_aggregation=agg_method,
        detection_threshold=threshold,
        prompt_construction_strategy=prompt_strategy,
        token_targeting_strategy=token_strategy,
        split_ratio=DEFAULT_SPLIT_RATIO, seed=DEFAULT_RANDOM_SEED, limit=args.limit,
        training_limit=None, testing_limit=None,
        device=args.device, save_classifier=None, output=None,
        inference_only=False, load_steering_vector=None,
        save_steering_vector=None, train_only=False, steering_method='caa',
    )


def _extract_metric(result, args, steering_evaluator):
    """Extract metric value from result based on evaluator type."""
    if steering_evaluator is not None:
        generated_responses = result.get('generated_responses', [])
        if generated_responses:
            eval_results = steering_evaluator.evaluate_responses(generated_responses)
            metric_value = eval_results.get('score', DEFAULT_SCORE)
            result['steering_score'] = metric_value
            result['steering_metrics'] = eval_results
        else:
            metric_value = 0
    else:
        metric_map = {
            'f1': 'f1_score', 'accuracy': 'accuracy',
            'precision': 'precision', 'recall': 'recall',
        }
        metric_key = metric_map.get(args.optimization_metric, 'f1_score')
        metric_value = result.get(metric_key, DEFAULT_SCORE)
    return metric_value


def _build_best_config(layer, ctype, agg, thresh, prompt_s, token_s, result):
    """Build best config dict from current best parameters."""
    return {
        'layer': layer, 'classifier_type': ctype,
        'aggregation': agg, 'threshold': thresh,
        'prompt_construction_strategy': prompt_s,
        'token_targeting_strategy': token_s,
        'accuracy': result.get('accuracy', DEFAULT_SCORE),
        'f1_score': result.get('f1_score', DEFAULT_SCORE),
        'precision': result.get('precision', DEFAULT_SCORE),
        'recall': result.get('recall', DEFAULT_SCORE),
        'generation_count': result.get('generation_count', 0),
        'steering_score': result.get('steering_score', None),
        'steering_metrics': result.get('steering_metrics', None),
    }


def _print_task_best(task_name, best_config):
    """Print the best configuration found for a task."""
    print(f"\n\n  Best config for {task_name}:")
    print(f"      Layer: {best_config['layer']}, Classifier: {best_config['classifier_type']}")
    print(f"      Aggregation: {best_config['aggregation']}, Threshold: {best_config['threshold']:.2f}")
    print(f"      Prompt: {best_config['prompt_construction_strategy']}")
    print(f"      Token: {best_config['token_targeting_strategy']}")
    print(f"      Accuracy={best_config['accuracy']:.4f} F1={best_config['f1_score']:.4f} "
          f"Precision={best_config['precision']:.4f} Recall={best_config['recall']:.4f}")


def _train_final_if_needed(args, task_name, best_config, execute_tasks_fn):
    """Train and save final classifier with best config if classifiers_dir is set."""
    if not (hasattr(args, 'classifiers_dir') and args.classifiers_dir):
        return
    print(f"\n  Training final classifier with best config...")
    c = best_config
    final_args = SimpleNamespace(
        task_names=[task_name], model=args.model,
        layer=c['layer'], classifier_type=c['classifier_type'],
        token_aggregation=c['aggregation'],
        detection_threshold=c['threshold'],
        prompt_construction_strategy=c['prompt_construction_strategy'],
        token_targeting_strategy=c['token_targeting_strategy'],
        split_ratio=DEFAULT_SPLIT_RATIO, seed=DEFAULT_RANDOM_SEED, limit=args.limit,
        training_limit=None, testing_limit=None, device=args.device,
        save_classifier=os.path.join(args.classifiers_dir, f"{task_name}_classifier.pt"),
        output=os.path.join(args.classifiers_dir, task_name),
        inference_only=False, load_steering_vector=None,
        save_steering_vector=None, train_only=False, steering_method='caa',
    )
    execute_tasks_fn(final_args)
    print(f"      Classifier saved to: {final_args.save_classifier}")
