"""Classification optimization command - uses native Wisent methods.

Tests different configurations (layer, aggregation, threshold) by calling
execute_tasks() for each configuration, then evaluates using Wisent evaluators.
Results are persisted to ~/.wisent/configs/ via WisentConfigManager.
"""
from __future__ import annotations
import json
import os
from typing import Dict, Any

from wisent.core.utils.config_tools.constants import (
    DEFAULT_NUM_EVAL_PROMPTS,
    DEFAULT_SHOW_COMPARISONS,
    JSON_INDENT,
    CLASSIFIER_THRESHOLD,
    SEPARATOR_WIDTH_MAX,
)
from wisent.core.utils.config_tools.config import save_classification_config
from wisent.core.reading.evaluators.steering_evaluators import SteeringEvaluatorFactory, EvaluatorConfig
from wisent.core.utils.cli.optimization.specific.optimize_classification_runner import (
    run_classification_optimization,
)


def execute_optimize_classification(args):
    """Execute classification optimization using native Wisent methods."""
    from wisent.core.utils.cli.tasks import execute_tasks
    print(f"\n{'='*80}")
    print(f"CLASSIFICATION PARAMETER OPTIMIZATION")
    print(f"{'='*80}")
    print(f"   Model: {args.model}")
    print(f"   Limit per task: {args.limit}")
    print(f"   Device: {args.device or 'auto'}")
    evaluator_type = getattr(args, 'evaluator', 'task')
    trait = getattr(args, 'trait', None)
    if evaluator_type != 'task':
        print(f"   Evaluator: {evaluator_type}")
        if trait:
            print(f"   Trait: {trait}")
    print(f"{'='*80}\n")
    steering_evaluator = None
    if evaluator_type in ['refusal', 'personalization', 'custom']:
        eval_config = EvaluatorConfig(
            evaluator_type=evaluator_type, trait=trait,
            eval_prompts_path=getattr(args, 'eval_prompts', None),
            num_eval_prompts=getattr(args, 'num_eval_prompts', DEFAULT_NUM_EVAL_PROMPTS),
            custom_evaluator_path=getattr(args, 'custom_evaluator', None),
        )
        steering_evaluator = SteeringEvaluatorFactory.create(eval_config, args.model)
        print(f"Using {evaluator_type} evaluator for optimization\n")
    from wisent.core.primitives.models.wisent_model import WisentModel
    print(f"Loading model to determine layer range...")
    model = WisentModel(args.model, device=args.device)
    total_layers = model.num_layers
    print(f"   Model has {total_layers} layers\n")
    if args.layer_range:
        start, end = map(int, args.layer_range.split('-'))
        layers_to_test = list(range(start, end + 1))
    else:
        layers_to_test = list(range(total_layers))
    classifier_types = ['logistic', 'mlp']
    prompt_strategies = [
        'chat_template', 'direct_completion', 'multiple_choice',
        'role_playing', 'instruction_following',
    ]
    token_strategies = [
        'choice_token', 'continuation_token', 'last_token',
        'first_token', 'mean_pooling',
    ]
    if hasattr(args, 'tasks') and args.tasks:
        task_list = args.tasks if isinstance(args.tasks, list) else [args.tasks]
    else:
        task_list = ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "gsm8k"]
    print(f"Testing {len(layers_to_test)} layers x {len(classifier_types)} classifiers x "
          f"{len(args.aggregation_methods)} agg x {len(prompt_strategies)} prompts x "
          f"{len(token_strategies)} tokens x {len(args.threshold_range)} thresholds")
    print(f"Optimizing {len(task_list)} tasks\n")
    all_results = run_classification_optimization(
        args=args, execute_tasks_fn=execute_tasks,
        steering_evaluator=steering_evaluator, total_layers=total_layers,
        layers_to_test=layers_to_test, classifier_types=classifier_types,
        prompt_strategies=prompt_strategies, token_strategies=token_strategies,
        task_list=task_list,
    )
    _save_results(args, all_results, total_layers)


def _save_results(args, all_results, total_layers):
    """Save optimization results to disk and display summary."""
    results_dir = getattr(args, 'classifiers_dir', None) or './optimization_results'
    os.makedirs(results_dir, exist_ok=True)
    model_safe = args.model.replace('/', '_')
    results_file = os.path.join(results_dir, f'classification_optimization_{model_safe}.json')
    with open(results_file, 'w') as f:
        json.dump({'model': args.model, 'optimization_metric': args.optimization_metric,
                   'results': all_results}, f, indent=JSON_INDENT)
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE - Results saved to: {results_file}")
    print(f"{'='*80}\n")
    _persist_configs(args, all_results)
    _print_summary(args, all_results)
    _handle_comparisons(args, all_results, total_layers)


def _persist_configs(args, all_results):
    """Save configs to WisentConfigManager for persistence."""
    print(f"Saving optimal parameters to ~/.wisent/configs/...")
    for task_name, result in all_results.items():
        c = result['best_config']
        save_classification_config(
            model_name=args.model, task_name=task_name, layer=c['layer'],
            token_aggregation=c['aggregation'], detection_threshold=c['threshold'],
            classifier_type=c['classifier_type'],
            prompt_construction_strategy=c['prompt_construction_strategy'],
            token_targeting_strategy=c['token_targeting_strategy'],
            accuracy=c['accuracy'], f1_score=c['f1_score'],
            precision=c['precision'], recall=c['recall'],
            optimization_method="grid_search",
            set_as_default=(task_name == list(all_results.keys())[0]),
        )
    best_score, best_cfg, best_task = -1, None, None
    for task_name, result in all_results.items():
        if result['best_score'] > best_score:
            best_score = result['best_score']
            best_cfg = result['best_config']
            best_task = task_name
    if best_cfg:
        save_classification_config(
            model_name=args.model, task_name=None, layer=best_cfg['layer'],
            token_aggregation=best_cfg['aggregation'],
            detection_threshold=best_cfg['threshold'],
            classifier_type=best_cfg['classifier_type'],
            prompt_construction_strategy=best_cfg['prompt_construction_strategy'],
            token_targeting_strategy=best_cfg['token_targeting_strategy'],
            accuracy=best_cfg['accuracy'], f1_score=best_cfg['f1_score'],
            precision=best_cfg['precision'], recall=best_cfg['recall'],
            optimization_method="grid_search",
        )
        print(f"   Default layer: {best_cfg['layer']} (from {best_task})")
        print(f"   Task configs saved for: {', '.join(all_results.keys())}\n")


def _print_summary(args, all_results):
    """Print summary table of results."""
    sep = "-" * SEPARATOR_WIDTH_MAX
    print(f"SUMMARY BY TASK:")
    print(sep)
    hdr = (f"{'Task':<20} | {'Layer':>5} | {'Classifier':<10} | {'Agg':<12} | "
           f"{'Prompt':<20} | {'Token':<15} | {'Thresh':>6} | {'F1':>6} | {'Acc':>6}")
    print(hdr)
    print(sep)
    for task_name, result in all_results.items():
        c = result['best_config']
        print(f"{task_name:<20} | {c['layer']:>5} | {c['classifier_type']:<10} | "
              f"{c['aggregation']:<12} | {c['prompt_construction_strategy']:<20} | "
              f"{c['token_targeting_strategy']:<15} | {c['threshold']:>6.2f} | "
              f"{c['f1_score']:>6.4f} | {c['accuracy']:>6.4f}")
    print(sep + "\n")


def _handle_comparisons(args, all_results, total_layers):
    """Handle --show-comparisons and --save-comparisons flags."""
    show = getattr(args, 'show_comparisons', DEFAULT_SHOW_COMPARISONS)
    save_path = getattr(args, 'save_comparisons', None)
    if show <= 0 and not save_path:
        return
    print("\nGenerating comparison data (optimized vs default config)...")
    comparisons = []
    for task_name, result in all_results.items():
        bc = result['best_config']
        dc = {
            'layer': total_layers // 2, 'aggregation': 'average', 'threshold': CLASSIFIER_THRESHOLD,
            'classifier_type': 'logistic', 'prompt_construction_strategy': 'multiple_choice',
            'token_targeting_strategy': 'last_token',
        }
        comparisons.append({
            'task': task_name, 'default_config': dc,
            'optimized_config': {k: bc[k] for k in dc},
            'optimized_metrics': {
                'f1': bc['f1_score'], 'accuracy': bc['accuracy'],
                'precision': bc['precision'], 'recall': bc['recall'],
            },
            'improvements': {
                'layer_change': bc['layer'] - dc['layer'],
                'aggregation_change': dc['aggregation'] != bc['aggregation'],
                'threshold_change': bc['threshold'] - dc['threshold'],
            },
        })
    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump({'model': args.model, 'optimization_metric': args.optimization_metric,
                       'comparisons': comparisons}, f, indent=JSON_INDENT)
        print(f"Saved comparisons to: {save_path}")
    if show > 0:
        print(f"\nComparisons (showing {min(show, len(comparisons))} tasks):\n")
        for comp in comparisons[:show]:
            dc, oc = comp['default_config'], comp['optimized_config']
            print(f"{'_'*80}")
            print(f"Task: {comp['task']}")
            print(f"DEFAULT:   Layer={dc['layer']}, Agg={dc['aggregation']}, Thresh={dc['threshold']}")
            print(f"OPTIMIZED: Layer={oc['layer']}, Agg={oc['aggregation']}, Thresh={oc['threshold']:.2f}")
            print(f"           Cls={oc['classifier_type']}, Prompt={oc['prompt_construction_strategy']}")
            m = comp['optimized_metrics']
            print(f"METRICS:   F1={m['f1']:.4f}, Accuracy={m['accuracy']:.4f}")
            print()
