"""Utility functions for tasks command: task listing, info, and optimization."""

import os
import json
import sys


# ============================================================================
# Task listing and info functions
# ============================================================================

def execute_list_tasks(args, LMEvalDataLoader):
    """Handle --list-tasks flag to show available tasks."""
    from wisent.core.tasks.base.task_selector import TaskSelector
    selector = TaskSelector()

    if hasattr(args, 'skills') and args.skills:
        print(f"\n📋 Tasks matching skills: {', '.join(args.skills)}")
        tasks = selector.find_tasks_by_tags(
            skills=args.skills,
            min_quality_score=getattr(args, 'min_quality_score', 2)
        )
    elif hasattr(args, 'risks') and args.risks:
        print(f"\n📋 Tasks matching risks: {', '.join(args.risks)}")
        tasks = selector.find_tasks_by_tags(
            risks=args.risks,
            min_quality_score=getattr(args, 'min_quality_score', 2)
        )
    else:
        print(f"\n📋 All available tasks:")
        tasks = selector.find_tasks_by_tags(min_quality_score=getattr(args, 'min_quality_score', 2))

    print(f"\n   Found {len(tasks)} tasks:\n")
    for i, task in enumerate(sorted(tasks), 1):
        print(f"   {i}. {task}")
    print()
    return {"tasks": tasks}


def execute_task_info(args, LMEvalDataLoader):
    """Handle --task-info flag to show task details."""
    task_name = args.task_info
    print(f"\n📋 Task Information: {task_name}")
    print(f"{'='*60}")

    try:
        loader = LMEvalDataLoader()
        task_obj = loader.load_lm_eval_task(task_name)

        if isinstance(task_obj, dict):
            print(f"\n   Type: Group Task")
            print(f"   Subtasks: {len(task_obj)}")
            for subname in task_obj.keys():
                print(f"      • {subname}")
        else:
            print(f"\n   Type: Single Task")
            if hasattr(task_obj, 'config'):
                config = task_obj.config
                if hasattr(config, 'description') and config.description:
                    print(f"   Description: {config.description}")
                if hasattr(config, 'dataset_name') and config.dataset_name:
                    print(f"   Dataset: {config.dataset_name}")
                if hasattr(config, 'metric_list') and config.metric_list:
                    metrics = [m.get('metric', str(m)) if isinstance(m, dict) else str(m) for m in config.metric_list]
                    print(f"   Metrics: {', '.join(metrics)}")
                if hasattr(config, 'num_fewshot'):
                    print(f"   Few-shot examples: {config.num_fewshot}")

        print()
        return {"task": task_name, "found": True}

    except Exception as e:
        print(f"\n   ❌ Error loading task: {e}")
        print()
        return {"task": task_name, "found": False, "error": str(e)}


def select_tasks_by_criteria(args):
    """Handle --skills or --risks task selection."""
    from wisent.core.tasks.base.task_selector import TaskSelector
    selector = TaskSelector()

    selected_tasks = selector.select_random_tasks(
        skills=getattr(args, 'skills', None),
        risks=getattr(args, 'risks', None),
        num_tasks=getattr(args, 'num_tasks', None),
        min_quality_score=getattr(args, 'min_quality_score', 2),
        seed=getattr(args, 'task_seed', None)
    )

    if not selected_tasks:
        print(f"❌ No tasks found matching criteria")
        sys.exit(1)

    args.task_names = selected_tasks
    print(f"\n🎯 Selected {len(selected_tasks)} tasks based on criteria:")
    for task in selected_tasks:
        print(f"   • {task}")
    print()

    return selected_tasks


# ============================================================================
# Hyperparameter optimization functions
# ============================================================================

def execute_optimization(args, model, LMEvalDataLoader):
    """Handle --optimize flag for hyperparameter optimization."""
    from wisent.core.hyperparameter_optimizer import HyperparameterOptimizer, OptimizationConfig

    print(f"\n🔍 HYPERPARAMETER OPTIMIZATION MODE")
    print(f"   Model: {args.model}")
    print(f"   Task: {args.task_names}")

    num_layers = model.num_layers
    print(f"   ✓ Model loaded with {num_layers} layers")

    layer_range = list(range(num_layers))

    config = OptimizationConfig(
        layer_range=layer_range,
        aggregation_methods=['average', 'final', 'first', 'max'],
        prompt_construction_strategies=['multiple_choice', 'role_playing', 'direct_completion', 'instruction_following'],
        token_targeting_strategies=['choice_token', 'continuation_token', 'last_token', 'first_token', 'mean_pooling'],
        threshold_range=[0.3, 0.5, 0.7],
        classifier_types=[args.classifier_type],
        metric=getattr(args, 'optimize_metric', 'f1'),
        max_combinations=getattr(args, 'optimize_max_combinations', 1000)
    )

    total_combos = (len(layer_range) * len(config.aggregation_methods) *
                   len(config.prompt_construction_strategies) * len(config.token_targeting_strategies) *
                   len(config.threshold_range) * len(config.classifier_types))

    print(f"\n📊 Testing {total_combos} hyperparameter combinations:")
    print(f"   • Layers: {len(layer_range)} ({layer_range[0]}-{layer_range[-1]})")
    print(f"   • Aggregation methods: {len(config.aggregation_methods)}")
    print(f"   • Prompt strategies: {len(config.prompt_construction_strategies)}")
    print(f"   • Token strategies: {len(config.token_targeting_strategies)}")
    print(f"   • Thresholds: {len(config.threshold_range)}")

    task_name = args.task_names[0] if isinstance(args.task_names, list) else args.task_names
    print(f"\n📊 Loading task '{task_name}'...")
    loader = LMEvalDataLoader()
    result = loader._load_one_task(
        task_name=task_name,
        split_ratio=args.split_ratio,
        seed=args.seed,
        limit=args.limit,
        training_limit=args.training_limit,
        testing_limit=args.testing_limit
    )
    train_pair_set = result['train_qa_pairs']
    test_pair_set = result['test_qa_pairs']
    print(f"   ✓ Loaded {len(train_pair_set.pairs)} training pairs, {len(test_pair_set.pairs)} test pairs")

    optimizer = HyperparameterOptimizer(config)
    opt_result = optimizer.optimize(
        model=model,
        train_pair_set=train_pair_set,
        test_pair_set=test_pair_set,
        device=args.device,
        verbose=True
    )

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        results_path = os.path.join(args.output, 'optimization_results.json')
        optimizer.save_results(opt_result, results_path)
        print(f"\n💾 Optimization results saved to: {results_path}")

    print(f"\n✅ Optimization complete!")
    print(f"   Best layer: {opt_result.best_layer}")
    print(f"   Best aggregation: {opt_result.best_aggregation}")
    print(f"   Best prompt strategy: {opt_result.best_prompt_construction_strategy}")
    print(f"   Best token strategy: {opt_result.best_token_targeting_strategy}")
    print(f"   Best threshold: {opt_result.best_threshold}")
    print(f"   Best {config.metric}: {opt_result.best_score:.4f}")

    return {
        'best_hyperparameters': {
            'layer': opt_result.best_layer,
            'aggregation': opt_result.best_aggregation,
            'prompt_construction_strategy': opt_result.best_prompt_construction_strategy,
            'token_targeting_strategy': opt_result.best_token_targeting_strategy,
            'threshold': opt_result.best_threshold,
            'classifier_type': opt_result.best_classifier_type
        },
        'best_score': opt_result.best_score,
        'best_metrics': opt_result.best_metrics,
        'combinations_tested': len(opt_result.all_results)
    }
