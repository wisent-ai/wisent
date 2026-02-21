"""Classification optimization command - uses native Wisent methods.

This optimizer tests different configurations (layer, aggregation, threshold)
by calling the native execute_tasks() function for each configuration,
then evaluates using Wisent's native evaluators.

Supports multiple evaluation modes:
- Task accuracy (default): Uses benchmark-specific evaluators
- Refusal: Uses shared RefusalEvaluator for compliance rate
- Personalization: Uses shared PersonalizationEvaluator for trait alignment

Results are persisted to ~/.wisent/configs/ via WisentConfigManager
so they can be automatically loaded on subsequent runs.
"""

import sys
import json
import time
from typing import List, Dict, Any
import os

from wisent.core.config_manager import get_config_manager, save_classification_config
from wisent.core.evaluators.steering_evaluators import (
    SteeringEvaluatorFactory,
    EvaluatorConfig,
)



from wisent.core.cli.optimization.specific.optimize_classification_runner import (
    run_classification_optimization,
)


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
    
    # Check for steering evaluation mode (refusal, personalization)
    evaluator_type = getattr(args, 'evaluator', 'task')
    trait = getattr(args, 'trait', None)
    if evaluator_type != 'task':
        print(f"   Evaluator: {evaluator_type}")
        if trait:
            print(f"   Trait: {trait}")
    print(f"{'='*80}\n")
    
    # Setup steering evaluator if needed
    steering_evaluator = None
    if evaluator_type in ['refusal', 'personalization', 'custom']:
        from wisent.core.models.wisent_model import WisentModel as WM
        eval_config = EvaluatorConfig(
            evaluator_type=evaluator_type,
            trait=trait,
            eval_prompts_path=getattr(args, 'eval_prompts', None),
            num_eval_prompts=getattr(args, 'num_eval_prompts', 30),
            custom_evaluator_path=getattr(args, 'custom_evaluator', None),
        )
        steering_evaluator = SteeringEvaluatorFactory.create(
            eval_config, args.model
        )
        print(f"📊 Using {evaluator_type} evaluator for optimization\n")

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
        # Test ALL layers by default (0-indexed)
        layers_to_test = list(range(total_layers))

    # Classifier types to test
    classifier_types = ['logistic', 'mlp']

    # Prompt construction strategies to test (5 strategies)
    prompt_construction_strategies = [
        'chat_template',
        'direct_completion',
        'multiple_choice',
        'role_playing',
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

    run_classification_optimization(args, evaluator, wisent_model, device)
