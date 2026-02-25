"""Evaluate responses command execution logic."""

import json
import os
from pathlib import Path

from wisent.core.constants import JSON_INDENT
from wisent.core.errors import TaskNotFoundError
from wisent.core.evaluators.steering_evaluators import (
    SteeringEvaluatorFactory,
    EvaluatorConfig,
    RefusalEvaluator,
    PersonalizationEvaluator as SteeringPersonalizationEvaluator,
)
from wisent.core.evaluators.rotator import EvaluatorRotator


from wisent.core.cli.analysis.evaluation.evaluate_docker import evaluate_docker_execution
from wisent.core.cli.analysis.evaluation.evaluate_personalization import evaluate_personalization
from wisent.core.cli.analysis.evaluation.evaluate_refusal import evaluate_refusal
from wisent.core.cli.analysis.evaluation.evaluate_standard import evaluate_standard


def execute_evaluate_responses(args):
    """
    Execute the evaluate-responses command.

    Evaluates generated responses using benchmark-specific evaluators.
    Uses EvaluatorRotator to auto-select the appropriate evaluator based on task name.

    Uses unified split strategy: all available splits are combined and split 80/20.
    Evaluation uses the TEST portion (20%) to ensure no data leakage with training.
    """
    from lm_eval.tasks import TaskManager
    from wisent.core.evaluators.benchmark_specific import (
        GenerationEvaluator,
        ExactMatchEvaluator,
        F1Evaluator,
        PerplexityEvaluator
    )
    from wisent.core.utils import get_all_docs_from_task, create_deterministic_split

    print(f"\n{'='*80}")
    print(f"📊 EVALUATING GENERATED RESPONSES")
    print(f"{'='*80}")
    print(f"   Input: {args.input}")
    print(f"{'='*80}\n")

    # Load input file
    print(f"📂 Loading generated responses...")
    try:
        with open(args.input, 'r') as f:
            input_data = json.load(f)

        # Handle both dict format (with 'responses' key) and direct list format
        if isinstance(input_data, list):
            responses = input_data
            task_name = args.task if args.task else "generic"  # Default to generic for list format
        else:
            responses = input_data.get('responses', [])
            task_name = args.task if args.task else input_data.get('task', "generic")

        if not task_name:
            raise ValueError("Task name not found in input file and not provided via --task")
        print(f"   ✓ Loaded {len(responses)} responses")
        print(f"   Task: {task_name}\n")
    except Exception as e:
        print(f"   ❌ Failed to load input file: {e}")
        raise

    # Load task-evaluator mapping
    print(f"📋 Loading task evaluation config...")
    try:
        # Find task-evaluator.json in project root
        current_dir = Path(__file__).resolve()
        project_root = None
        for parent in current_dir.parents:
            task_eval_file = parent / 'task-evaluator.json'
            if task_eval_file.exists():
                project_root = parent
                break

        if not project_root:
            import wisent
            wisent_pkg_dir = Path(wisent.__file__).resolve().parent
            pkg_task_eval = wisent_pkg_dir / "task-evaluator.json"
            if pkg_task_eval.exists():
                project_root = wisent_pkg_dir

        if not project_root:
            raise FileNotFoundError("Could not find task-evaluator.json")
        with open(project_root / 'task-evaluator.json', 'r') as f:
            task_evaluator_map = json.load(f)

        # Get task evaluation type
        task_config = task_evaluator_map['tasks'].get(task_name)
        if not task_config:
            raise TaskNotFoundError(task_name=task_name)

        evaluation_type = task_config['evaluation_type']
        primary_metric = task_config['primary_metric']

        print(f"   ✓ Task evaluation type: {evaluation_type}")
        print(f"   ✓ Primary metric: {primary_metric}\n")
    except Exception as e:
        print(f"   ❌ Could not load task config: {e}")
        raise

    # Load task to get ground truth (skip for docker_execution, personalization, or if responses have references)
    task_docs = None
    task = None
    
    # Check if all responses already have positive_reference (ground truth from generation)
    has_references = all(
        r.get('positive_reference') is not None 
        for r in responses
    )
    
    if evaluation_type not in ["docker_execution", "personalization"] and not has_references:
        print(f"📚 Loading task data using unified split strategy...")
        try:
            tm = TaskManager()
            task_dict = tm.load_task_or_group(task_name)
            task = task_dict[task_name]

            # Use unified split strategy: combine ALL available splits, then use TEST portion (20%)
            all_docs, split_counts = get_all_docs_from_task(task)
            _, task_docs = create_deterministic_split(all_docs, task_name)

            print(f"   ✓ Combined {len(all_docs)} total docs from splits: {split_counts}")
            print(f"   ✓ Using TEST portion: {len(task_docs)} task documents for evaluation\n")
        except Exception as e:
            print(f"   ❌ Could not load task: {e}")
            raise
    elif has_references:
        print(f"📚 Using references from responses file (task loading skipped)\n")

    # Select evaluator using EvaluatorRotator (auto-selects based on task name)
    print(f"🔧 Selecting evaluator for task '{task_name}'...")
    evaluator = None
    evaluator_rotator = None
    
    # Special handling for certain evaluation types
    if evaluation_type == "docker_execution":
        from wisent.core.evaluators.benchmark_specific.coding.metrics.evaluator import (
            CodingEvaluator,
            EvaluatorConfig
        )
        from wisent.core.evaluators.benchmark_specific.coding.providers.livecodebench import LiveCodeBenchProvider
        print(f"   Using: CodingEvaluator (Docker sandbox execution)\n")
        docker_config = task_config.get('docker_config', {})
        provider_name = task_config.get('provider', 'livecodebench')
    elif evaluation_type == "personalization":
        print(f"   Using: SteeringPersonalizationEvaluator\n")
    elif evaluation_type == "refusal":
        print(f"   Using: RefusalEvaluator\n")
    else:
        # Use EvaluatorRotator to auto-select based on task name
        try:
            EvaluatorRotator.discover_evaluators("wisent.core.evaluators.oracles")
            EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
            evaluator_rotator = EvaluatorRotator(evaluator=None, task_name=task_name)
            evaluator = evaluator_rotator.current
            print(f"   Using: {evaluator.name} (auto-selected via EvaluatorRotator)\n")
        except Exception as e:
            # Fallback to manual selection if rotator fails
            print(f"   ⚠️  EvaluatorRotator failed: {e}")
            print(f"   Falling back to manual selection based on evaluation_type...")
            if evaluation_type == "multiple_choice":
                evaluator = F1Evaluator()
                print(f"   Using: F1Evaluator (compares response to choice texts)\n")
            elif evaluation_type == "generate_until":
                if primary_metric == "exact_match":
                    evaluator = ExactMatchEvaluator()
                    print(f"   Using: ExactMatchEvaluator (extracts and compares answers)\n")
                elif primary_metric in ["em", "f1"]:
                    evaluator = F1Evaluator()
                    print(f"   Using: F1Evaluator (token-level comparison)\n")
                else:
                    evaluator = GenerationEvaluator()
                    print(f"   Using: GenerationEvaluator (extracts and compares answers)\n")
            elif evaluation_type == "loglikelihood_rolling":
                evaluator = PerplexityEvaluator()
                print(f"   Using: PerplexityEvaluator (perplexity computation)\n")
            else:
                evaluator = F1Evaluator()
                print(f"   Using: F1Evaluator (default fallback)\n")

    # Evaluate responses
    print(f"🎯 Evaluating responses...\n")
    evaluation_results = []
    task_results = []  # For aggregation


    # Handle docker_execution separately
    if evaluation_type == "docker_execution":
        aggregated_metrics = evaluate_docker_execution(
            args, input_data, responses, task_name, evaluation_results, task_results,
            task_config=task_config)
        if aggregated_metrics is not None:
            return

    # Handle personalization separately
    if evaluation_type == "personalization":
        aggregated_metrics = evaluate_personalization(
            args, input_data, responses, task_name, evaluation_results, task_results)
        if aggregated_metrics is not None:
            return

    # Handle refusal evaluation
    if evaluation_type == "refusal":
        aggregated_metrics = evaluate_refusal(
            args, input_data, responses, task_name, evaluation_results, task_results)
        if aggregated_metrics is not None:
            return

    # Standard evaluation loop
    evaluate_standard(
        args, input_data, responses, task_name,
        evaluation_results, task_results, evaluator, task_docs,
        evaluation_type=evaluation_type,
    )

    # Aggregate results
    aggregated_metrics = {}
    if task_results:
        # Get all metric keys
        all_metric_keys = set()
        for result in task_results:
            all_metric_keys.update(result.keys())

        # Aggregate each metric
        for metric_key in all_metric_keys:
            values = [r[metric_key] for r in task_results if metric_key in r]
            if values:
                # Most tasks use mean aggregation
                aggregated_metrics[metric_key] = sum(values) / len(values)

    # Save results
    print(f"\n💾 Saving evaluation results...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    output_data = {
        "input_file": args.input,
        "task": task_name if isinstance(input_data, list) else input_data.get('task'),
        "model": None if isinstance(input_data, list) else input_data.get('model'),
        "evaluation_type": evaluation_type,
        "evaluator_used": evaluator.name,
        "aggregated_metrics": aggregated_metrics,
        "num_evaluated": len(task_results),
        "num_total": len(responses),
        "evaluations": evaluation_results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=JSON_INDENT)

    print(f"   ✓ Results saved to: {args.output}\n")

    # Print summary
    print(f"{'='*80}")
    print(f"✅ EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"   Evaluator: {evaluator.name}")
    print(f"   Total responses: {len(responses)}")
    print(f"   Successfully evaluated: {len(task_results)}")
    print(f"   Failed/skipped: {len(responses) - len(task_results)}")
    print(f"\n   Metrics:")
    for metric_name, metric_value in aggregated_metrics.items():
        if isinstance(metric_value, float):
            print(f"     {metric_name}: {metric_value:.4f}")
        else:
            print(f"     {metric_name}: {metric_value}")
    print(f"{'='*80}\n")
