"""Evaluate responses command execution logic."""

import json
import os
import sys
from pathlib import Path


def execute_evaluate_responses(args):
    """
    Execute the evaluate-responses command.

    Evaluates generated responses using benchmark-specific evaluators.
    Routes to appropriate evaluator based on task type from task-evaluator.json.
    """
    from lm_eval.tasks import TaskManager
    from wisent.core.evaluators.benchmark_specific import (
        GenerationEvaluator,
        ExactMatchEvaluator,
        F1Evaluator,
        PerplexityEvaluator
    )

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

        responses = input_data.get('responses', [])
        task_name = input_data.get('task')
        print(f"   ✓ Loaded {len(responses)} responses")
        print(f"   Task: {task_name}\n")
    except Exception as e:
        print(f"   ❌ Failed to load input file: {e}")
        sys.exit(1)

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
            raise FileNotFoundError("Could not find task-evaluator.json")

        with open(project_root / 'task-evaluator.json', 'r') as f:
            task_evaluator_map = json.load(f)

        # Get task evaluation type
        task_config = task_evaluator_map['tasks'].get(task_name)
        if not task_config:
            raise ValueError(f"Task {task_name} not found in task-evaluator.json")

        evaluation_type = task_config['evaluation_type']
        primary_metric = task_config['primary_metric']

        print(f"   ✓ Task evaluation type: {evaluation_type}")
        print(f"   ✓ Primary metric: {primary_metric}\n")
    except Exception as e:
        print(f"   ❌ Could not load task config: {e}")
        sys.exit(1)

    # Load task to get ground truth
    print(f"📚 Loading task data...")
    task_docs = None
    task = None
    try:
        tm = TaskManager()
        task_dict = tm.load_task_or_group(task_name)
        task = task_dict[task_name]

        # Get validation docs
        task_docs = list(task.validation_docs())

        print(f"   ✓ Loaded {len(task_docs)} task documents\n")
    except Exception as e:
        print(f"   ❌ Could not load task: {e}")
        sys.exit(1)

    # Select evaluator based on evaluation type
    print(f"🔧 Selecting evaluator for {evaluation_type} task...")
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

    for idx, response_data in enumerate(responses, 1):
        if 'error' in response_data:
            if args.verbose:
                print(f"Question {idx}: Skipped (generation error)")
            evaluation_results.append({
                **response_data,
                "evaluation": {
                    "error": "Generation failed"
                }
            })
            continue

        try:
            generated_response = response_data.get('generated_response', '')
            prompt = response_data.get('prompt', '')

            # Find matching task doc by question text
            task_doc = None
            if task_docs:
                for doc in task_docs:
                    doc_question = doc.get('question', '').strip()
                    if doc_question and doc_question in prompt:
                        task_doc = doc
                        break

            if not task_doc:
                if args.verbose:
                    print(f"Question {idx}: Could not match to task doc")
                evaluation_results.append({
                    **response_data,
                    "evaluation": {
                        "error": "Could not match to task document"
                    }
                })
                continue

            # Get expected answer based on evaluation type
            if evaluation_type == "multiple_choice":
                # Get all choice texts and gold index
                gold_idx = None
                choice_texts = []

                if 'mc1_targets' in task_doc:
                    # truthfulqa_mc1 format
                    labels = task_doc['mc1_targets']['labels']
                    gold_idx = labels.index(1)
                    choice_texts = task_doc['mc1_targets']['choices']
                elif 'choices' in task_doc:
                    # arc_easy, piqa, etc. format
                    answer_key = task_doc.get('answerKey', 'A')
                    gold_idx = ord(answer_key) - ord('A')
                    if isinstance(task_doc['choices'], dict):
                        choice_texts = task_doc['choices']['text']
                    else:
                        choice_texts = task_doc['choices']
                elif 'gold' in task_doc:
                    # Some tasks have gold directly
                    gold_idx = task_doc['gold']
                    choice_texts = task.doc_to_choice(task_doc)
                else:
                    if args.verbose:
                        print(f"Question {idx}: Unknown multiple-choice format")
                    evaluation_results.append({
                        **response_data,
                        "evaluation": {
                            "error": "Unknown task format"
                        }
                    })
                    continue

                # Use F1Evaluator to match response to best choice
                best_score = 0.0
                best_choice_idx = None

                for i, choice_text in enumerate(choice_texts):
                    result = evaluator.evaluate(generated_response, choice_text)
                    if result.confidence > best_score:
                        best_score = result.confidence
                        best_choice_idx = i

                # Check if correct
                is_correct = (best_choice_idx == gold_idx)

                # Store result
                task_results.append({
                    'acc': 1.0 if is_correct else 0.0,
                    'f1_score': best_score
                })

                if args.verbose:
                    doc_question = task_doc.get('question', '')
                    print(f"Question {idx}:")
                    print(f"   Question: {doc_question[:60]}...")
                    print(f"   Predicted choice: {best_choice_idx} (F1: {best_score:.3f})")
                    print(f"   Correct choice: {gold_idx}")
                    print(f"   Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}\n")

                evaluation_results.append({
                    **response_data,
                    "evaluation": {
                        "predicted_choice_idx": best_choice_idx,
                        "predicted_choice_text": choice_texts[best_choice_idx] if best_choice_idx is not None else None,
                        "correct_choice_idx": gold_idx,
                        "correct_choice_text": choice_texts[gold_idx],
                        "f1_score": best_score,
                        "correct": is_correct
                    }
                })

            elif evaluation_type == "generate_until":
                # Get expected answer
                expected = None
                if 'answer' in task_doc:
                    expected = task_doc['answer']
                elif 'answers' in task_doc:
                    expected = task_doc['answers']
                elif 'target' in task_doc:
                    expected = task_doc['target']
                else:
                    if args.verbose:
                        print(f"Question {idx}: No expected answer found")
                    evaluation_results.append({
                        **response_data,
                        "evaluation": {
                            "error": "No expected answer in task document"
                        }
                    })
                    continue

                # Evaluate using selected evaluator
                result = evaluator.evaluate(generated_response, expected)

                is_correct = (result.ground_truth == "TRUTHFUL")

                # Store result
                task_results.append({
                    'acc': 1.0 if is_correct else 0.0,
                    'confidence': result.confidence
                })

                if args.verbose:
                    doc_question = task_doc.get('question', '')
                    print(f"Question {idx}:")
                    print(f"   Question: {doc_question[:60]}...")
                    print(f"   Ground truth: {result.ground_truth}")
                    print(f"   Confidence: {result.confidence:.3f}")
                    print(f"   Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}\n")

                evaluation_results.append({
                    **response_data,
                    "evaluation": {
                        "ground_truth": result.ground_truth,
                        "confidence": result.confidence,
                        "details": result.details,
                        "correct": is_correct
                    }
                })

            else:
                # Other evaluation types (loglikelihood_rolling, etc.)
                if args.verbose:
                    print(f"Question {idx}: Evaluation type {evaluation_type} not fully implemented")
                evaluation_results.append({
                    **response_data,
                    "evaluation": {
                        "error": f"Evaluation type {evaluation_type} not implemented"
                    }
                })

        except Exception as e:
            print(f"   ❌ Error evaluating question {idx}: {e}")
            import traceback
            traceback.print_exc()
            evaluation_results.append({
                **response_data,
                "evaluation": {
                    "error": str(e)
                }
            })

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
        "task": input_data.get('task'),
        "model": input_data.get('model'),
        "evaluation_type": evaluation_type,
        "evaluator_used": evaluator.name,
        "aggregated_metrics": aggregated_metrics,
        "num_evaluated": len(task_results),
        "num_total": len(responses),
        "evaluations": evaluation_results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

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
