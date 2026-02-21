"""Refusal evaluation for evaluate-responses command."""
from __future__ import annotations

import json
import os

from wisent.core.evaluators.steering_evaluators import (
    EvaluatorConfig,
    RefusalEvaluator,
)


def execute_evaluate_refusal(args):
    """Execute the evaluate-refusal standalone command."""
    with open(args.input, 'r') as f:
        input_data = json.load(f)

    if isinstance(input_data, list):
        responses = input_data
        task_name = getattr(args, 'task', 'unknown')
    else:
        responses = input_data.get('responses', input_data.get('evaluations', []))
        task_name = input_data.get('task', getattr(args, 'task', 'unknown'))

    evaluation_results = []
    task_results = []
    evaluate_refusal(args, input_data, responses, task_name, evaluation_results, task_results)


def evaluate_refusal(args, input_data, responses, task_name, evaluation_results, task_results):
    """Handle refusal evaluation. Returns aggregated_metrics or None."""
    print(f"🚫 Running refusal/compliance evaluation using shared steering evaluators...")

    # Get model name for evaluator
    if isinstance(input_data, dict):
        model_name = input_data.get('model', 'meta-llama/Llama-3.2-1B-Instruct')
    else:
        model_name = 'meta-llama/Llama-3.2-1B-Instruct'

    # Initialize shared steering evaluator for refusal
    eval_config = EvaluatorConfig(
        evaluator_type="refusal",
        eval_prompts_path=getattr(args, 'eval_prompts', None),
        num_eval_prompts=len(responses),
    )
    steering_evaluator = RefusalEvaluator(eval_config, model_name)

    # Collect all responses for batch evaluation
    response_texts = [r.get('generated_response', '') for r in responses]

    # Evaluate using shared evaluator
    eval_results = steering_evaluator.evaluate_responses(response_texts)

    # Process individual results
    refusal_count = 0
    for idx, response_data in enumerate(responses, 1):
        if 'error' in response_data:
            if args.verbose:
                print(f"Response {idx}: Skipped (generation error)")
            evaluation_results.append({
                **response_data,
                "evaluation": {"error": "Generation failed"}
            })
            continue

        response_text = response_data.get('generated_response', '')
        is_refusal = steering_evaluator.uncensor_bench.detect_refusal(response_text)
        
        if is_refusal:
            refusal_count += 1

        evaluation_results.append({
            **response_data,
            "evaluation": {
                "is_refusal": is_refusal,
                "compliance": not is_refusal,
            }
        })
        task_results.append({
            'compliance': 0.0 if is_refusal else 1.0
        })

        if args.verbose:
            status = '🚫 REFUSED' if is_refusal else '✅ COMPLIED'
            print(f"Response {idx}: {status}")

    # Aggregate results
    total = len(task_results)
    compliance_rate = eval_results.get('compliance_rate', 0.0)
    refusal_rate = eval_results.get('refusal_rate', 0.0)

    aggregated_metrics = {
        'compliance_rate': compliance_rate,
        'refusal_rate': refusal_rate,
        'refusal_count': refusal_count,
        'total': total,
    }

    # Save results
    print(f"\n💾 Saving evaluation results...")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_data = {
        "input_file": args.input,
        "task": task_name if isinstance(input_data, list) else input_data.get('task'),
        "model": model_name,
        "evaluation_type": "refusal",
        "evaluator_used": "RefusalEvaluator",
        "aggregated_metrics": aggregated_metrics,
        "num_evaluated": total,
        "num_total": len(responses),
        "evaluations": evaluation_results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"   ✓ Results saved to: {args.output}\n")
    print(f"{'='*80}")
    print(f"✅ REFUSAL EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"   Total responses: {total}")
    print(f"   Refusals: {refusal_count}")
    print(f"   Compliant: {total - refusal_count}")
    print(f"   Compliance rate: {compliance_rate:.2%}")
    print(f"   Refusal rate: {refusal_rate:.2%}")
    print(f"{'='*80}\n")
    return aggregated_metrics
