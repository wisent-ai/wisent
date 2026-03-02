"""Personalization evaluation for evaluate-responses command."""
import json
import os
from wisent.core.utils.config_tools.constants import JSON_INDENT, QUALITY_THRESHOLD, DEFAULT_SCORE, PERSONALIZATION_GOOD_THRESHOLD, SCORE_MIDPOINT_PCT

from wisent.core.reading.evaluators.steering_evaluators import (
    SteeringEvaluatorFactory,
    EvaluatorConfig,
    PersonalizationEvaluator as SteeringPersonalizationEvaluator,
)


def evaluate_personalization(args, input_data, responses, task_name, evaluation_results, task_results):
    """Handle personalization evaluation.

    Returns aggregated_metrics dict or None.
    """
    print(f"🎭 Running personality trait evaluation using shared steering evaluators...")

    # Check if baseline is provided
    if not hasattr(args, 'baseline') or not args.baseline:
        print(f"   ❌ Error: --baseline argument is required for personalization evaluation")
        raise ValueError("--baseline argument is required for personalization evaluation")

    # Load baseline responses
    print(f"📂 Loading baseline responses...")
    try:
        with open(args.baseline, 'r') as f:
            baseline_data = json.load(f)

        if isinstance(baseline_data, list):
            baseline_responses = baseline_data
        else:
            baseline_responses = baseline_data.get('responses', [])

        print(f"   ✓ Loaded {len(baseline_responses)} baseline responses\n")
    except Exception as e:
        print(f"   ❌ Failed to load baseline file: {e}")
        raise

    # Check lengths match
    if len(baseline_responses) != len(responses):
        raise ValueError(f"Baseline ({len(baseline_responses)}) and steered ({len(responses)}) response counts don't match")

    # Get trait information
    trait = args.trait if hasattr(args, 'trait') and args.trait else "unknown"
    trait_description = args.trait_description if hasattr(args, 'trait_description') and args.trait_description else f"The trait: {trait}"

    print(f"   Target trait: {trait}")
    print(f"   Trait description: {trait_description}")
    print(f"   Evaluating {len(responses)} response pairs...\n")

    # Load model for evaluation
    print(f"📦 Loading model for self-evaluation...")
    if isinstance(input_data, dict):
        model_name = input_data.get('model', 'meta-llama/Llama-3.2-1B-Instruct')
    else:
        model_name = 'meta-llama/Llama-3.2-1B-Instruct'  # Default model
    print(f"   Model: {model_name}")

    from wisent.core.primitives.models.wisent_model import WisentModel

    wisent_model = WisentModel(model_name, device=getattr(args, 'device', None))
    model = wisent_model.hf_model
    tokenizer = wisent_model.tokenizer
    device = wisent_model.device

    print(f"   ✓ Model loaded with {wisent_model.num_layers} layers\n")

    # Initialize shared steering evaluator for personalization
    eval_config = EvaluatorConfig(
        evaluator_type="personalization",
        trait=trait,
        num_eval_prompts=len(responses),
    )
    steering_evaluator = SteeringPersonalizationEvaluator(
        eval_config, model_name, wisent_model=wisent_model
    )
    # Set baseline responses for comparison
    baseline_texts = [b.get('generated_response', '') for b in baseline_responses]
    steering_evaluator._baseline_responses = baseline_texts

    evaluated_count = 0
    difference_scores = []
    quality_scores = []
    alignment_scores = []
    overall_scores = []

    # Collect all steered responses for batch evaluation
    steered_texts = [s.get('generated_response', '') for s in responses]
    
    # Evaluate using shared evaluator
    eval_results = steering_evaluator.evaluate_responses(steered_texts)
    
    # Process individual results
    for idx, (baseline_data, steered_data) in enumerate(zip(baseline_responses, responses), 1):
        # Check for errors in either response
        if 'error' in baseline_data or 'error' in steered_data:
            if args.verbose:
                print(f"Response pair {idx}: Skipped (generation error)")
            evaluation_results.append({
                "baseline": baseline_data,
                "steered": steered_data,
                "evaluation": {
                    "error": "Generation failed"
                }
            })
            continue

        try:
            baseline_response = baseline_data.get('generated_response', '')
            steered_response = steered_data.get('generated_response', '')
            prompt = steered_data.get('prompt', '')

            evaluated_count += 1

            # Use aggregate scores from batch evaluation
            diff_score = eval_results.get('difference_score', QUALITY_THRESHOLD)
            qual_score = eval_results.get('quality_score', QUALITY_THRESHOLD)
            align_score = eval_results.get('alignment_score', QUALITY_THRESHOLD)
            overall = eval_results.get('overall_score', DEFAULT_SCORE)

            # Collect scores
            difference_scores.append(diff_score)
            quality_scores.append(qual_score)
            alignment_scores.append(align_score)
            overall_scores.append(overall)

            # Store result
            eval_result = {
                'response_id': steered_data.get('id', idx),
                'prompt': prompt,
                'trait': trait,
                'difference_score': diff_score,
                'quality_score': qual_score,
                'alignment_score': align_score,
                'overall_score': overall,
                'baseline_response': baseline_response,
                'steered_response': steered_response,
            }

            evaluation_results.append(eval_result)
            task_results.append({
                'difference_score': diff_score,
                'quality_score': qual_score,
                'alignment_score': align_score,
                'overall_score': overall,
            })

            if args.verbose:
                score_icon = '✅' if overall >= PERSONALIZATION_GOOD_THRESHOLD else ('⚠️' if overall >= SCORE_MIDPOINT_PCT else '❌')
                print(f"{score_icon} Pair {idx}: Overall={overall:.1f} (diff={diff_score:.1f}, qual={qual_score:.1f}, align={align_score:.1f})")

        except Exception as e:
            import traceback
            traceback.print_exc()
            evaluation_results.append({
                "baseline": baseline_data,
                "steered": steered_data,
                "evaluation": {
                    "error": str(e)
                }
            })

    print(f"\n   ✓ Evaluated {evaluated_count} response pairs\n")

    # Aggregate results
    aggregated_metrics = {}
    if task_results:
        avg_difference = sum(r['difference_score'] for r in task_results) / len(task_results)
        avg_quality = sum(r['quality_score'] for r in task_results) / len(task_results)
        avg_alignment = sum(r['alignment_score'] for r in task_results) / len(task_results)
        avg_overall = sum(r['overall_score'] for r in task_results) / len(task_results)

        aggregated_metrics['avg_difference_score'] = avg_difference
        aggregated_metrics['avg_quality_score'] = avg_quality
        aggregated_metrics['avg_alignment_score'] = avg_alignment
        aggregated_metrics['avg_overall_score'] = avg_overall
        aggregated_metrics['total_evaluated'] = len(task_results)

    # Save results
    print(f"💾 Saving evaluation results...")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_data = {
        "input_file": args.input,
        "baseline_file": args.baseline,
        "task": task_name if isinstance(input_data, list) else input_data.get('task'),
        "model": None if isinstance(input_data, list) else input_data.get('model'),
        "evaluation_type": "personalization",
        "evaluator_used": "SteeringPersonalizationEvaluator",
        "trait": trait,
        "trait_description": trait_description,
        "aggregated_metrics": aggregated_metrics,
        "num_evaluated": len(task_results),
        "num_total": len(responses),
        "evaluations": evaluation_results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=JSON_INDENT)

    print(f"   ✓ Results saved to: {args.output}\n")
    print(f"{'='*80}")
    print(f"✅ PERSONALIZATION EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"   Trait: {trait}")
    print(f"   Total response pairs: {len(task_results)}")
    print(f"   Average difference score: {aggregated_metrics.get('avg_difference_score', 0):.3f}")
    print(f"   Average quality score: {aggregated_metrics.get('avg_quality_score', 0):.3f}")
    print(f"   Average alignment score: {aggregated_metrics.get('avg_alignment_score', 0):.3f}")
    print(f"   Average overall score: {aggregated_metrics.get('avg_overall_score', 0):.3f}")
    print(f"{'='*80}\n")
    return aggregated_metrics
