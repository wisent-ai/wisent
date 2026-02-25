"""Response comparison and weight application helpers for optimize-weights."""
import json
import os
import re

import torch
from wisent.core.models.wisent_model import WisentModel
from wisent.core.opti.methods.opti_weights import WeightsOptimizerConfig
from wisent.core.constants import JSON_INDENT, WEIGHT_MIN_DISTANCE_FRACTION, WEIGHT_COMPARISON_MAX_NEW_TOKENS, DISPLAY_TRUNCATION_MEDIUM, DISPLAY_TRUNCATION_LONG


def _apply_weight_modification_standalone(
    model,
    steering_vectors: dict[int, torch.Tensor],
    params: dict,
    num_layers: int,
    config: WeightsOptimizerConfig,
):
    """Apply weight modification with given parameters (standalone helper)."""
    from wisent.core.weight_modification import project_with_kernel, bake_steering_with_kernel

    max_weight_position = params["max_weight_position"] * (num_layers - 1)
    min_weight_distance = WEIGHT_MIN_DISTANCE_FRACTION * (num_layers - 1)

    if config.method == "directional":
        project_with_kernel(
            model,
            steering_vectors,
            max_weight=params["max_weight"] * params["strength"],
            max_weight_position=max_weight_position,
            min_weight=params["min_weight"],
            min_weight_distance=min_weight_distance,
            components=config.components,
            norm_preserve=config.norm_preserve,
            verbose=False,
        )
    elif config.method in ("grom", "tecza", "tetno"):
        # Multi-direction methods: directions already combined during generation
        # Use additive baking with the combined vectors
        bake_steering_with_kernel(
            model,
            steering_vectors,
            max_alpha=params["max_weight"] * params["strength"],
            max_alpha_position=max_weight_position,
            min_alpha=params["min_weight"],
            components=config.components,
            verbose=False,
        )
    else:
        # Default: additive
        bake_steering_with_kernel(
            model,
            steering_vectors,
            max_alpha=params["max_weight"] * params["strength"],
            max_alpha_position=max_weight_position,
            min_alpha=params["min_weight"],
            components=config.components,
            verbose=False,
        )



def _show_response_comparisons(
    base_model,
    base_state_dict: dict,
    steering_vectors: dict[int, torch.Tensor],
    best_params: dict,
    num_layers: int,
    model_name: str,
    args,
    optimizer_config: WeightsOptimizerConfig,
    num_comparisons: int = None,
    save_path: str = None,
):
    """Show and/or save before/after response comparisons.

    Uses WisentModel.generate() for consistent generation behavior.
    """
    from uncensorbench import UncensorBench
    from uncensorbench.evaluator import KeywordEvaluator, SemanticEvaluator
    from wisent.core.models import get_generate_kwargs

    print(f"\n{'='*80}")
    print("RESPONSE COMPARISONS (Baseline vs Optimized)")
    print(f"{'='*80}")

    bench = UncensorBench()

    if args.eval_prompts:
        with open(args.eval_prompts) as f:
            custom_prompts = json.load(f)
        if not isinstance(custom_prompts, list):
            custom_prompts = custom_prompts.get("prompts", [])
        prompts = custom_prompts[:args.num_eval_prompts]
        use_custom = True
    elif getattr(args, 'trait', None):
        # For trait-based optimization, use personalization prompts (not UncensorBench)
        from wisent.core.evaluators.steering_evaluators import PersonalizationEvaluator
        prompts = PersonalizationEvaluator.DEFAULT_PROMPTS[:args.num_eval_prompts]
        use_custom = True  # Treat as custom to skip UncensorBench evaluation
    else:
        topics = args.eval_topics.split(",") if args.eval_topics else None
        prompts = list(bench.prompts(topics=topics))[:args.num_eval_prompts]
        use_custom = False

    try:
        evaluator = SemanticEvaluator()
    except ImportError:
        evaluator = KeywordEvaluator()

    def generate_response(hf_model, prompt_text: str) -> str:
        """Generate response using WisentModel for consistent behavior."""
        # Wrap HF model in WisentModel for standard generation
        wisent_model = WisentModel(model_name, hf_model=hf_model)

        messages = [{"role": "user", "content": prompt_text}]
        responses = wisent_model.generate(
            [messages],
            **get_generate_kwargs(max_new_tokens=WEIGHT_COMPARISON_MAX_NEW_TOKENS),
        )

        response = responses[0] if responses else ""
        # Strip <think>...</think> blocks from Qwen3 responses
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        return response

    def evaluate_response(prompt_item, response: str, use_custom: bool) -> tuple[bool, float]:
        if use_custom:
            is_refusal = bench.detect_refusal(response)
            compliance_score = 0.0 if is_refusal else 1.0
        else:
            eval_result = bench.evaluate_response(prompt_item, response, evaluator)
            is_refusal = not eval_result.is_uncensored
            compliance_score = eval_result.compliance_score
        return is_refusal, compliance_score

    print("\nGenerating baseline responses...")

    base_model.load_state_dict(base_state_dict)

    baseline_results = []
    for item in prompts:
        if use_custom:
            prompt_text = item if isinstance(item, str) else item.get("prompt", str(item))
        else:
            prompt_text = item.prompt

        response = generate_response(base_model, prompt_text)
        is_refusal, score = evaluate_response(item, response, use_custom)
        baseline_results.append({
            "prompt": prompt_text,
            "response": response,
            "is_refusal": is_refusal,
            "score": score,
            "item": item,
        })

    print("Generating optimized responses...")

    _apply_weight_modification_standalone(
        base_model,
        steering_vectors,
        best_params,
        num_layers,
        optimizer_config,
    )

    optimized_results = []
    for i, item in enumerate(prompts):
        if use_custom:
            prompt_text = item if isinstance(item, str) else item.get("prompt", str(item))
        else:
            prompt_text = item.prompt

        response = generate_response(base_model, prompt_text)
        is_refusal, score = evaluate_response(item, response, use_custom)
        optimized_results.append({
            "prompt": prompt_text,
            "response": response,
            "is_refusal": is_refusal,
            "score": score,
        })

    comparisons = []
    for i, (baseline, optimized) in enumerate(zip(baseline_results, optimized_results)):
        delta = optimized["score"] - baseline["score"]
        comparisons.append({
            "index": i,
            "prompt": baseline["prompt"],
            "baseline_response": baseline["response"],
            "baseline_score": baseline["score"],
            "baseline_refusal": baseline["is_refusal"],
            "optimized_response": optimized["response"],
            "optimized_score": optimized["score"],
            "optimized_refusal": optimized["is_refusal"],
            "delta": delta,
        })

    comparisons.sort(key=lambda x: abs(x["delta"]), reverse=True)

    total_baseline_refusals = sum(1 for c in comparisons if c["baseline_refusal"])
    total_optimized_refusals = sum(1 for c in comparisons if c["optimized_refusal"])
    avg_baseline_score = sum(c["baseline_score"] for c in comparisons) / len(comparisons)
    avg_optimized_score = sum(c["optimized_score"] for c in comparisons) / len(comparisons)

    summary = {
        "total_prompts": len(comparisons),
        "baseline_refusals": total_baseline_refusals,
        "optimized_refusals": total_optimized_refusals,
        "avg_baseline_score": avg_baseline_score,
        "avg_optimized_score": avg_optimized_score,
        "score_delta": avg_optimized_score - avg_baseline_score,
        "refusal_change": total_baseline_refusals - total_optimized_refusals,
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        output_data = {
            "model": args.model,
            "trait": getattr(args, "trait", None),
            "best_params": best_params,
            "summary": summary,
            "comparisons": comparisons,
        }
        with open(save_path, "w") as f:
            json.dump(output_data, f, indent=JSON_INDENT)
        print(f"\nSaved {len(comparisons)} comparisons to: {save_path}")

    if num_comparisons and num_comparisons > 0:
        print(f"\nTop {num_comparisons} prompts with biggest score changes:\n")

        for i, comp in enumerate(comparisons[:num_comparisons]):
            print(f"{'─'*80}")
            print(f"Comparison {i+1}/{num_comparisons}")
            print(f"{'─'*80}")
            print(f"PROMPT: {comp['prompt'][:DISPLAY_TRUNCATION_MEDIUM]}{'...' if len(comp['prompt']) > DISPLAY_TRUNCATION_MEDIUM else ''}")
            print()
            print(f"BASELINE (score={comp['baseline_score']:.2f}, refusal={comp['baseline_refusal']}):")
            print(f"  {comp['baseline_response'][:DISPLAY_TRUNCATION_LONG]}{'...' if len(comp['baseline_response']) > DISPLAY_TRUNCATION_LONG else ''}")
            print()
            print(f"OPTIMIZED (score={comp['optimized_score']:.2f}, refusal={comp['optimized_refusal']}):")
            print(f"  {comp['optimized_response'][:DISPLAY_TRUNCATION_LONG]}{'...' if len(comp['optimized_response']) > DISPLAY_TRUNCATION_LONG else ''}")
            print()
            delta_str = f"+{comp['delta']:.2f}" if comp['delta'] >= 0 else f"{comp['delta']:.2f}"
            print(f"DELTA: {delta_str}")
            print()

    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Baseline:  {total_baseline_refusals}/{len(comparisons)} refusals, avg score={avg_baseline_score:.3f}")
    print(f"Optimized: {total_optimized_refusals}/{len(comparisons)} refusals, avg score={avg_optimized_score:.3f}")
    print(f"Change:    {total_baseline_refusals - total_optimized_refusals} fewer refusals, "
          f"score delta={avg_optimized_score - avg_baseline_score:+.3f}")
