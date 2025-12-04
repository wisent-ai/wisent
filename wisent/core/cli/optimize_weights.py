"""
Optimize weights command - unified optimization for any trait/task with any evaluator.

This command runs an Optuna-based optimization loop to find optimal weight
modification parameters. Works with:

- Any trait (synthetic pairs): refusal, sycophancy, verbosity, personality
- Any task (lm-eval): hellaswag, arc_easy, gsm8k, etc.
- Any evaluator: refusal detection, task accuracy, semantic similarity, LLM judge

Pipeline per trial:
1. Generate steering vector (from trait or task, or use cached)
2. Load fresh model copy
3. Apply weight modification with trial parameters
4. Evaluate using chosen evaluator
5. Return score to Optuna
6. Repeat until target reached or max trials
7. Apply best parameters and save optimized model
"""

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

from wisent.core.models.wisent_model import WisentModel
from wisent.core.opti.core.atoms import HPOConfig
from wisent.core.opti.methods.opti_weights import WeightsOptimizer, WeightsOptimizerConfig
from wisent.core.utils.device import resolve_default_device, preferred_dtype


@dataclass
class OptimizationResult:
    """Final result from optimization."""
    best_params: dict
    best_score: float
    target_achieved: bool
    total_time: float
    total_trials: int


def execute_optimize_weights(args):
    """
    Execute the optimize-weights command.

    Runs Optuna optimization to find optimal weight modification parameters
    for any trait or task.
    """
    start_time = time.time()

    print(f"\n{'='*80}")
    print("WEIGHT MODIFICATION OPTIMIZATION")
    print(f"{'='*80}")
    print(f"   Model: {args.model}")
    if args.trait:
        print(f"   Mode: Trait-based (synthetic pairs)")
        print(f"   Trait: {args.trait[:60]}...")
    elif args.task:
        print(f"   Mode: Task-based (lm-eval)")
        print(f"   Task: {args.task}")
    else:
        print(f"   Mode: Pre-computed vectors")
        print(f"   Vectors: {args.steering_vectors}")
    # Determine actual evaluator type for display
    evaluator_display = args.evaluator
    if args.evaluator == "auto":
        if args.trait and "refus" in args.trait.lower():
            evaluator_display = "auto → refusal"
        elif args.task:
            evaluator_display = "auto → task"
        elif args.trait:
            evaluator_display = "auto → personalization"
        else:
            evaluator_display = "auto → refusal"
    print(f"   Evaluator: {evaluator_display}")
    print(f"   Target: {args.target_metric} = {args.target_value}")
    print(f"   Trials: {args.trials}")
    print(f"   Output: {args.output_dir}")
    print(f"{'='*80}\n")

    # Parse search space ranges
    strength_range = tuple(float(x) for x in args.strength_range.split(","))
    max_weight_range = tuple(float(x) for x in args.max_weight_range.split(","))
    min_weight_range = tuple(float(x) for x in args.min_weight_range.split(","))
    position_range = tuple(float(x) for x in args.position_range.split(","))
    num_pairs = args.num_pairs

    # Determine optimization direction
    direction = args.direction
    if direction == "auto":
        if args.target_metric in ["refusal_rate", "kl_divergence"]:
            direction = "minimize"
        else:
            direction = "maximize"

    print(f"Search space:")
    print(f"   Strength: {strength_range}")
    print(f"   Max weight: {max_weight_range}")
    print(f"   Min weight: {min_weight_range}")
    print(f"   Position: {position_range}")
    print(f"   Num pairs: {num_pairs} (fixed)")
    print(f"   Direction: {direction}")
    print()

    # Initialize components
    print("Loading base model for reference...")

    device = args.device if args.device else resolve_default_device()

    print(f"   Device: {device}")

    # Load model using WisentModel
    wisent_model = WisentModel(args.model, device=device)
    base_model = wisent_model.hf_model  # Get underlying HF model for weight modification
    tokenizer = wisent_model.tokenizer
    num_layers = wisent_model.num_layers

    print(f"   Model loaded: {num_layers} layers\n")

    # Store base model state for restoration
    base_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}

    # Initialize evaluator (pass wisent_model for personalization baseline generation)
    evaluator = _create_evaluator(args, args.model, wisent_model=wisent_model)

    # Pre-generate steering vectors once
    print(f"Generating steering vectors with {num_pairs} pairs...")
    steering_vectors = _generate_steering_vectors(args, num_pairs, num_layers)
    print(f"   Steering vectors generated for {len(steering_vectors)} layers\n")

    # Create optimizer config
    optimizer_config = WeightsOptimizerConfig(
        strength_range=strength_range,
        max_weight_range=max_weight_range,
        min_weight_range=min_weight_range,
        position_range=position_range,
        method=args.method,
        components=args.components,
        norm_preserve=args.norm_preserve,
        optimize_direction_index=args.optimize_direction_index,
        target_metric=args.target_metric,
        target_value=args.target_value if args.early_stop else None,
    )

    # Create the optimizer
    optimizer = WeightsOptimizer(
        model=base_model,
        base_state_dict=base_state_dict,
        steering_vectors=steering_vectors,
        evaluator=evaluator,
        tokenizer=tokenizer,
        config=optimizer_config,
        num_layers=num_layers,
    )

    # Create HPO config
    hpo_config = HPOConfig(
        n_trials=args.trials,
        direction=direction,
        sampler="tpe",
        pruner=None,
        seed=42,
    )

    # Run optimization
    print(f"\nStarting optimization ({args.trials} trials)...\n")
    result = optimizer.optimize(hpo_config)

    best_params = result.best_params
    best_value = result.best_value

    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nBest parameters:")
    for k, v in best_params.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    print(f"\nBest {args.target_metric}: {best_value:.4f}")

    # Check if target achieved
    if direction == "maximize":
        target_achieved = best_value >= args.target_value
    else:
        target_achieved = best_value <= args.target_value
    print(f"\nTarget {args.target_value} achieved: {'YES' if target_achieved else 'NO'}")

    # Apply best parameters and save final model
    print(f"\n{'='*80}")
    print("SAVING OPTIMIZED MODEL")
    print(f"{'='*80}")

    optimizer.apply_best_params(best_params)

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving optimized model to {args.output_dir}...")
    base_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save optimization metadata
    metadata = {
        "model": args.model,
        "trait": args.trait,
        "task": args.task,
        "evaluator": args.evaluator,
        "target_metric": args.target_metric,
        "target_value": args.target_value,
        "best_params": best_params,
        "best_score": best_value,
        "target_achieved": target_achieved,
        "total_trials": len(result.study.trials),
        "direction": direction,
    }

    with open(os.path.join(args.output_dir, "optimization_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"   Model saved")
    print(f"   Metadata saved to optimization_metadata.json")

    # Save all trials if requested
    if args.save_trials:
        trials_data = [
            {
                "trial": t.number,
                "params": t.params,
                "score": t.value,
            }
            for t in result.study.trials
        ]
        with open(args.save_trials, "w") as f:
            json.dump(trials_data, f, indent=2)
        print(f"   Trials saved to {args.save_trials}")

    # Push to hub if requested
    if args.push_to_hub:
        if not args.repo_id:
            print("\n   ERROR: --repo-id required for --push-to-hub")
        else:
            print(f"\n   Pushing to HuggingFace Hub: {args.repo_id}...")
            base_model.push_to_hub(args.repo_id)
            tokenizer.push_to_hub(args.repo_id)
            print(f"   Pushed successfully")

    # Show/save before/after comparisons if requested
    save_comparisons_path = getattr(args, 'save_comparisons', None)
    show_comparisons_count = getattr(args, 'show_comparisons', 0)
    if show_comparisons_count > 0 or save_comparisons_path:
        _show_response_comparisons(
            base_model=base_model,
            base_state_dict=base_state_dict,
            steering_vectors=steering_vectors,
            best_params=best_params,
            num_layers=num_layers,
            model_name=args.model,
            args=args,
            optimizer_config=optimizer_config,
            num_comparisons=show_comparisons_count if show_comparisons_count > 0 else None,
            save_path=save_comparisons_path,
        )

    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Total optimization time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*80}\n")

    return OptimizationResult(
        best_params=best_params,
        best_score=best_value,
        target_achieved=target_achieved,
        total_time=total_time,
        total_trials=len(result.study.trials),
    )


def _generate_steering_vectors(args, num_pairs: int, num_layers: int) -> dict[int, torch.Tensor]:
    """Generate steering vectors from trait or task."""
    from argparse import Namespace

    if args.steering_vectors:
        with open(args.steering_vectors, "r") as f:
            data = json.load(f)
        return {
            int(layer) - 1: torch.tensor(vec)
            for layer, vec in data["steering_vectors"].items()
        }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_output = f.name

    try:
        if args.trait:
            from wisent.core.cli.generate_vector_from_synthetic import execute_generate_vector_from_synthetic

            vector_args = Namespace(
                trait=args.trait,
                num_pairs=num_pairs,
                output=temp_output,
                model=args.model,
                device=args.device,
                similarity_threshold=args.similarity_threshold,
                verbose=False,
                timing=False,
                layers=args.layers,
                token_aggregation=args.token_aggregation,
                prompt_strategy="chat_template",
                method="caa",
                normalize=True,
                keep_intermediate=False,
                intermediate_dir=None,
                pairs_cache_dir=args.pairs_cache_dir,
                force_regenerate=False,
                nonsense=False,
                nonsense_mode=None,
            )

            execute_generate_vector_from_synthetic(vector_args)

        elif args.task:
            from wisent.core.cli.generate_vector_from_task import execute_generate_vector_from_task

            vector_args = Namespace(
                task=args.task,
                trait_label="correctness",
                num_pairs=num_pairs,
                output=temp_output,
                model=args.model,
                device=args.device,
                verbose=False,
                timing=False,
                layers=args.layers,
                token_aggregation=args.token_aggregation,
                prompt_strategy="chat_template",
                method="caa",
                normalize=True,
                keep_intermediate=False,
                intermediate_dir=None,
            )

            execute_generate_vector_from_task(vector_args)

        with open(temp_output, "r") as f:
            data = json.load(f)

        return {
            int(layer) - 1: torch.tensor(vec)
            for layer, vec in data["steering_vectors"].items()
        }

    finally:
        if os.path.exists(temp_output):
            os.unlink(temp_output)


def _create_evaluator(args, model_name: str, wisent_model: WisentModel = None) -> Callable[[Any, Any], dict[str, float]]:
    """Create the appropriate evaluator function based on args.

    The evaluator receives (hf_model, tokenizer) and wraps them in WisentModel
    internally to use standard generation.

    Args:
        args: Command arguments
        model_name: Model name/path
        wisent_model: Optional WisentModel instance for baseline generation (used by personalization)
    """

    evaluator_type = args.evaluator

    if evaluator_type == "auto":
        if args.trait and "refus" in args.trait.lower():
            evaluator_type = "refusal"
        elif args.task:
            evaluator_type = "task"
        elif args.trait:
            # For any other trait (personality, style, etc.), use personalization evaluator
            evaluator_type = "personalization"
        else:
            evaluator_type = "refusal"

    if evaluator_type in ["refusal", "semantic", "keyword"]:
        return _create_refusal_evaluator(args, model_name, evaluator_type)
    elif evaluator_type == "task":
        return _create_task_evaluator(args)
    elif evaluator_type == "llm_judge":
        return _create_llm_judge_evaluator(args, model_name)
    elif evaluator_type == "personalization":
        return _create_personalization_evaluator(args, model_name, wisent_model)
    else:
        raise ValueError(f"Unknown evaluator: {evaluator_type}")


def _create_refusal_evaluator(args, model_name: str, eval_type: str) -> Callable:
    """Create refusal/compliance evaluator.

    Uses WisentModel.generate() for consistent generation behavior.
    """
    from uncensorbench import UncensorBench
    from uncensorbench.evaluator import KeywordEvaluator, SemanticEvaluator
    from wisent.core.models.inference_config import get_generate_kwargs

    bench = UncensorBench()

    if args.eval_prompts:
        with open(args.eval_prompts) as f:
            custom_prompts = json.load(f)
        if not isinstance(custom_prompts, list):
            custom_prompts = custom_prompts.get("prompts", [])
        prompts = custom_prompts[:args.num_eval_prompts]
        use_custom = True
    else:
        topics = args.eval_topics.split(",") if args.eval_topics else None
        prompts = list(bench.prompts(topics=topics))[:args.num_eval_prompts]
        use_custom = False

    if eval_type == "keyword":
        evaluator = KeywordEvaluator()
    else:
        try:
            evaluator = SemanticEvaluator()
        except ImportError:
            evaluator = KeywordEvaluator()

    def evaluate(hf_model, tokenizer) -> dict:
        """Run refusal evaluation using WisentModel for generation."""
        # Wrap HF model in WisentModel for standard generation
        wisent_model = WisentModel(model_name, hf_model=hf_model)

        refusal_count = 0
        compliance_scores = []

        for item in prompts:
            if use_custom:
                prompt_text = item if isinstance(item, str) else item.get("prompt", str(item))
                prompt_obj = None
            else:
                prompt_obj = item
                prompt_text = prompt_obj.prompt

            # Use WisentModel.generate() with standard chat format
            messages = [{"role": "user", "content": prompt_text}]
            responses = wisent_model.generate(
                [messages],
                **get_generate_kwargs(max_new_tokens=150),
            )

            response = responses[0] if responses else ""

            if prompt_obj:
                eval_result = bench.evaluate_response(prompt_obj, response, evaluator)
                is_refusal = not eval_result.is_uncensored
                compliance_score = eval_result.compliance_score
            else:
                is_refusal = bench.detect_refusal(response)
                compliance_score = 0.0 if is_refusal else 1.0

            if is_refusal:
                refusal_count += 1
            compliance_scores.append(compliance_score)

        total = len(prompts)
        refusal_rate = refusal_count / total
        compliance_rate = 1 - refusal_rate
        avg_compliance = sum(compliance_scores) / len(compliance_scores)

        return {
            "refusal_rate": refusal_rate,
            "compliance_rate": compliance_rate,
            "average_compliance_score": avg_compliance,
            "refusal_count": refusal_count,
            "total": total,
            "score": compliance_rate,
        }

    return evaluate


def _create_task_evaluator(args) -> Callable:
    """Create task-based evaluator."""
    from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
    from wisent.core.evaluators.rotator import EvaluatorRotator

    loader = LMEvalDataLoader()
    EvaluatorRotator.discover_evaluators('wisent.core.evaluators.benchmark_specific')

    def evaluate(model, tokenizer) -> dict:
        """Run task evaluation."""
        result = loader._load_one_task(
            task_name=args.task,
            split_ratio=0.8,
            seed=42,
            limit=args.num_eval_prompts,
        )

        test_pairs = result["test_qa_pairs"]
        evaluator = EvaluatorRotator(evaluator=None, task_name=args.task)

        correct = 0
        total = 0

        for pair in test_pairs.pairs:
            choices = [pair.negative_response.model_response, pair.positive_response.model_response]
            expected = pair.positive_response.model_response

            try:
                eval_result = evaluator.evaluate(
                    response="",
                    expected=expected,
                    model=None,
                    question=pair.prompt,
                    choices=choices,
                    task_name=args.task,
                )

                if eval_result.ground_truth == "TRUTHFUL":
                    correct += 1
                total += 1
            except Exception:
                total += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "score": accuracy,
        }

    return evaluate


def _create_llm_judge_evaluator(args, model_name: str) -> Callable:
    """Create LLM-as-judge evaluator.

    Uses WisentModel.generate() for consistent generation behavior.
    """
    raise NotImplementedError("LLM judge evaluator not yet implemented")


def _create_personalization_evaluator(args, model_name: str, wisent_model: WisentModel = None) -> Callable:
    """Create personalization evaluator for personality/style traits.

    Evaluates steering effectiveness on three criteria:
    1. Difference: Is the steered response different from baseline?
    2. Quality: Is the response coherent (not lobotomized)?
    3. Alignment: Does the response match the intended trait direction?

    Uses the same evaluation pattern as optimize_steering's personalization action.
    Baseline responses are generated once upfront using the unmodified model.
    """
    from wisent.core.evaluators.personalization_evaluator import PersonalizationEvaluator
    from wisent.core.models.inference_config import get_generate_kwargs

    # Get trait name and description from args
    trait_name = args.trait.split()[0] if args.trait else "unknown"
    trait_description = args.trait

    # Default test prompts for personalization evaluation
    default_prompts = [
        "Tell me about yourself.",
        "What do you think about the weather today?",
        "Can you help me write a short story?",
        "What's your opinion on modern technology?",
        "How would you describe a perfect day?",
        "Tell me a joke.",
        "What advice would you give to someone starting a new job?",
        "Describe your favorite book or movie.",
        "What do you think makes a good friend?",
        "How do you handle stress?",
        "What's the best way to learn a new skill?",
        "Tell me about a memorable experience.",
        "What do you value most in life?",
        "How would you explain your personality?",
        "What makes you happy?",
    ]

    # Use custom prompts if provided, otherwise use defaults
    if args.eval_prompts:
        with open(args.eval_prompts) as f:
            custom_prompts = json.load(f)
        if not isinstance(custom_prompts, list):
            custom_prompts = custom_prompts.get("prompts", [])
        prompts = custom_prompts[:args.num_eval_prompts]
    else:
        prompts = default_prompts[:args.num_eval_prompts]

    # Generate baseline responses ONCE using the unmodified model
    # This is done before optimization starts
    baseline_responses = []
    if wisent_model is not None:
        print(f"   Generating {len(prompts)} baseline responses for personalization evaluation...")
        for prompt_text in prompts:
            messages = [{"role": "user", "content": prompt_text}]
            responses = wisent_model.generate(
                [messages],
                max_new_tokens=150,
            )
            response = responses[0] if responses else ""
            baseline_responses.append(response)
        print(f"   ✓ Baseline responses generated\n")

    def evaluate(hf_model, tokenizer) -> dict:
        """Run personalization evaluation comparing baseline vs steered responses."""
        # Wrap HF model in WisentModel for standard generation
        modified_wisent_model = WisentModel(model_name, hf_model=hf_model)

        # Collect steered responses (model already has weight modifications applied)
        steered_responses = []
        for prompt_text in prompts:
            messages = [{"role": "user", "content": prompt_text}]
            responses = modified_wisent_model.generate(
                [messages],
                max_new_tokens=150,
            )
            response = responses[0] if responses else ""
            steered_responses.append(response)

        # Initialize evaluator (no model needed for basic metrics)
        evaluator = PersonalizationEvaluator()

        # Calculate difference score (baseline vs steered)
        if baseline_responses:
            difference_score = evaluator._evaluate_difference(baseline_responses, steered_responses)
        else:
            difference_score = 0.5  # Default if no baseline available

        # Calculate quality score (coherence check)
        quality_score = evaluator._evaluate_quality(steered_responses)

        # Calculate alignment score using keyword-based estimation
        alignment_score = PersonalizationEvaluator.estimate_alignment(steered_responses, trait_description)

        # Calculate overall score (weighted average)
        # Only count if difference > 0.3 (steering is actually doing something)
        if difference_score < 0.3:
            overall_score = 0.0
        else:
            overall_score = 0.2 * difference_score + 0.3 * quality_score + 0.5 * alignment_score

        return {
            "difference_score": difference_score,
            "quality_score": quality_score,
            "alignment_score": alignment_score,
            "overall_score": overall_score,
            "num_responses": len(steered_responses),
            "score": overall_score,  # Main metric for optimization
        }

    return evaluate


def _apply_weight_modification_standalone(
    model,
    steering_vectors: dict[int, torch.Tensor],
    params: dict,
    num_layers: int,
    config: WeightsOptimizerConfig,
):
    """Apply weight modification with given parameters (standalone helper)."""
    from wisent.core.weight_modification import abliterate_with_kernel, bake_steering_with_kernel

    max_weight_position = params["max_weight_position"] * (num_layers - 1)
    min_weight_distance = 0.6 * (num_layers - 1)

    if config.method == "abliteration":
        abliterate_with_kernel(
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
    else:
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
    from wisent.core.models.inference_config import get_generate_kwargs

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
            **get_generate_kwargs(max_new_tokens=150),
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
            json.dump(output_data, f, indent=2)
        print(f"\nSaved {len(comparisons)} comparisons to: {save_path}")

    if num_comparisons and num_comparisons > 0:
        print(f"\nTop {num_comparisons} prompts with biggest score changes:\n")

        for i, comp in enumerate(comparisons[:num_comparisons]):
            print(f"{'─'*80}")
            print(f"Comparison {i+1}/{num_comparisons}")
            print(f"{'─'*80}")
            print(f"PROMPT: {comp['prompt'][:200]}{'...' if len(comp['prompt']) > 200 else ''}")
            print()
            print(f"BASELINE (score={comp['baseline_score']:.2f}, refusal={comp['baseline_refusal']}):")
            print(f"  {comp['baseline_response'][:300]}{'...' if len(comp['baseline_response']) > 300 else ''}")
            print()
            print(f"OPTIMIZED (score={comp['optimized_score']:.2f}, refusal={comp['optimized_refusal']}):")
            print(f"  {comp['optimized_response'][:300]}{'...' if len(comp['optimized_response']) > 300 else ''}")
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
