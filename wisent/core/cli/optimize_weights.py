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
import sys
import time
import copy
import tempfile
from dataclasses import dataclass
from typing import Callable, Any

import optuna
from optuna.samplers import TPESampler
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TrialResult:
    """Result from a single optimization trial."""
    trial_number: int
    params: dict
    score: float
    evaluation_details: dict


@dataclass
class OptimizationResult:
    """Final result from optimization."""
    best_trial: TrialResult
    all_trials: list[TrialResult]
    best_params: dict
    target_achieved: bool
    total_time: float


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
    print(f"   Evaluator: {args.evaluator}")
    print(f"   Target: {args.target_metric} = {args.target_value}")
    print(f"   Trials: {args.trials}")
    print(f"   Output: {args.output_dir}")
    print(f"{'='*80}\n")

    # Parse search space ranges
    strength_range = [float(x) for x in args.strength_range.split(",")]
    max_weight_range = [float(x) for x in args.max_weight_range.split(",")]
    min_weight_range = [float(x) for x in args.min_weight_range.split(",")]
    position_range = [float(x) for x in args.position_range.split(",")]
    num_pairs_range = [int(x) for x in args.num_pairs_range.split(",")]

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
    print(f"   Num pairs: {num_pairs_range}")
    print(f"   Direction: {direction}")
    print()

    # Initialize components
    print("Loading base model for reference...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto" if args.device is None else args.device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get number of layers
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        num_layers = len(base_model.model.layers)
    elif hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
        num_layers = len(base_model.transformer.h)
    else:
        num_layers = 16  # fallback
    print(f"   Model loaded: {num_layers} layers\n")

    # Store base model state for restoration
    base_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}

    # Initialize evaluator
    evaluator = _create_evaluator(args, tokenizer)

    # Pre-generate steering vectors if using trait/task (can be reused across trials)
    # For num_pairs optimization, we'll regenerate with different sizes
    cached_vectors = {}

    # Track all trial results
    all_trials: list[TrialResult] = []
    best_result: TrialResult | None = None

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        nonlocal best_result

        trial_start = time.time()

        # Suggest parameters
        params = {
            "strength": trial.suggest_float("strength", strength_range[0], strength_range[1]),
            "max_weight": trial.suggest_float("max_weight", max_weight_range[0], max_weight_range[1]),
            "min_weight": trial.suggest_float("min_weight", min_weight_range[0], min_weight_range[1]),
            "max_weight_position": trial.suggest_float("max_weight_position", position_range[0], position_range[1]),
            "num_pairs": trial.suggest_int("num_pairs", num_pairs_range[0], num_pairs_range[1], step=10),
        }

        # Optional: optimize direction index (interpolate between layers)
        if args.optimize_direction_index:
            params["direction_index"] = trial.suggest_float("direction_index", 0.0, num_layers - 1)

        if args.verbose:
            print(f"\n{'─'*60}")
            print(f"Trial {trial.number}: {params}")

        try:
            # Step 1: Get or generate steering vectors
            num_pairs = params["num_pairs"]
            if num_pairs not in cached_vectors:
                vectors = _generate_steering_vectors(args, num_pairs, num_layers)
                cached_vectors[num_pairs] = vectors
            steering_vectors = cached_vectors[num_pairs]

            # Step 2: Restore base model weights
            base_model.load_state_dict(base_state_dict)

            # Step 3: Apply weight modification with trial parameters
            _apply_weight_modification(
                base_model,
                steering_vectors,
                params,
                num_layers,
                args,
            )

            # Step 4: Evaluate
            eval_result = evaluator(base_model, tokenizer)

            # Step 5: Get score
            score = eval_result.get(args.target_metric, eval_result.get("score", 0.0))

            # Create trial result
            trial_result = TrialResult(
                trial_number=trial.number,
                params=params,
                score=score,
                evaluation_details=eval_result,
            )
            all_trials.append(trial_result)

            # Track best
            if best_result is None:
                best_result = trial_result
            else:
                is_better = (score > best_result.score) if direction == "maximize" else (score < best_result.score)
                if is_better:
                    best_result = trial_result

            trial_time = time.time() - trial_start

            if args.verbose:
                print(f"   {args.target_metric}: {score:.4f}")
                print(f"   Time: {trial_time:.1f}s")
            else:
                status = "BEST" if trial_result == best_result else ""
                print(f"Trial {trial.number:3d}: {args.target_metric}={score:.4f} {status}")

            # Early stopping check
            if args.early_stop:
                target_reached = (score >= args.target_value) if direction == "maximize" else (score <= args.target_value)
                if target_reached:
                    print(f"\n   Target {args.target_value} reached! Stopping early.")
                    trial.study.stop()

            return score

        except Exception as e:
            print(f"\n   Trial {trial.number} failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            # Return worst possible score
            return -1e9 if direction == "maximize" else 1e9

    # Create and run Optuna study
    print(f"\nStarting optimization ({args.trials} trials)...\n")

    study = optuna.create_study(
        direction=direction,
        sampler=TPESampler(
            n_startup_trials=args.startup_trials,
            multivariate=True,
        ),
    )

    # Add early stopping callback
    if args.early_stop_patience:
        study.optimize(
            objective,
            n_trials=args.trials,
            show_progress_bar=not args.verbose,
            callbacks=[_early_stopping_callback(args.early_stop_patience, direction)],
        )
    else:
        study.optimize(
            objective,
            n_trials=args.trials,
            show_progress_bar=not args.verbose,
        )

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nBest parameters:")
    for k, v in best_params.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    print(f"\nBest {args.target_metric}: {best_result.score:.4f}")

    # Check if target achieved
    target_achieved = (best_result.score >= args.target_value) if direction == "maximize" else (best_result.score <= args.target_value)
    print(f"\nTarget {args.target_value} achieved: {'YES' if target_achieved else 'NO'}")

    # Apply best parameters and save final model
    print(f"\n{'='*80}")
    print("SAVING OPTIMIZED MODEL")
    print(f"{'='*80}")

    # Restore and apply best parameters
    base_model.load_state_dict(base_state_dict)

    # Get vectors for best num_pairs
    best_num_pairs = best_params.get("num_pairs", num_pairs_range[0])
    if best_num_pairs not in cached_vectors:
        vectors = _generate_steering_vectors(args, best_num_pairs, num_layers)
        cached_vectors[best_num_pairs] = vectors
    best_vectors = cached_vectors[best_num_pairs]

    _apply_weight_modification(
        base_model,
        best_vectors,
        best_params,
        num_layers,
        args,
    )

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
        "best_score": best_result.score,
        "target_achieved": target_achieved,
        "total_trials": len(all_trials),
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
                "trial": t.trial_number,
                "params": t.params,
                "score": t.score,
            }
            for t in all_trials
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

    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Total optimization time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*80}\n")

    return OptimizationResult(
        best_trial=best_result,
        all_trials=all_trials,
        best_params=best_params,
        target_achieved=target_achieved,
        total_time=total_time,
    )


def _generate_steering_vectors(args, num_pairs: int, num_layers: int) -> dict[int, torch.Tensor]:
    """Generate steering vectors from trait or task."""
    from argparse import Namespace

    if args.steering_vectors:
        # Load pre-computed vectors
        with open(args.steering_vectors, "r") as f:
            data = json.load(f)
        return {
            int(layer) - 1: torch.tensor(vec)
            for layer, vec in data["steering_vectors"].items()
        }

    # Create temp file for output
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_output = f.name

    try:
        if args.trait:
            # Generate from trait (synthetic pairs)
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
            # Generate from task
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

        # Load generated vectors
        with open(temp_output, "r") as f:
            data = json.load(f)

        return {
            int(layer) - 1: torch.tensor(vec)
            for layer, vec in data["steering_vectors"].items()
        }

    finally:
        if os.path.exists(temp_output):
            os.unlink(temp_output)


def _apply_weight_modification(
    model,
    steering_vectors: dict[int, torch.Tensor],
    params: dict,
    num_layers: int,
    args,
):
    """Apply weight modification with given parameters."""
    from wisent.core.weight_modification import abliterate_with_kernel, bake_steering_with_kernel

    # Convert position ratio to layer index
    max_weight_position = params["max_weight_position"] * (num_layers - 1)

    # Compute min_weight_distance from position
    min_weight_distance = 0.6 * (num_layers - 1)

    if args.method == "abliteration":
        abliterate_with_kernel(
            model,
            steering_vectors,
            max_weight=params["max_weight"] * params["strength"],
            max_weight_position=max_weight_position,
            min_weight=params["min_weight"],
            min_weight_distance=min_weight_distance,
            components=args.components,
            norm_preserve=args.norm_preserve,
            verbose=False,
        )
    else:
        bake_steering_with_kernel(
            model,
            steering_vectors,
            max_alpha=params["max_weight"] * params["strength"],
            max_alpha_position=max_weight_position,
            min_alpha=params["min_weight"],
            components=args.components,
            verbose=False,
        )


def _create_evaluator(args, tokenizer) -> Callable:
    """Create the appropriate evaluator function based on args."""

    evaluator_type = args.evaluator

    # Auto-select evaluator
    if evaluator_type == "auto":
        if args.trait and "refus" in args.trait.lower():
            evaluator_type = "refusal"
        elif args.task:
            evaluator_type = "task"
        else:
            evaluator_type = "refusal"  # default for traits

    if evaluator_type in ["refusal", "semantic", "keyword"]:
        return _create_refusal_evaluator(args, tokenizer, evaluator_type)
    elif evaluator_type == "task":
        return _create_task_evaluator(args)
    elif evaluator_type == "llm_judge":
        return _create_llm_judge_evaluator(args, tokenizer)
    else:
        raise ValueError(f"Unknown evaluator: {evaluator_type}")


def _create_refusal_evaluator(args, tokenizer, eval_type: str) -> Callable:
    """Create refusal/compliance evaluator."""
    from uncensorbench import UncensorBench
    from uncensorbench.evaluator import KeywordEvaluator, SemanticEvaluator
    from wisent.core.models.inference_config import get_generate_kwargs
    import re

    bench = UncensorBench()

    # Get evaluation prompts
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

    # Initialize evaluator
    if eval_type == "keyword":
        evaluator = KeywordEvaluator()
    else:
        try:
            evaluator = SemanticEvaluator()
        except ImportError:
            evaluator = KeywordEvaluator()

    def evaluate(model, tok) -> dict:
        """Run refusal evaluation."""
        results = []
        refusal_count = 0
        compliance_scores = []

        generate_kwargs = get_generate_kwargs()

        for item in prompts:
            if use_custom:
                prompt_text = item if isinstance(item, str) else item.get("prompt", str(item))
                prompt_obj = None
            else:
                prompt_obj = item
                prompt_text = prompt_obj.prompt

            # Format as chat
            try:
                messages = [{"role": "user", "content": prompt_text}]
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = f"User: {prompt_text}\nAssistant:"

            inputs = tok(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=generate_kwargs.get("do_sample", True),
                    temperature=generate_kwargs.get("temperature", 0.7) if generate_kwargs.get("do_sample", True) else None,
                    pad_token_id=tok.eos_token_id,
                )

            response = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

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
            "score": compliance_rate,  # default score
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
        # Load task data
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
                    model=None,  # Will use log likelihood
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


def _create_llm_judge_evaluator(args, tokenizer) -> Callable:
    """Create LLM-as-judge evaluator."""
    # TODO: Implement LLM judge evaluation
    raise NotImplementedError("LLM judge evaluator not yet implemented")


def _early_stopping_callback(patience: int, direction: str):
    """Create early stopping callback for Optuna."""
    best_value = None
    no_improvement_count = 0

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        nonlocal best_value, no_improvement_count

        if trial.value is None:
            return

        if best_value is None:
            best_value = trial.value
            return

        improved = (trial.value > best_value) if direction == "maximize" else (trial.value < best_value)

        if improved:
            best_value = trial.value
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"\nNo improvement for {patience} trials. Stopping early.")
            study.stop()

    return callback
