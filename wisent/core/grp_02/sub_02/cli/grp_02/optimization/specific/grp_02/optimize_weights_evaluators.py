"""Evaluator creation for optimize-weights."""
from typing import Any, Callable

from wisent.core.constants import DEFAULT_NUM_EVAL_PROMPTS, EVAL_MAX_NEW_TOKENS
from wisent.core.models.wisent_model import WisentModel
from wisent.core.evaluators.steering_evaluators import (
    SteeringEvaluatorFactory,
    EvaluatorConfig,
)
from wisent.core.cli.optimization.specific.optimize_weights_pooled import _create_pooled_evaluator


def _create_evaluator(args, model_name: str, wisent_model: WisentModel = None,
                      positive_examples: list[str] = None, negative_examples: list[str] = None) -> Callable[[Any, Any], dict[str, float]]:
    """Create the appropriate evaluator function based on --task argument.

    Uses shared steering evaluators from wisent.core.evaluators.steering_evaluators.

    Task types:
    - 'refusal' → RefusalEvaluator (compliance rate)
    - 'personalization' → PersonalizationEvaluator (requires --trait)
    - benchmark name → TaskEvaluator (e.g., 'arc_easy', 'gsm8k')
    - comma-separated benchmarks → PooledEvaluator (e.g., 'arc_easy,gsm8k')

    The evaluator receives (hf_model, tokenizer) and wraps them in WisentModel
    internally to use standard generation.

    Args:
        args: Command arguments (must have args.task)
        model_name: Model name/path
        wisent_model: Optional WisentModel instance for baseline generation (used by personalization)
        positive_examples: List of positive example responses from contrastive pairs
        negative_examples: List of negative example responses from contrastive pairs
    """
    from wisent.core.models import get_generate_kwargs

    task = args.task.lower() if args.task else ""
    
    # Determine evaluator type from task
    if task == "refusal":
        evaluator_type = "refusal"
    elif task == "personalization" or (not task and getattr(args, 'trait', None)):
        # Use personalization evaluator when --trait is provided (with or without --task personalization)
        evaluator_type = "personalization"
    elif task == "custom" or (not task and getattr(args, 'custom_evaluator', None)):
        # Use custom evaluator when --custom-evaluator is provided (with or without --task custom)
        if not getattr(args, 'custom_evaluator', None):
            raise ValueError("--custom-evaluator is required when --task custom")
        evaluator_type = "custom"
    elif "," in task:
        # Multiple benchmarks: use pooled evaluator
        evaluator_type = "pooled"
    elif task:
        # Single benchmark: use task evaluator (only if task is specified)
        evaluator_type = "task"
    else:
        raise ValueError("Either --task, --trait, or --custom-evaluator must be specified")

    # Pooled evaluator for multiple benchmarks
    if evaluator_type == "pooled":
        print(f"   [DEBUG] Creating pooled evaluator for tasks: {task}")
        print(f"   [DEBUG] args._pooled_eval_pairs count: {len(getattr(args, '_pooled_eval_pairs', []))}")
        return _create_pooled_evaluator(args)
    
    # Custom evaluator
    if evaluator_type == "custom":
        return _create_custom_evaluator(args, model_name)

    # Use shared steering evaluators for refusal, task, personalization
    eval_config = EvaluatorConfig(
        evaluator_type=evaluator_type,
        trait=getattr(args, 'trait', None),
        task=task if evaluator_type == "task" else None,
        eval_prompts_path=getattr(args, 'eval_prompts', None),
        eval_topics=getattr(args, 'eval_topics', None),
        num_eval_prompts=getattr(args, 'num_eval_prompts', DEFAULT_NUM_EVAL_PROMPTS),
    )

    # Create shared evaluator
    shared_evaluator = SteeringEvaluatorFactory.create(
        eval_config, model_name, wisent_model, positive_examples, negative_examples
    )
    
    # Pre-generate baseline responses BEFORE optimization modifies the model weights
    # This is critical because the same wisent_model/hf_model is used for both baseline and steered generation
    if hasattr(shared_evaluator, 'generate_baseline_responses'):
        print("   Pre-generating baseline responses with unmodified model...")
        baseline = shared_evaluator.generate_baseline_responses()
        print(f"   Generated {len(baseline)} baseline responses")

    def evaluate(hf_model, tokenizer) -> dict:
        """Run evaluation using shared steering evaluator."""
        # Wrap HF model in WisentModel for standard generation
        temp_wisent_model = WisentModel(model_name, hf_model=hf_model)
        
        # Get prompts from evaluator
        prompts = shared_evaluator.get_prompts()
        
        # Generate responses
        responses = []
        for prompt_text in prompts:
            messages = [{"role": "user", "content": prompt_text}]
            result = temp_wisent_model.generate(
                [messages],
                **get_generate_kwargs(max_new_tokens=EVAL_MAX_NEW_TOKENS),
            )
            responses.append(result[0] if result else "")
        
        # Evaluate using shared evaluator
        return shared_evaluator.evaluate_responses(responses)

    return evaluate


def _create_custom_evaluator(args, model_name: str) -> Callable:
    """Create custom evaluator for --task custom.
    
    Uses the custom evaluator framework from wisent.core.evaluators.custom.
    
    Args:
        args: Command arguments (must have args.custom_evaluator)
        model_name: Model name/path
    """
    import json
    from wisent.core.evaluators.custom import create_custom_evaluator
    from wisent.core.models import get_generate_kwargs
    
    # Parse custom evaluator kwargs
    custom_kwargs = {}
    if getattr(args, 'custom_evaluator_kwargs', None):
        custom_kwargs = json.loads(args.custom_evaluator_kwargs)
    
    # Create the custom evaluator
    custom_eval = create_custom_evaluator(args.custom_evaluator, **custom_kwargs)
    
    # Get test prompts - for custom, we need either trait-based prompts or user-provided prompts
    eval_prompts = []
    if getattr(args, 'eval_prompts', None):
        # Load from file
        with open(args.eval_prompts) as f:
            eval_prompts = [line.strip() for line in f if line.strip()]
    elif getattr(args, 'trait', None):
        # Use default prompts from PersonalizationEvaluator
        from wisent.core.evaluators.steering_evaluators import PersonalizationEvaluator
        num_prompts = getattr(args, 'num_eval_prompts', DEFAULT_NUM_EVAL_PROMPTS)
        eval_prompts = PersonalizationEvaluator.DEFAULT_PROMPTS[:num_prompts]
    else:
        # Default generic prompts
        eval_prompts = [
            "Tell me about yourself.",
            "What do you think about artificial intelligence?",
            "How would you solve world hunger?",
            "Explain quantum computing in simple terms.",
            "What's the meaning of life?",
        ]
    
    def evaluate(hf_model, tokenizer) -> dict:
        """Run evaluation using custom evaluator."""
        # Create WisentModel wrapper without reloading tokenizer from HuggingFace
        # Use object.__new__ to avoid __init__ which tries to load from HF
        temp_wisent_model = object.__new__(WisentModel)
        temp_wisent_model.hf_model = hf_model
        temp_wisent_model.tokenizer = tokenizer
        temp_wisent_model.model_name = model_name
        # Set internal attributes
        if hasattr(hf_model, 'model') and hasattr(hf_model.model, 'layers'):
            temp_wisent_model._layers = hf_model.model.layers
        elif hasattr(hf_model, 'transformer') and hasattr(hf_model.transformer, 'h'):
            temp_wisent_model._layers = hf_model.transformer.h
        else:
            temp_wisent_model._layers = []
        temp_wisent_model._hidden_size = hf_model.config.hidden_size
        temp_wisent_model.device = next(hf_model.parameters()).device
        
        # Generate responses
        scores = []
        for prompt_text in eval_prompts:
            messages = [{"role": "user", "content": prompt_text}]
            result = temp_wisent_model.generate(
                [messages],
                **get_generate_kwargs(max_new_tokens=EVAL_MAX_NEW_TOKENS),
            )
            response = result[0] if result else ""
            
            # Score using custom evaluator - pass prompt for coherence checking
            score = custom_eval(response, prompt=prompt_text)
            if isinstance(score, dict):
                # Take the primary score (first value or 'score' key)
                score = score.get('score', list(score.values())[0])
            scores.append(float(score))
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "score": avg_score,
            "custom_score": avg_score,
            "num_evaluated": len(scores),
        }
    
    return evaluate

