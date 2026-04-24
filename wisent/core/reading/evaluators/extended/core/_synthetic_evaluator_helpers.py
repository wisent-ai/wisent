"""Synthetic evaluator helper functions.

Extracted from synthetic_evaluator.py to keep file under 300 lines.
Contains evaluate_model implementation and create_synthetic_evaluator factory.
"""

from typing import Dict


def evaluate_model_impl(evaluator, model=None, num_samples: int = None) -> Dict:
    """Evaluate a model on the test prompts.

    Args:
        evaluator: SyntheticEvaluator instance
        model: Model to evaluate (uses evaluator.model if None)
        num_samples: Number of prompts to test (uses all if None)

    Returns:
        Dict with scores, mean, std, and individual results
    """
    eval_model = model or evaluator.model
    if eval_model is None:
        raise ValueError("No model available for evaluation")

    prompts = evaluator._test_prompts
    if num_samples:
        prompts = prompts[:num_samples]

    results = []
    scores = []

    for prompt in prompts:
        # Generate response
        messages = [{"role": "user", "content": prompt}]
        response = eval_model.generate(messages)

        # Evaluate
        score = evaluator.evaluate_response(response, prompt=prompt)
        scores.append(score)

        results.append({
            "prompt": prompt,
            "response": response,
            "score": score,
        })

    import statistics
    return {
        "mean_score": statistics.mean(scores) if scores else 0.0,
        "std_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "num_samples": len(scores),
        "results": results,
    }


def create_synthetic_evaluator(
    trait_description: str,
    model=None,
    test_prompts_file: str = None,
    generate_prompts: bool = False,
    verbose: bool = False,
):
    """Factory function to create a SyntheticEvaluator.

    Args:
        trait_description: Free-form description of the trait to evaluate
        model: WisentModel instance to use for generation and judging
        test_prompts_file: Optional file with test prompts
        generate_prompts: If True, generate prompts relevant to trait
        verbose: Enable verbose output

    Returns:
        Configured SyntheticEvaluator instance
    """
    from wisent.core.reading.evaluators.synthetic_evaluator import SyntheticEvaluator
    return SyntheticEvaluator(
        trait_description=trait_description,
        model=model,
        test_prompts_file=test_prompts_file,
        generate_prompts=generate_prompts,
        verbose=verbose,
    )
