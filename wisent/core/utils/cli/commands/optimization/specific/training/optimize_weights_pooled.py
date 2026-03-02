"""Pooled evaluator for multi-benchmark optimization."""
from typing import Callable

from wisent.core.primitives.models.wisent_model import WisentModel


def _create_pooled_evaluator(args) -> Callable:
    """Create pooled evaluator for multi-benchmark mode (--task bench1,bench2,...).
    
    Evaluates on ALL benchmarks from training, using each benchmark's
    native evaluator (routing by source_benchmark metadata).
    """
    from wisent.core.reading.evaluators.rotator import EvaluatorRotator
    from wisent.core.primitives.models import get_generate_kwargs

    # Get eval pairs stored during vector generation
    eval_pairs = getattr(args, '_pooled_eval_pairs', [])
    benchmarks_used = getattr(args, '_benchmarks_used', [])

    if not eval_pairs:
        raise ValueError("No eval pairs found for pooled evaluation. "
                        "Make sure train_unified_goodness saved eval_pairs to checkpoint.")

    print(f"   Pooled evaluator: {len(eval_pairs)} eval pairs across {len(benchmarks_used)} benchmarks")

    # Discover evaluators once
    EvaluatorRotator.discover_evaluators('wisent.core.reading.evaluators.oracles')
    EvaluatorRotator.discover_evaluators('wisent.core.reading.evaluators.benchmark_specific')

    # Group pairs by benchmark
    pairs_by_benchmark = {}
    for pair in eval_pairs:
        bench = pair.get('source_benchmark', 'unknown')
        if bench not in pairs_by_benchmark:
            pairs_by_benchmark[bench] = []
        pairs_by_benchmark[bench].append(pair)

    print(f"   Benchmarks in eval set: {list(pairs_by_benchmark.keys())}")

    def evaluate(hf_model, tokenizer) -> dict[str, float]:
        """Evaluate modified model on pooled benchmark data."""
        from wisent.core.primitives.models.wisent_model import WisentModel

        # Create WisentModel wrapper - use object.__new__ to avoid __init__ loading
        wisent_model = object.__new__(WisentModel)
        wisent_model.hf_model = hf_model
        wisent_model.tokenizer = tokenizer
        wisent_model.model_name = "modified_model"
        # Set internal attributes that the properties depend on
        if hasattr(hf_model, 'model') and hasattr(hf_model.model, 'layers'):
            wisent_model._layers = hf_model.model.layers
        elif hasattr(hf_model, 'transformer') and hasattr(hf_model.transformer, 'h'):
            wisent_model._layers = hf_model.transformer.h
        else:
            wisent_model._layers = []
        wisent_model._hidden_size = hf_model.config.hidden_size
        wisent_model.device = next(hf_model.parameters()).device

        total_correct = 0
        total_samples = 0
        benchmark_scores = {}

        for bench_name, bench_pairs in pairs_by_benchmark.items():
            # Get evaluator for this benchmark
            evaluator = EvaluatorRotator(
                evaluator=None,
                task_name=bench_name,
                autoload=False
            )

            correct = 0
            evaluated = 0

            for pair in bench_pairs:
                try:
                    question = pair['prompt']
                    expected = pair['positive_response']
                    metadata = pair.get('metadata', {}) or {}
                    test_code = metadata.get('test_code', '')
                    entry_point = metadata.get('entry_point', '')

                    messages = [{"role": "user", "content": question}]
                    response = wisent_model.generate(
                        [messages],
                        **get_generate_kwargs(),
                    )[0]

                    # Evaluate
                    eval_result = evaluator.evaluate(
                        response=response,
                        expected=expected,
                        model=wisent_model,
                        question=question,
                        task_name=bench_name,
                        test_code=test_code,
                        entry_point=entry_point,
                    )

                    if eval_result.ground_truth == "TRUTHFUL":
                        correct += 1
                    evaluated += 1

                except Exception as e:
                    # Count failed evals but continue
                    evaluated += 1

            if evaluated > 0:
                benchmark_scores[bench_name] = correct / evaluated
                total_correct += correct
                total_samples += evaluated

        # Compute aggregate accuracy
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": total_correct,
            "total": total_samples,
            "score": accuracy,
            "benchmark_scores": benchmark_scores,
        }

    return evaluate

