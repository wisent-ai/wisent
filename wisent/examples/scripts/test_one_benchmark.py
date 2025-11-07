"""Test if a benchmark can create contrastive pairs and evaluation works."""

import json
from pathlib import Path

from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
from wisent.core.data_loaders.loaders.huggingface_loader import HuggingFaceDataLoader
from wisent.core.evaluators.rotator import EvaluatorRotator
from wisent.core.models.wisent_model import WisentModel


def test_benchmark(task_name: str, model_name: str = "distilgpt2", output_dir: str = ".", loader_type: str = "auto"):
    """Test if we can create contrastive pairs from a benchmark and if evaluation works.

    This function:
    1. Creates 2 contrastive pairs from the benchmark
    2. Evaluates the positive example (should return 1.0)
    3. Evaluates the negative example (should return 0.0)
    4. Saves pairs and evaluation results to JSON files

    Args:
        task_name: Name of the benchmark (e.g., "boolq", "gsm8k", "humaneval")
        model_name: Model to use for testing
        output_dir: Directory to save results
        loader_type: Type of loader to use ("lm_eval", "huggingface", or "auto")

    Returns:
        True if successful (positive=1.0, negative=0.0), False otherwise
    """
    try:
        print(f"\nTesting {task_name}...")
        output_path = Path(output_dir)
        # Create results directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Load data and create contrastive pairs
        print("  [1/4] Creating contrastive pairs...")

        # Auto-detect loader type if needed
        if loader_type == "auto":
            # Try HuggingFace first for known non-lm-eval tasks
            hf_tasks = [
                # Math benchmarks
                "math", "math_500", "aime", "hmmt", "polymath", "livemathbench",
                "minerva_math",
                # Coding benchmarks
                "humaneval", "humaneval_plus", "mbpp", "mbpp_plus",
                "instruct_humaneval", "apps", "conala", "concode",
                "ds", "ds1000", "ds_1000", "mercury", "recode",
                "multipl", "multiple_", "multipl_e",
                "codexglue", "livecodebench", "code_x_glue",
                # Reasoning benchmarks
                "super_gpqa", "supergpqa", "hle"
            ]
            if any(task_name.lower().startswith(t) for t in hf_tasks):
                loader_type = "huggingface"
            else:
                loader_type = "lm_eval"

        # Select appropriate loader
        if loader_type == "huggingface":
            print(f"    Using HuggingFaceDataLoader")
            loader = HuggingFaceDataLoader()
        else:
            print(f"    Using LMEvalDataLoader")
            loader = LMEvalDataLoader()

        result = loader._load_one_task(
            task_name=task_name,
            split_ratio=0.8,
            seed=42,
            limit=4,
            training_limit=2,
            testing_limit=2
        )

        test_pairs = result['test_qa_pairs']
        print(f"    Created {len(test_pairs.pairs)} contrastive pairs")

        # Save the pairs
        pairs_data = []
        for i, pair in enumerate(test_pairs.pairs):
            pairs_data.append({
                "pair_id": i,
                "prompt": pair.prompt,
                "positive_response": pair.positive_response.model_response,
                "negative_response": pair.negative_response.model_response,
            })

        pairs_file = output_path / f"test_{task_name}_pairs.json"
        with open(pairs_file, 'w') as f:
            json.dump(pairs_data, f, indent=2)
        print(f"    Saved pairs to: {pairs_file}")

        # Step 2: Find evaluator
        print("  [2/4] Finding evaluator...")
        EvaluatorRotator.discover_evaluators('wisent.core.evaluators.benchmark_specific')
        rotator = EvaluatorRotator(task_name=task_name)
        evaluator_name = rotator._evaluator.name
        print(f"    Using evaluator: {evaluator_name}")

        # Step 3: Load model
        print("  [3/4] Loading model...")
        model = WisentModel(model_name)
        print(f"    Loaded: {model_name}")

        # Step 4: Evaluate both positive and negative examples
        print("  [4/4] Evaluating examples...")

        results = []
        all_correct = True

        for i, pair in enumerate(test_pairs.pairs):
            pair_results = {
                "pair_id": i,
                "prompt": pair.prompt[:100] + "...",
                "positive_response": pair.positive_response.model_response,
                "negative_response": pair.negative_response.model_response,
            }

            # Get test code from pair metadata if available (for coding tasks)
            test_code = None
            entry_point = None
            if hasattr(pair, 'metadata') and pair.metadata:
                test_code = pair.metadata.get('test_code')
                entry_point = pair.metadata.get('entry_point')

            # Evaluate with POSITIVE as expected (should return TRUTHFUL = 1.0)
            # Evaluator compares log-probs of both choices and picks highest
            eval_kwargs = {
                "response": "",
                "expected": pair.positive_response.model_response,
                "model": model,
                "question": pair.prompt,
                "choices": [pair.positive_response.model_response, pair.negative_response.model_response],
                "task_name": task_name,
            }
            if test_code:
                eval_kwargs["test_code"] = test_code
            if entry_point:
                eval_kwargs["entry_point"] = entry_point

            positive_eval = rotator._evaluator.evaluate(**eval_kwargs)

            positive_correct = positive_eval.ground_truth == "TRUTHFUL"
            pair_results["positive_evaluation"] = {
                "ground_truth": positive_eval.ground_truth,
                "confidence": positive_eval.confidence,
                "expected": "TRUTHFUL",
                "correct": positive_correct,
                "details": positive_eval.details
            }

            # Evaluate with NEGATIVE as expected (should return UNTRUTHFUL = 0.0)
            # Because evaluator will pick positive (higher log-prob), not the negative we expect
            eval_kwargs_neg = {
                "response": "",
                "expected": pair.negative_response.model_response,
                "model": model,
                "question": pair.prompt,
                "choices": [pair.positive_response.model_response, pair.negative_response.model_response],
                "task_name": task_name
            }
            if test_code:
                eval_kwargs_neg["test_code"] = test_code
            if entry_point:
                eval_kwargs_neg["entry_point"] = entry_point

            negative_eval = rotator._evaluator.evaluate(**eval_kwargs_neg)

            negative_correct = negative_eval.ground_truth == "UNTRUTHFUL"  # Should be UNTRUTHFUL (predicted != expected)
            pair_results["negative_evaluation"] = {
                "ground_truth": negative_eval.ground_truth,
                "confidence": negative_eval.confidence,
                "expected": "UNTRUTHFUL",  # We expect it to fail (pick positive instead)
                "correct": negative_correct,
                "details": negative_eval.details
            }

            # Check if both evaluations are correct
            pair_correct = positive_correct and negative_correct
            pair_results["both_correct"] = pair_correct

            if not pair_correct:
                all_correct = False

            results.append(pair_results)

            print(f"    Pair {i+1}:")
            print(f"      Positive: {positive_eval.ground_truth} (expected TRUTHFUL) - {'✓' if positive_correct else '✗'}")
            print(f"      Negative: {negative_eval.ground_truth} (expected UNTRUTHFUL) - {'✓' if negative_correct else '✗'}")

        # Save evaluation results
        eval_file = output_path / f"test_{task_name}_evaluation.json"
        summary = {
            "task_name": task_name,
            "model_name": model_name,
            "evaluator_name": evaluator_name,
            "num_pairs": len(test_pairs.pairs),
            "all_correct": all_correct,
            "pairs": results
        }

        with open(eval_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"    Saved results to: {eval_file}")

        if all_correct:
            print(f"  ✓ SUCCESS: All evaluations correct!\n")
        else:
            print(f"  ✗ FAILED: Some evaluations incorrect\n")

        return all_correct

    except Exception as e:
        print(f"  ✗ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "boolq"
    model = sys.argv[2] if len(sys.argv) > 2 else "distilgpt2"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "."
    success = test_benchmark(task, model, output_dir)
    sys.exit(0 if success else 1)
