"""Test if a benchmark can create contrastive pairs and evaluation works."""

import json
import os
from pathlib import Path

from wisent.core.constants import (
    SPLIT_RATIO_HALF, DEFAULT_RANDOM_SEED, BENCHMARK_TEST_DATA_LIMIT,
    BENCHMARK_SPLIT_LIMIT,
)
from wisent.core.data_loaders.loaders.lm_eval.lm_loader import LMEvalDataLoader
from wisent.core.data_loaders.loaders.huggingface_loader import HuggingFaceDataLoader
from wisent.core.evaluators.rotator import EvaluatorRotator
from wisent.examples.scripts.test_one_benchmark_helpers import (
    MockModel,
    detect_loader_type,
)

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def test_benchmark(task_name: str, model_name: str = "distilgpt2", output_dir: str = ".", loader_type: str = "auto"):
    """Test if we can create contrastive pairs and evaluate them with mock model.

    This function:
    1. Creates contrastive pairs from the benchmark
    2. Finds the appropriate evaluator
    3. Evaluates pairs using a mock model (no real inference)
    4. Verifies positive=TRUTHFUL and negative=UNTRUTHFUL
    5. Saves pairs and evaluation results to JSON files

    Args:
        task_name: Name of the benchmark (e.g., "boolq", "gsm8k", "humaneval")
        model_name: Unused (kept for backward compatibility)
        output_dir: Directory to save results
        loader_type: Type of loader to use ("lm_eval", "huggingface", or "auto")

    Returns:
        True if all evaluations correct (positive=TRUTHFUL, negative=UNTRUTHFUL), False otherwise
    """
    try:
        print(f"\nTesting {task_name}...")
        output_path = Path(output_dir)
        # Create results directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Load data and create contrastive pairs
        print("  [1/3] Creating contrastive pairs...")


        # Auto-detect loader type if needed
        if loader_type == "auto":
            loader_type = detect_loader_type(task_name)


        # Select appropriate loader
        if loader_type == "huggingface":
            print(f"    Using HuggingFaceDataLoader")
            loader = HuggingFaceDataLoader()
        else:
            print(f"    Using LMEvalDataLoader")
            loader = LMEvalDataLoader()
            
        result = loader._load_one_task(
            task_name=task_name,
            split_ratio=SPLIT_RATIO_HALF,
            seed=DEFAULT_RANDOM_SEED,
            limit=BENCHMARK_TEST_DATA_LIMIT,
            training_limit=BENCHMARK_SPLIT_LIMIT,
            testing_limit=BENCHMARK_SPLIT_LIMIT
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
        evaluator_name = rotator._plugin.name
        print(f"    Using evaluator: {evaluator_name}")

        # Step 3: Monkey patch evaluator if it's log_likelihoods
        print("  [3/4] Setting up mock evaluation...")
        if evaluator_name == "log_likelihoods":
            # Monkey patch the log likelihood computation to return mock values
            # First choice always gets higher log prob (-0.5), rest get lower (-2.0)
            original_compute = rotator._plugin._compute_choice_log_likelihood
            choice_index = [0]  # Track which choice we're on

            def mock_compute_log_likelihood(model, question, choice):
                """Return mock log prob - first choice is higher."""
                idx = choice_index[0]
                choice_index[0] += 1
                if choice_index[0] > 1:  # Reset after both choices
                    choice_index[0] = 0
                return -0.5 if idx == 0 else -2.0

            rotator._plugin._compute_choice_log_likelihood = mock_compute_log_likelihood
            print(f"    Patched log_likelihoods evaluator with mock")
        elif evaluator_name == "perplexity":
            # Monkey patch perplexity computation
            def mock_compute_perplexity(model, text):
                """Return mock perplexity - lower for shorter text."""
                return len(text) * 0.1

            rotator._plugin._compute_perplexity = mock_compute_perplexity
            print(f"    Patched perplexity evaluator with mock")
        else:
            print(f"    No patching needed for {evaluator_name}")

        model = MockModel(model_name="mock")
        print(f"    Using MockModel (no real inference)")

        # Step 4: Evaluate with mock model
        print("  [4/4] Evaluating pairs...")
        results = []
        all_correct = True

        for i, pair in enumerate(test_pairs.pairs):
            # Verify required fields
            if not pair.prompt or not pair.positive_response.model_response or not pair.negative_response.model_response:
                print(f"    ✗ Pair {i+1}: Missing required fields")
                all_correct = False
                continue

            pair_results = {
                "pair_id": i,
                "prompt": pair.prompt + "...",
                "positive_response": pair.positive_response.model_response,
                "negative_response": pair.negative_response.model_response,
            }

            # Get test code from pair metadata if available (for coding tasks)
            test_code = None
            entry_point = None
            if hasattr(pair, 'metadata') and pair.metadata:
                test_code = pair.metadata.get('test_code')
                entry_point = pair.metadata.get('entry_point')

            # Evaluate with POSITIVE as expected (should return TRUTHFUL)
            # For exact_match evaluators, response should match the expected
            positive_response_value = pair.positive_response.model_response if evaluator_name == "exact_match" else ""
            eval_kwargs = {
                "response": positive_response_value,
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

            positive_eval = rotator._plugin.evaluate(**eval_kwargs)

            positive_correct = positive_eval.ground_truth == "TRUTHFUL"
            pair_results["positive_evaluation"] = {
                "ground_truth": positive_eval.ground_truth,
                "confidence": positive_eval.confidence,
                "expected": "TRUTHFUL",
                "correct": positive_correct,
                "details": positive_eval.details
            }

            # Evaluate with NEGATIVE as expected (should return UNTRUTHFUL)
            # For exact_match evaluators, provide positive_response (not negative) to test that it's marked UNTRUTHFUL
            negative_response_value = pair.positive_response.model_response if evaluator_name == "exact_match" else ""
            eval_kwargs_neg = {
                "response": negative_response_value,
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

            negative_eval = rotator._plugin.evaluate(**eval_kwargs_neg)

            negative_correct = negative_eval.ground_truth == "UNTRUTHFUL"
            pair_results["negative_evaluation"] = {
                "ground_truth": negative_eval.ground_truth,
                "confidence": negative_eval.confidence,
                "expected": "UNTRUTHFUL",
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
            "model_name": "mock",
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
    # Default to results directory in same folder as this script
    default_output = Path(__file__).parent / "results"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else str(default_output)
    success = test_benchmark(task, model, output_dir)
    sys.exit(0 if success else 1)
