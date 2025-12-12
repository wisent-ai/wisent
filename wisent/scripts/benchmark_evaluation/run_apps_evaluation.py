"""
Run APPSEvaluator on APPS benchmark.

This script:
1. Loads codeparrot/apps dataset
2. Prompts an LLM to generate Python code in JSON format
3. Extracts code and runs against test cases
4. Computes accuracy and strict accuracy per difficulty level
5. Saves results to JSON file

Dataset: https://huggingface.co/datasets/codeparrot/apps
Paper: https://arxiv.org/abs/2105.09938

Metrics:
- Accuracy: % of problems with at least one test case passed
- Strict Accuracy: % of problems with ALL test cases passed
"""

import json
import re
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from wisent.core.models.wisent_model import WisentModel
from wisent.core.evaluators.benchmark_specific.apps_evaluator import APPSEvaluator


# Generation config
GENERATION_CONFIG = {
    "max_new_tokens": 2048,
    "temperature": 0.0,
    "do_sample": False,
}

# APPS difficulty levels
DIFFICULTIES = ["introductory", "interview", "competition"]


def evaluate_apps(
    model: WisentModel,
    evaluator: APPSEvaluator,
    split: str = "test",
    difficulty: str | None = None,
    limit: int | None = None,
) -> dict:
    """Evaluate model on APPS dataset.

    Args:
        model: WisentModel instance
        evaluator: APPSEvaluator instance
        split: Dataset split ('train' or 'test')
        difficulty: Optional difficulty filter ('introductory', 'interview', 'competition')
        limit: Optional limit on number of examples

    Returns:
        Dictionary with accuracy metrics and detailed results
    """
    # Load dataset with optional difficulty filter
    if difficulty:
        ds = load_dataset(
            "codeparrot/apps",
            split=split,
            difficulties=[difficulty],
            trust_remote_code=True,
        )
    else:
        ds = load_dataset("codeparrot/apps", split=split, trust_remote_code=True)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    results = []
    total_problems = 0
    problems_with_any_pass = 0  # At least one test passed
    problems_all_passed = 0  # All tests passed (strict accuracy)
    total_tests_passed = 0
    total_tests = 0

    # Track by difficulty
    difficulty_stats = {d: {"total": 0, "any_pass": 0, "all_pass": 0} for d in DIFFICULTIES}

    for example in tqdm(ds, desc=f"Evaluating {split}" + (f" ({difficulty})" if difficulty else "")):
        problem_id = example.get("problem_id", "")
        question = example.get("question", "")
        input_output_str = example.get("input_output", "")
        starter_code = example.get("starter_code", "")
        prob_difficulty = example.get("difficulty", "unknown")

        # Skip if no question or test cases
        if not question:
            continue

        # Parse input_output
        try:
            input_output = json.loads(input_output_str) if input_output_str else {}
        except json.JSONDecodeError:
            input_output = {}

        if not input_output.get("inputs") or not input_output.get("outputs"):
            continue

        total_problems += 1
        difficulty_stats[prob_difficulty]["total"] += 1

        # Generate prompt
        prompt = evaluator.get_prompt(question, starter_code)

        # Generate code
        responses = model.generate(
            inputs=prompt,
            **GENERATION_CONFIG,
            prompt_is_formatted=True,
        )

        raw_response = responses[0] if responses else ""

        # Extract code
        code = evaluator.extract_code_from_json(raw_response)

        if code is None:
            results.append({
                "problem_id": problem_id,
                "difficulty": prob_difficulty,
                "passed": 0,
                "total": len(input_output.get("inputs", [])),
                "all_passed": False,
                "error": "Failed to extract code from response",
                "raw_response": raw_response[:500],
            })
            total_tests += len(input_output.get("inputs", []))
            continue

        # Build test code and prepare solution
        test_code, _ = evaluator.build_test_code(input_output)
        code = evaluator.prepend_imports(code)

        if not test_code:
            num_tests = len(input_output.get("inputs", []))
            results.append({
                "problem_id": problem_id,
                "difficulty": prob_difficulty,
                "passed": 0,
                "total": num_tests,
                "all_passed": False,
                "error": "Failed to build test code",
            })
            total_tests += num_tests
            continue

        # Calculate timeout based on number of tests (1s per test, min 10s)
        num_tests_for_timeout = len(input_output.get("inputs", []))
        timeout = max(10, num_tests_for_timeout)

        # Run evaluation
        eval_result = evaluator.evaluate(code, expected=None, test_code=test_code, timeout=timeout)

        # Parse "PASSED:X/Y" from stdout
        stdout = eval_result.meta.get("stdout", "") if eval_result.meta else ""
        match = re.search(r'PASSED:(\d+)/(\d+)', stdout)
        if match:
            passed = int(match.group(1))
            num_tests = int(match.group(2))
        else:
            num_tests = len(input_output.get("inputs", []))
            passed = num_tests if eval_result.ground_truth == "TRUTHFUL" else 0

        all_passed = (passed == num_tests)

        total_tests += num_tests
        total_tests_passed += passed

        if passed > 0:
            problems_with_any_pass += 1
            difficulty_stats[prob_difficulty]["any_pass"] += 1

        if all_passed:
            problems_all_passed += 1
            difficulty_stats[prob_difficulty]["all_pass"] += 1

        results.append({
            "problem_id": problem_id,
            "difficulty": prob_difficulty,
            "passed": passed,
            "total": num_tests,
            "all_passed": all_passed,
            "question": question,
            "raw_response": raw_response,
            "code": code,
            "details": eval_result.details,
        })

    # Compute metrics
    accuracy = problems_with_any_pass / total_problems * 100 if total_problems > 0 else 0.0
    strict_accuracy = problems_all_passed / total_problems * 100 if total_problems > 0 else 0.0
    test_pass_rate = total_tests_passed / total_tests * 100 if total_tests > 0 else 0.0

    # Per-difficulty metrics
    difficulty_metrics = {}
    for d in DIFFICULTIES:
        stats = difficulty_stats[d]
        if stats["total"] > 0:
            difficulty_metrics[d] = {
                "total": stats["total"],
                "accuracy": stats["any_pass"] / stats["total"] * 100,
                "strict_accuracy": stats["all_pass"] / stats["total"] * 100,
            }

    return {
        "accuracy": accuracy,
        "strict_accuracy": strict_accuracy,
        "test_pass_rate": test_pass_rate,
        "total_problems": total_problems,
        "problems_with_any_pass": problems_with_any_pass,
        "problems_all_passed": problems_all_passed,
        "total_tests": total_tests,
        "total_tests_passed": total_tests_passed,
        "difficulty_metrics": difficulty_metrics,
        "results": results,
    }


def main(
    limit: int | None = None,
    split: str = "test",
    difficulty: str | None = None,
):
    """Run APPS evaluation and save results."""
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = WisentModel(model_name=model_name)

    evaluator = APPSEvaluator()

    print(f"\nEvaluating on APPS {split} split")
    if difficulty:
        print(f"Difficulty filter: {difficulty}")
    print("=" * 60)

    metrics = evaluate_apps(model, evaluator, split, difficulty, limit)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: codeparrot/apps")
    print(f"Split: {split}")
    if difficulty:
        print(f"Difficulty: {difficulty}")
    print(f"Problems evaluated: {metrics['total_problems']}")
    print()
    print(f"Accuracy (any test pass): {metrics['accuracy']:.2f}%")
    print(f"Strict Accuracy (all tests pass): {metrics['strict_accuracy']:.2f}%")
    print(f"Test Pass Rate: {metrics['test_pass_rate']:.2f}%")
    print()
    print(f"Problems with any pass: {metrics['problems_with_any_pass']}/{metrics['total_problems']}")
    print(f"Problems all passed: {metrics['problems_all_passed']}/{metrics['total_problems']}")
    print(f"Tests passed: {metrics['total_tests_passed']}/{metrics['total_tests']}")

    # Per-difficulty breakdown
    if metrics["difficulty_metrics"]:
        print("\nPer-Difficulty Metrics:")
        for d in DIFFICULTIES:
            if d in metrics["difficulty_metrics"]:
                dm = metrics["difficulty_metrics"][d]
                print(f"  {d:15s}: Acc={dm['accuracy']:5.2f}%, Strict={dm['strict_accuracy']:5.2f}% ({dm['total']} problems)")

    # Save results to JSON
    output_dir = Path(__file__).parent / "results_test_evaluator"
    output_dir.mkdir(exist_ok=True)
    suffix = f"_{difficulty}" if difficulty else ""
    output_file = output_dir / f"apps_evaluator_results_{split}{suffix}.json"

    output_data = {
        "model_name": model_name,
        "dataset": "codeparrot/apps",
        "split": split,
        "difficulty_filter": difficulty,
        **metrics,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main(limit=2, split="test", difficulty="introductory")
