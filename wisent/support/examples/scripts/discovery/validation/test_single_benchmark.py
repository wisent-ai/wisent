"""Test a single benchmark: extraction + evaluation.

Usage:
    python test_single_benchmark.py boolq
    python test_single_benchmark.py truthfulqa_mc1
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from wisent.support.examples.scripts.discovery.validation.test_extractor import test_extractor
from wisent.support.examples.scripts.discovery.validation.test_evaluator import test_evaluator


def pairs_to_responses(pairs: list[dict]) -> list[dict]:
    """Convert extracted pairs to the format expected by evaluate-responses.

    Args:
        pairs: List of pair dicts from generate-pairs-from-task output.

    Returns:
        List of response dicts with keys: question, response, expected, choices.
    """
    responses = []
    for pair in pairs:
        positive = pair["positive_response"]["model_response"]
        negative = pair["negative_response"]["model_response"]
        responses.append({
            "prompt": pair["prompt"],
            "generated_response": positive,
            "positive_reference": positive,
            "correct_answers": [positive],
            "incorrect_answers": [negative],
        })
    return responses


def test_benchmark(task_name: str) -> dict:
    """Run extractor and evaluator tests for a single benchmark.

    Args:
        task_name: Benchmark task name.

    Returns:
        Dict with extraction and evaluator results.
    """
    result = {"task": task_name, "extraction": None, "evaluator": None}

    with tempfile.TemporaryDirectory() as tmpdir:
        pairs_file = os.path.join(tmpdir, "pairs.json")

        # --- Extraction ---
        print(f"\n[1/2] Extraction: {task_name}...")
        extraction_result = test_extractor(task_name, pairs_file)
        result["extraction"] = extraction_result
        print(f"  {extraction_result['status']} ({extraction_result.get('pair_count', 0)} pairs)")

        if extraction_result["status"] != "PASS":
            result["evaluator"] = {"status": "SKIP", "detail": "extraction failed"}
            return result

        # --- Evaluator ---
        print(f"\n[2/2] Evaluator: {task_name}...")
        with open(pairs_file) as f:
            data = json.load(f)
        pairs = data["pairs"] if isinstance(data, dict) else data
        responses = pairs_to_responses(pairs)

        evaluator_result = test_evaluator(task_name, responses)
        result["evaluator"] = evaluator_result
        print(f"  {evaluator_result['status']}")
        if evaluator_result.get("detail"):
            print(f"  {evaluator_result['detail']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Test a single benchmark.")
    parser.add_argument("task", help="Benchmark task name")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    result = test_benchmark(args.task)

    ext = result["extraction"]["status"]
    evl = result["evaluator"]["status"]
    print(f"\nResult: extraction={ext}  evaluator={evl}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved to {args.output}")

    sys.exit(0 if ext != "FAIL" and evl != "FAIL" else 1)


if __name__ == "__main__":
    main()
