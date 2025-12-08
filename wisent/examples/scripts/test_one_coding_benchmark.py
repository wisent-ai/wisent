"""Test coding benchmarks with Docker execution.

This script tests coding benchmarks (HumanEval, MBPP, etc.) by:
1. Creating contrastive pairs from the benchmark
2. Executing code in Docker sandbox
3. Verifying positive code passes tests (TRUTHFUL)
4. Verifying negative code fails tests (UNTRUTHFUL)

Usage:
    python test_one_coding_benchmark.py humaneval
    python test_one_coding_benchmark.py mbpp
    python test_one_coding_benchmark.py humaneval --limit 5
"""

import json
import os
import sys
from pathlib import Path

# Set environment variables
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = '1'
os.environ['HF_ALLOW_CODE_EVAL'] = '1'

from wisent.core.data_loaders.loaders.huggingface_loader import HuggingFaceDataLoader
from wisent.core.evaluators.benchmark_specific.coding.metrics.evaluator import CodingEvaluator, EvaluatorConfig


def test_coding_benchmark(
    task_name: str,
    output_dir: str = ".",
    limit: int = 10,
):
    """
    Test a coding benchmark using Docker sandbox execution.

    Args:
        task_name: Name of the benchmark (e.g., "humaneval", "mbpp")
        output_dir: Directory to save results
        limit: Maximum number of pairs to test

    Returns:
        True if all evaluations correct, False otherwise
    """
    try:
        print(f"\n{'='*60}")
        print(f"Testing coding benchmark: {task_name}")
        print(f"{'='*60}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Load data and create contrastive pairs
        print("\n[1/4] Creating contrastive pairs...")
        loader = HuggingFaceDataLoader()

        result = loader._load_one_task(
            task_name=task_name,
            split_ratio=0.5,
            seed=42,
            limit=limit * 3,  # Load more to account for filtering
            training_limit=limit,
            testing_limit=limit,
        )

        test_pairs = result['test_qa_pairs']
        print(f"  Created {len(test_pairs.pairs)} contrastive pairs")

        if len(test_pairs.pairs) == 0:
            print("  ERROR: No pairs created!")
            return False

        # Step 2: Initialize CodingEvaluator with Docker
        print("\n[2/4] Initializing CodingEvaluator with Docker sandbox...")

        cfg = EvaluatorConfig(
            image="coding/sandbox:polyglot-1.0",
            time_limit_s=10,
            cpu_limit_s=5,
            mem_limit_mb=512,
            pre_sanitize=True,
        )
        evaluator = CodingEvaluator(cfg=cfg)
        print(f"  Docker image: {cfg.image}")
        print(f"  Time limit: {cfg.time_limit_s}s, CPU limit: {cfg.cpu_limit_s}s")

        # Step 3: Save the pairs
        print("\n[3/4] Saving contrastive pairs...")
        pairs_data = []
        for i, pair in enumerate(test_pairs.pairs):
            pair_info = {
                "pair_id": i,
                "prompt": pair.prompt,
                "positive_response": pair.positive_response.model_response,
                "negative_response": pair.negative_response.model_response,
                "has_test_code": bool(pair.metadata and pair.metadata.get('test_code')),
                "entry_point": pair.metadata.get('entry_point') if pair.metadata else None,
            }
            pairs_data.append(pair_info)

        pairs_file = output_path / f"coding_{task_name}_pairs.json"
        with open(pairs_file, 'w') as f:
            json.dump(pairs_data, f, indent=2)
        print(f"  Saved pairs to: {pairs_file}")

        # Step 4: Evaluate with Docker execution
        print("\n[4/4] Evaluating pairs with Docker execution...")
        results = []
        all_correct = True
        passed_count = 0
        failed_count = 0

        for i, pair in enumerate(test_pairs.pairs):
            print(f"\n  Pair {i+1}/{len(test_pairs.pairs)}:")

            # Get test code from metadata
            test_code = None
            entry_point = None
            if pair.metadata:
                test_code = pair.metadata.get('test_code')
                entry_point = pair.metadata.get('entry_point')

            if not test_code:
                print(f"    SKIP: No test_code in metadata")
                continue

            print(f"    Entry point: {entry_point}")
            print(f"    Test code length: {len(test_code)} chars")

            pair_result = {
                "pair_id": i,
                "entry_point": entry_point,
            }

            # Evaluate POSITIVE response (should pass tests -> TRUTHFUL)
            print(f"    Evaluating positive response...")
            try:
                pos_eval = evaluator.evaluate(
                    response=pair.positive_response.model_response,
                    expected="",
                    test_code=test_code,
                    entry_point=entry_point,
                    task_name=task_name,
                    language="python",
                )
                pos_correct = pos_eval.ground_truth == "TRUTHFUL"
                pair_result["positive"] = {
                    "ground_truth": pos_eval.ground_truth,
                    "expected": "TRUTHFUL",
                    "correct": pos_correct,
                    "details": pos_eval.details if pos_eval.details else "",
                }
                status = "PASS" if pos_correct else "FAIL"
                print(f"      Result: {pos_eval.ground_truth} (expected TRUTHFUL) -> {status}")
                if not pos_correct:
                    print(f"      Details: {pos_eval.details if pos_eval.details else 'N/A'}")
            except Exception as e:
                print(f"      ERROR: {e}")
                pair_result["positive"] = {"error": str(e)}
                pos_correct = False

            # Evaluate NEGATIVE response (should fail tests -> UNTRUTHFUL)
            print(f"    Evaluating negative response...")
            try:
                neg_eval = evaluator.evaluate(
                    response=pair.negative_response.model_response,
                    expected="",
                    test_code=test_code,
                    entry_point=entry_point,
                    task_name=task_name,
                    language="python",
                )
                neg_correct = neg_eval.ground_truth == "UNTRUTHFUL"
                pair_result["negative"] = {
                    "ground_truth": neg_eval.ground_truth,
                    "expected": "UNTRUTHFUL",
                    "correct": neg_correct,
                    "details": neg_eval.details if neg_eval.details else "",
                }
                status = "PASS" if neg_correct else "FAIL"
                print(f"      Result: {neg_eval.ground_truth} (expected UNTRUTHFUL) -> {status}")
            except Exception as e:
                print(f"      ERROR: {e}")
                pair_result["negative"] = {"error": str(e)}
                neg_correct = False

            # Track overall success
            pair_correct = pos_correct and neg_correct
            pair_result["both_correct"] = pair_correct

            if pair_correct:
                passed_count += 1
                print(f"    PAIR RESULT: PASS")
            else:
                failed_count += 1
                all_correct = False
                print(f"    PAIR RESULT: FAIL")

            results.append(pair_result)

        # Save evaluation results
        eval_file = output_path / f"coding_{task_name}_evaluation.json"
        summary = {
            "task_name": task_name,
            "evaluator": "CodingEvaluator (Docker)",
            "num_pairs": len(test_pairs.pairs),
            "evaluated": len(results),
            "passed": passed_count,
            "failed": failed_count,
            "all_correct": all_correct,
            "pairs": results,
        }

        with open(eval_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Final summary
        print(f"\n{'='*60}")
        print(f"SUMMARY: {task_name}")
        print(f"{'='*60}")
        print(f"  Total pairs: {len(test_pairs.pairs)}")
        print(f"  Evaluated: {len(results)}")
        print(f"  Passed: {passed_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Success rate: {passed_count}/{len(results)} ({100*passed_count/max(1,len(results)):.1f}%)")
        print(f"  Results saved to: {eval_file}")

        if all_correct and len(results) > 0:
            print(f"\n  SUCCESS: All evaluations correct!")
        else:
            print(f"\n  FAILED: Some evaluations incorrect")

        return all_correct and len(results) > 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_docker_available():
    """Check if Docker is available and running."""
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            print("ERROR: Docker daemon is not running")
            print(f"  {result.stderr}")
            return False
        print("Docker is available and running")
        return True
    except FileNotFoundError:
        print("ERROR: Docker command not found. Please install Docker.")
        return False
    except subprocess.TimeoutExpired:
        print("ERROR: Docker command timed out")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test coding benchmarks with Docker execution")
    parser.add_argument("task", nargs="?", default="humaneval", help="Benchmark name (default: humaneval)")
    parser.add_argument("--limit", type=int, default=5, help="Number of pairs to test (default: 5)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    # Default output directory
    if args.output is None:
        args.output = str(Path(__file__).parent / "results")

    print("Checking Docker availability...")
    if not check_docker_available():
        sys.exit(1)

    print(f"\nRunning test for: {args.task}")
    print(f"Limit: {args.limit} pairs")
    print(f"Output: {args.output}")

    success = test_coding_benchmark(
        task_name=args.task,
        output_dir=args.output,
        limit=args.limit,
    )

    sys.exit(0 if success else 1)
