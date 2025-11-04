"""Test only the benchmarks that previously failed."""

import sys
from wisent.examples.scripts.test_one_benchmark import test_benchmark


# Only benchmarks that are still failing after all fixes
# Note: These fail because the 1B model gets answers wrong, not due to bugs
FAILED_BENCHMARKS = [
    "boolq",
    "coqa",
    "hellaswag",
    "math",
    "math500",
    "openbookqa",
    "polymath_en_medium",
    "polymath_zh_medium",
    "race",
    "swag",
    "truthfulqa_mc1",
    "truthfulqa_mc2",
    "winogrande",
]


def test_failed_benchmarks(model_name: str = "meta-llama/Llama-3.1-8B-Instruct", output_dir: str = "."):
    """Test only the failed benchmarks.

    Args:
        model_name: Model to use for testing
        output_dir: Directory to save results

    Returns:
        Dictionary with results for each benchmark
    """
    results = {
        "model": model_name,
        "total": len(FAILED_BENCHMARKS),
        "passed": 0,
        "failed": 0,
        "benchmarks": {}
    }

    print(f"\n{'='*70}")
    print(f"Testing {len(FAILED_BENCHMARKS)} previously failed benchmarks with {model_name}")
    print(f"{'='*70}\n")

    for i, benchmark in enumerate(FAILED_BENCHMARKS, 1):
        print(f"[{i}/{len(FAILED_BENCHMARKS)}] Testing {benchmark}...")

        try:
            success = test_benchmark(benchmark, model_name, output_dir)
            results["benchmarks"][benchmark] = {
                "status": "passed" if success else "failed",
                "success": success
            }

            if success:
                results["passed"] += 1
                print(f"   PASSED\n")
            else:
                results["failed"] += 1
                print(f"   FAILED\n")

        except Exception as e:
            results["benchmarks"][benchmark] = {
                "status": "error",
                "success": False,
                "error": str(e)
            }
            results["failed"] += 1
            print(f"   ERROR: {e}\n")

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {results['passed']/results['total']*100:.1f}%")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B-Instruct"
    # Default to results directory in scripts folder
    from pathlib import Path
    default_output = Path(__file__).parent / "results"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(default_output)

    results = test_failed_benchmarks(model, output_dir)

    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)
