"""Test only the failing benchmarks from failing_benchmarks.json."""

import sys
import json
import signal
from contextlib import contextmanager
from wisent.examples.scripts.test_one_benchmark import test_benchmark


class TimeoutError(Exception):
    """Raised when a test times out."""
    pass


@contextmanager
def timeout(seconds):
    """Context manager for timing out operations."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Test timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def main():
    """Run tests for all failing benchmarks."""
    if len(sys.argv) < 2:
        print("Usage: python test_failing_benchmarks.py <model_name>")
        print("Example: python test_failing_benchmarks.py mock")
        sys.exit(1)

    model_name = sys.argv[1]
    output_dir = "results"

    # Load failing benchmarks
    failures_file = "results/failing_benchmarks.json"
    with open(failures_file) as f:
        data = json.load(f)

    benchmarks = [f["benchmark"] for f in data["failures"]]

    print("=" * 70)
    print(f"Testing {len(benchmarks)} failing benchmarks with {model_name}")
    print("=" * 70)
    print()

    results = {
        "model": model_name,
        "total": len(benchmarks),
        "passed": 0,
        "failed": 0,
        "timeout": 0,
        "benchmarks": {}
    }

    for i, benchmark in enumerate(benchmarks, 1):
        print(f"[{i}/{len(benchmarks)}] Testing {benchmark}...")

        try:
            # Use 1200 second timeout for each test
            with timeout(1200):
                success = test_benchmark(benchmark, model_name, output_dir)

            if success:
                results["benchmarks"][benchmark] = {
                    "status": "passed",
                    "success": True
                }
                results["passed"] += 1
                print(f"   PASSED\n")
            else:
                results["benchmarks"][benchmark] = {
                    "status": "failed",
                    "success": False
                }
                results["failed"] += 1
                print(f"   FAILED\n")

        except TimeoutError as e:
            results["benchmarks"][benchmark] = {
                "status": "timeout",
                "success": False,
                "error": str(e)
            }
            results["timeout"] += 1
            print(f"   TIMEOUT: {e}\n")

        except Exception as e:
            results["benchmarks"][benchmark] = {
                "status": "error",
                "success": False,
                "error": str(e)
            }
            results["failed"] += 1
            print(f"   ERROR: {e}\n")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total:   {results['total']}")
    print(f"Passed:  {results['passed']} ({results['passed']*100//results['total']}%)")
    print(f"Failed:  {results['failed']} ({results['failed']*100//results['total']}%)")
    print(f"Timeout: {results['timeout']}")

    # Save results
    results_file = f"{output_dir}/failing_benchmarks_test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
