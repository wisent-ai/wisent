"""Test all benchmarks to verify extractor and evaluator work."""

import json
import os
import sys
import signal
from contextlib import contextmanager
from pathlib import Path
from wisent.examples.scripts.test_one_benchmark import test_benchmark

# Set environment variable to trust remote code for datasets like meddialog
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = '1'
# Set environment variable to allow code eval for coding tasks like humaneval, instructhumaneval
os.environ['HF_ALLOW_CODE_EVAL'] = '1'


class TimeoutError(Exception):
    """Raised when a test times out."""
    pass


@contextmanager
def timeout(seconds):
    """Context manager for timing out operations."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Test timed out after {seconds} seconds")

    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def load_benchmarks():
    """Load benchmarks from central registry."""
    from wisent.core.benchmark_registry import get_all_benchmarks, get_broken_tasks
    
    broken_tasks = get_broken_tasks()
    if broken_tasks:
        print(f"Skipping {len(broken_tasks)} broken benchmarks: {', '.join(broken_tasks)}")

    return get_all_benchmarks()


BENCHMARKS = load_benchmarks()


def test_all_benchmarks(model_name: str = "meta-llama/Llama-3.1-8B-Instruct", output_dir: str = ".", start_index: int = 0):
    """Test all benchmarks.

    Args:
        model_name: Model to use for testing
        output_dir: Directory to save results
        start_index: Index to start testing from (0-based)

    Returns:
        Dictionary with results for each benchmark
    """
    results = {
        "model": model_name,
        "total": len(BENCHMARKS),
        "passed": 0,
        "failed": 0,
        "benchmarks": {}
    }

    print(f"\n{'='*70}")
    print(f"Testing {len(BENCHMARKS)} benchmarks with {model_name}")
    if start_index > 0:
        print(f"Starting from benchmark {start_index + 1} ({BENCHMARKS[start_index]})")
    print(f"{'='*70}\n")

    for i, benchmark in enumerate(BENCHMARKS, 1):
        if i - 1 < start_index:
            continue
        print(f"[{i}/{len(BENCHMARKS)}] Testing {benchmark}...")

        try:
            with timeout(1200):
                success = test_benchmark(benchmark, model_name, output_dir)
            results["benchmarks"][benchmark] = {
                "status": "passed" if success else "failed",
                "success": success
            }

            if success:
                results["passed"] += 1
                print(f"   PASSED\n")
            else:
                results["failed"] += 1
                print(f"   FAILED\n")

        except TimeoutError as e:
            results["benchmarks"][benchmark] = {
                "status": "timeout",
                "success": False,
                "error": str(e)
            }
            results["failed"] += 1
            print(f"   TIMEOUT: {e}\n")

        except Exception as e:
            results["benchmarks"][benchmark] = {
                "status": "error",
                "success": False,
                "error": str(e)
            }
            results["failed"] += 1
            print(f"   ERROR: {e}\n")

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
    default_output = Path(__file__).parent / "results"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(default_output)
    start_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    results = test_all_benchmarks(model, output_dir, start_index)

    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)
