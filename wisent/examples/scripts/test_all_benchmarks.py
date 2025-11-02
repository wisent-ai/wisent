"""Test all benchmarks to verify extractor and evaluator work."""

import sys
from wisent.examples.scripts.test_one_benchmark import test_benchmark


BENCHMARKS = [
    # Log-likelihood tasks
    "boolq", "winogrande", "piqa", "copa", "cb",
    "hellaswag", "swag", "openbookqa", "race",
    "arc_easy", "arc_challenge", "mmlu", "gpqa", "super_gpqa", "hle",
    "truthfulqa_mc1", "truthfulqa_mc2",
    # Generation/Math tasks
    "gsm8k", "asdiv", "arithmetic", "math", "math_500",
    "aime", "hmmt", "polymath", "livemathbench",
    "drop", "triviaqa", "record", "squadv2",
    "webqs", "nq_open", "coqa",
    # Perplexity tasks
    "wikitext", "wikitext103", "ptb", "penn_treebank",
    "lambada_openai", "lambada_standard",
    # Coding tasks
    "humaneval", "humaneval_plus", "instruct_humaneval",
    "mbpp", "mbpp_plus", "apps", "conala", "concode",
    "ds_1000", "mercury", "recode", "multipl_e",
    "codexglue", "livecodebench"
]


def test_all_benchmarks(model_name: str = "meta-llama/Llama-3.1-8B-Instruct", output_dir: str = "."):
    """Test all benchmarks.

    Args:
        model_name: Model to use for testing
        output_dir: Directory to save results

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
    print(f"{'='*70}\n")

    for i, benchmark in enumerate(BENCHMARKS, 1):
        print(f"[{i}/{len(BENCHMARKS)}] Testing {benchmark}...")

        try:
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
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    results = test_all_benchmarks(model, output_dir)

    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)
