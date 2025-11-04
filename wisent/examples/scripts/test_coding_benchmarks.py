"""Test all coding benchmarks to verify they use actual code execution."""

import sys
from pathlib import Path
from wisent.examples.scripts.test_one_benchmark import test_benchmark


# All coding benchmarks that should use code execution
CODING_BENCHMARKS = [
    # Python code generation with tests
    "humaneval",
    "humaneval_plus", 
    "mbpp",
    "mbpp_plus",
    "instruct_humaneval",
    "apps",
    "ds1000",
    "livecodebench",
    # Python snippets
    "conala",
    # Java code generation
    "concode",
    # Multi-language code translation
    "mercury",
    "recode",
    # Code-to-text (these might not have test code - will return UNKNOWN)
    "codexglue_code_to_text_python",
    "codexglue_code_to_text_go",
    "codexglue_code_to_text_ruby",
    "codexglue_code_to_text_java",
    "codexglue_code_to_text_javascript",
    "codexglue_code_to_text_php",
]


def test_coding_benchmarks(model_name: str = "meta-llama/Llama-3.1-8B-Instruct", output_dir: str = "."):
    """Test all coding benchmarks.
    
    Args:
        model_name: Model to use for testing
        output_dir: Directory to save results
        
    Returns:
        Dictionary with results for each benchmark
    """
    results = {
        "model": model_name,
        "total": len(CODING_BENCHMARKS),
        "passed": 0,
        "failed": 0,
        "unknown": 0,
        "benchmarks": {}
    }
    
    print(f"\n{'='*70}")
    print(f"Testing {len(CODING_BENCHMARKS)} coding benchmarks with {model_name}")
    print(f"{'='*70}\n")
    
    for i, benchmark in enumerate(CODING_BENCHMARKS, 1):
        print(f"[{i}/{len(CODING_BENCHMARKS)}] Testing {benchmark}...")
        
        try:
            success = test_benchmark(benchmark, model_name, output_dir)
            
            # Check if it returned UNKNOWN (no test code)
            import json
            eval_file = Path(output_dir) / f"test_{benchmark}_evaluation.json"
            if eval_file.exists():
                with open(eval_file, 'r') as f:
                    data = json.load(f)
                    if data.get("pairs"):
                        details = data["pairs"][0].get("positive_evaluation", {}).get("details", "")
                        has_test_code = "Code executed" in details or "execution" in details
                        is_unknown = data["pairs"][0].get("positive_evaluation", {}).get("ground_truth") == "UNKNOWN"
                        
                        if is_unknown:
                            results["benchmarks"][benchmark] = {
                                "status": "unknown",
                                "success": False,
                                "reason": "No test code available"
                            }
                            results["unknown"] += 1
                            print(f"   UNKNOWN (no test code)\n")
                        elif success and has_test_code:
                            results["benchmarks"][benchmark] = {
                                "status": "passed",
                                "success": True,
                                "method": "code_execution"
                            }
                            results["passed"] += 1
                            print(f"   PASSED (code execution)\n")
                        else:
                            results["benchmarks"][benchmark] = {
                                "status": "failed",
                                "success": False,
                                "reason": "Evaluation incorrect"
                            }
                            results["failed"] += 1
                            print(f"   FAILED\n")
                    else:
                        results["benchmarks"][benchmark] = {
                            "status": "failed",
                            "success": False,
                            "reason": "No pairs in evaluation"
                        }
                        results["failed"] += 1
                        print(f"   FAILED\n")
            else:
                results["benchmarks"][benchmark] = {
                    "status": "error",
                    "success": False,
                    "error": "No evaluation file"
                }
                results["failed"] += 1
                print(f"   ERROR: No evaluation file\n")
                
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
    print(f"Passed (code execution): {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Unknown (no test code): {results['unknown']}")
    if results['passed'] > 0:
        print(f"Success rate: {results['passed']/(results['total']-results['unknown'])*100:.1f}% (excluding UNKNOWN)")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B-Instruct"
    # Default to results directory in scripts folder
    default_output = Path(__file__).parent / "results"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(default_output)
    
    results = test_coding_benchmarks(model, output_dir)
    
    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)
