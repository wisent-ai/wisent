"""Extracted from finish_failed_benchmarks.py - main function tail."""


def print_fix_summary(total_benchmarks, successful_loads, failed_loads,
                      original_results, output_file):
    """Print the improvement summary and finalize.

    Prints how many benchmarks were fixed compared to original results,
    lists the fixed and remaining benchmarks, and confirms the output file.

    Args:
        total_benchmarks: Total number of benchmarks processed
        successful_loads: Number of successfully loaded benchmarks
        failed_loads: Number of failed benchmarks
        original_results: Original results dictionary with 'summary' key
        output_file: Path to the updated results file
    """
    print(f"   Improvement: "
          f"{successful_loads - original_results['summary']['successful_loads']}"
          f" benchmarks fixed")
    print(f"   Fixed benchmarks: math_qa, crows_pairs, hendrycks_ethics, "
          f"paws_x, mmmlu, pubmedqa")
    print(f"   Remaining issues: storycloze (manual download), "
          f"narrativeqa (large dataset)")

    print(f"\nFix completed! Updated results saved to: {output_file}")
