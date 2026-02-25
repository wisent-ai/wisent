"""Extracted from test_geometry_comprehensive.py - results save and main block."""

import json
import argparse

from wisent.core.constants import PARSER_DEFAULT_NUM_PAIRS, SEPARATOR_WIDTH_REPORT, JSON_INDENT


def save_results_and_finish(all_results, results_file):
    """Save analysis results to JSON and print completion message.

    Writes the comprehensive geometry analysis results to a JSON file and
    prints the final analysis complete banner.

    Args:
        all_results: List of result objects with attributes: layer,
                     token_aggregation, prompt_strategy, best_structure, best_score
        results_file: Path to save the JSON results file
    """
    results_data = {
        "results": [
            {
                "layer": r.layer,
                "token_aggregation": r.token_aggregation,
                "prompt_strategy": r.prompt_strategy,
                "best_structure": r.best_structure,
                "best_score": r.best_score,
            }
            for r in all_results
        ],
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=JSON_INDENT)
    print(f"\nFull results saved to: {results_file}")

    print("\n" + "=" * SEPARATOR_WIDTH_REPORT)
    print("ANALYSIS COMPLETE")
    print("=" * SEPARATOR_WIDTH_REPORT)


def main():
    """Main entry point for comprehensive geometry analysis."""
    from wisent.tests.test_geometry_comprehensive import (
        run_comprehensive_geometry_analysis,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="truthfulqa_gen")
    parser.add_argument("--model",
                        default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--num-pairs", type=int, default=PARSER_DEFAULT_NUM_PAIRS)
    parser.add_argument("--output-dir", default="/home/ubuntu/output")
    args = parser.parse_args()

    run_comprehensive_geometry_analysis(
        task=args.task,
        model=args.model,
        num_pairs=args.num_pairs,
        output_dir=args.output_dir,
    )
