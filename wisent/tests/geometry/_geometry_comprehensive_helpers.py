"""Extracted from test_geometry_comprehensive.py - results save and main block."""

import json
import argparse

from wisent.core.utils.config_tools.constants import SEPARATOR_WIDTH_REPORT, JSON_INDENT


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


def _build_preflight_thresholds_from_file(path: str):
    """Load PreflightThresholds from a JSON file, filtering to known fields."""
    from dataclasses import fields as dc_fields
    from wisent.core.control.steering_methods._helpers.preflight_helpers import PreflightThresholds
    with open(path, "r") as fh:
        data = json.load(fh)
    known = {f.name for f in dc_fields(PreflightThresholds)}
    return PreflightThresholds(**{k: v for k, v in data.items() if k in known})


def main():
    """Main entry point for comprehensive geometry analysis."""
    from wisent.tests.test_geometry_comprehensive import (
        run_comprehensive_geometry_analysis,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-pairs", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-thresholds-file", type=str, required=True,
                        help="JSON file with all PreflightThresholds fields")
    args = parser.parse_args()

    pf_thresholds = _build_preflight_thresholds_from_file(
        args.preflight_thresholds_file,
    )
    run_comprehensive_geometry_analysis(
        task=args.task,
        model=args.model,
        num_pairs=args.num_pairs,
        output_dir=args.output_dir,
        preflight_thresholds=pf_thresholds,
    )
