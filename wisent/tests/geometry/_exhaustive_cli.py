"""CLI entry point for exhaustive geometry analysis."""

import json
import os
import sys
from datetime import datetime

from wisent.core.constants import GEO_MAX_LAYER_COMBO_SIZE, PAIR_GENERATORS_DEFAULT_N, TEST_MAX_COMBO_SIZE, DISPLAY_TOP_N_SMALL, JSON_INDENT
from wisent.tests.test_geometry_exhaustive import (
    run_exhaustive_layer_analysis,
    run_limited_layer_analysis,
    run_contiguous_layer_analysis,
    run_smart_layer_analysis,
    TOKEN_AGGREGATIONS,
    PROMPT_STRATEGIES,
)


def run_comprehensive_sweep(
    task: str = "truthfulqa_gen",
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    num_pairs: int = PAIR_GENERATORS_DEFAULT_N,
    max_combo_size: int = TEST_MAX_COMBO_SIZE,
    output_dir: str = "/home/ubuntu/output",
):
    """Run sweep across all token agg and prompt strategies."""
    sys.stdout.reconfigure(line_buffering=True)
    total_configs = len(TOKEN_AGGREGATIONS) * len(PROMPT_STRATEGIES)
    print(f"Total configurations: {total_configs}")
    all_results = []
    config_idx = 0
    for token_agg in TOKEN_AGGREGATIONS:
        for prompt_strat in PROMPT_STRATEGIES:
            config_idx += 1
            print(
                f"\nCONFIG {config_idx}/{total_configs}:"
                f" {token_agg} + {prompt_strat}"
            )
            try:
                result = run_smart_layer_analysis(
                    task=task, model=model,
                    num_pairs=num_pairs,
                    max_combo_size=max_combo_size,
                    token_aggregation=token_agg,
                    prompt_strategy=prompt_strat,
                    output_dir=output_dir,
                )
                if result:
                    all_results.append({
                        "token_aggregation": token_agg,
                        "prompt_strategy": prompt_strat,
                        "best_combination": list(
                            result.best_combination
                        ),
                        "best_score": result.best_score,
                        "best_structure": result.best_structure.value,
                        "single_layer_best": result.single_layer_best,
                        "single_layer_best_score": (
                            result.single_layer_best_score
                        ),
                        "improvement_over_single": (
                            result.improvement_over_single
                        ),
                    })
            except Exception as e:
                print(f"ERROR: {e}")
                all_results.append({
                    "token_aggregation": token_agg,
                    "prompt_strategy": prompt_strat,
                    "error": str(e),
                })

    successful = [r for r in all_results if "best_score" in r]
    successful.sort(key=lambda x: x["best_score"], reverse=True)
    print(f"\nCompleted {len(successful)}/{total_configs}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(
        output_dir,
        f"geometry_sweep_summary_{task}_{timestamp}.json",
    )
    summary = {
        "task": task, "model": model,
        "num_pairs": num_pairs,
        "max_combo_size": max_combo_size,
        "total_configurations": total_configs,
        "successful_configurations": len(successful),
        "all_results": all_results,
        "top_10": successful[:DISPLAY_TOP_N_SMALL],
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=JSON_INDENT)
    print(f"Sweep summary saved to: {summary_file}")
    return summary


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="truthfulqa_gen")
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.2-1B-Instruct"
    )
    parser.add_argument("--num-pairs", type=int, default=PAIR_GENERATORS_DEFAULT_N)
    parser.add_argument(
        "--max-layers", type=int, default=None,
        help="DEBUG ONLY - DO NOT USE IN PRODUCTION.",
    )
    parser.add_argument("--output-dir", default="/home/ubuntu/output")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--smart", action="store_true", default=True)
    parser.add_argument("--limited", action="store_true")
    parser.add_argument("--contiguous", action="store_true")
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--max-combo-size", type=int, default=GEO_MAX_LAYER_COMBO_SIZE)
    parser.add_argument(
        "--token-aggregation", default="final",
        choices=TOKEN_AGGREGATIONS,
    )
    parser.add_argument(
        "--prompt-strategy", default="chat_template",
        choices=PROMPT_STRATEGIES,
    )
    args = parser.parse_args()
    if args.max_layers is not None:
        print("!" * 80)
        print("WARNING: --max-layers is set! DEBUG ONLY.")
        print("!" * 80)
    if args.sweep:
        run_comprehensive_sweep(
            task=args.task, model=args.model,
            num_pairs=args.num_pairs,
            max_combo_size=args.max_combo_size,
            output_dir=args.output_dir,
        )
    elif args.exhaustive:
        run_exhaustive_layer_analysis(
            task=args.task, model=args.model,
            num_pairs=args.num_pairs,
            max_layers=args.max_layers,
            output_dir=args.output_dir,
        )
    elif args.contiguous:
        run_contiguous_layer_analysis(
            task=args.task, model=args.model,
            num_pairs=args.num_pairs,
            output_dir=args.output_dir,
        )
    elif args.limited:
        run_limited_layer_analysis(
            task=args.task, model=args.model,
            num_pairs=args.num_pairs,
            max_combo_size=args.max_combo_size,
            output_dir=args.output_dir,
        )
    else:
        run_smart_layer_analysis(
            task=args.task, model=args.model,
            num_pairs=args.num_pairs,
            max_combo_size=args.max_combo_size,
            token_aggregation=args.token_aggregation,
            prompt_strategy=args.prompt_strategy,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
