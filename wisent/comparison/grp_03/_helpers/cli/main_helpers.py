"""Extracted helpers: run_comparison and main CLI entry point for comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_comparison(
    model_name: str,
    tasks: list[str],
    methods: list[str] = None,
    num_pairs: int = 50,
    steering_scales: list[float] = None,
    device: str = "cuda:0",
    batch_size: int | str = 1,
    max_batch_size: int = 8,
    eval_limit: int | None = None,
    output_dir: str = "comparison_results",
    train_ratio: float = 0.8,
    caa_layers: str = "12",
    sae_layers: str = "12",
    extraction_strategies: list[str] = None,
    bos_features_source: str = "detected",
    run_single_task_fn=None,
) -> list[dict]:
    """
    Run full comparison for multiple tasks, methods, scales, and extraction strategies.
    """
    if methods is None:
        methods = ["caa"]
    if steering_scales is None:
        steering_scales = [1.0]
    if extraction_strategies is None:
        extraction_strategies = ["mc_balanced"]

    output_dir = Path(output_dir)
    # Add model name to path (sanitize "/" -> "_")
    model_dir_name = model_name.replace("/", "_")
    output_dir = output_dir / model_dir_name
    vectors_dir = output_dir / "steering_vectors"
    results_dir = output_dir / "results"

    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for task in tasks:
        print(f"\n{'#'*60}")
        print(f"# TASK: {task}")
        print(f"{'#'*60}")

        task_results = run_single_task_fn(
            model_name=model_name,
            task=task,
            methods=methods,
            num_pairs=num_pairs,
            steering_scales=steering_scales,
            device=device,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            eval_limit=eval_limit,
            vectors_dir=vectors_dir,
            train_ratio=train_ratio,
            caa_layers=caa_layers,
            sae_layers=sae_layers,
            extraction_strategies=extraction_strategies,
            bos_features_source=bos_features_source,
        )
        all_results.extend(task_results)

        # Save results for this task (includes all strategies)
        task_results_file = results_dir / f"{task}_results.json"
        with open(task_results_file, "w") as f:
            json.dump(task_results, f, indent=2)
        print(f"Results for {task} saved to: {task_results_file}")

    # Print final summary table
    _print_summary(all_results, model_name, num_pairs, caa_layers,
                   sae_layers, extraction_strategies, vectors_dir, results_dir)

    return all_results


def _print_summary(
    all_results, model_name, num_pairs, caa_layers,
    sae_layers, extraction_strategies, vectors_dir, results_dir,
):
    """Print the final comparison summary table."""
    print(f"\n{'='*150}")
    print(f"FINAL COMPARISON RESULTS")
    print(f"{'='*150}")
    print(f"Model: {model_name}")
    print(f"Num pairs: {num_pairs}")
    print(f"CAA Layers: {caa_layers}")
    print(f"SAE/FGAA Layers: {sae_layers}")
    print(f"Strategies: {', '.join(extraction_strategies)}")
    print(f"{'='*150}")
    header = (f"{'Strategy':<16} {'Task':<10} {'Method':<8} {'Scale':<6} "
              f"{'Base(E)':<8} {'Base(L)':<8} {'Steer(E)':<9} {'Steer(L)':<9} "
              f"{'Native':<8} {'Diff(E)':<8} {'Diff(L)':<8} {'Diff(N)':<8}")
    print(header)
    print(f"{'-'*150}")

    for r in all_results:
        strat = r.get('extraction_strategy', 'N/A')
        print(f"{strat:<16} {r['task']:<10} {r['method']:<8} "
              f"{r['steering_scale']:<6.1f} "
              f"{r['base_accuracy_lm_eval']:<8.4f} "
              f"{r['base_accuracy_ll']:<8.4f} "
              f"{r['steered_accuracy_lm_eval']:<9.4f} "
              f"{r['steered_accuracy_ll']:<9.4f} "
              f"{r['steered_accuracy_lm_eval_native']:<8.4f} "
              f"{r['difference_lm_eval']:+<8.4f} "
              f"{r['difference_ll']:+<8.4f} "
              f"{r['difference_lm_eval_native']:+<8.4f}")

    print(f"{'='*150}")
    print(f"\nSteering vectors saved to: {vectors_dir}")
    print(f"Results saved to: {results_dir}")


def main(run_single_task_fn, run_comparison_fn):
    """CLI entry point for comparison."""
    parser = argparse.ArgumentParser(description="Compare steering methods")
    parser.add_argument("--model", default="EleutherAI/gpt-neo-125M",
                        help="Model name")
    parser.add_argument("--tasks", default="boolq",
                        help="Comma-separated lm-eval tasks")
    parser.add_argument("--methods", default="caa",
                        help="Comma-separated methods (caa,sae,fgaa)")
    parser.add_argument("--num-pairs", type=int, default=50,
                        help="Number of contrastive pairs")
    parser.add_argument("--scales", default="1.0",
                        help="Comma-separated steering scales")
    parser.add_argument("--caa-layers", default="12",
                        help="Layer(s) for CAA steering")
    parser.add_argument("--sae-layers", default="12",
                        help="Layer(s) for SAE/FGAA steering")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--batch-size", default=1,
                        help="Batch size (int or 'auto')")
    parser.add_argument("--max-batch-size", type=int, default=8,
                        help="Max batch size for lm-eval internal batching")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit eval examples")
    parser.add_argument("--output-dir",
                        default="wisent/comparison/comparison_results",
                        help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Train/test split ratio")
    parser.add_argument("--extraction-strategy", default="mc_balanced",
                        help="Extraction strategy (comma-separated)")
    parser.add_argument("--bos-features-source", default="detected",
                        help="BOS features source for FGAA")

    args = parser.parse_args()

    # Parse comma-separated values
    tasks = [t.strip() for t in args.tasks.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    scales = [float(s.strip()) for s in args.scales.split(",")]
    extraction_strategies = [s.strip()
                             for s in args.extraction_strategy.split(",")]

    # Parse batch_size (can be int or "auto")
    batch_size = (args.batch_size if args.batch_size == "auto"
                  else int(args.batch_size))

    run_comparison_fn(
        model_name=args.model,
        tasks=tasks,
        methods=methods,
        num_pairs=args.num_pairs,
        steering_scales=scales,
        device=args.device,
        batch_size=batch_size,
        max_batch_size=args.max_batch_size,
        eval_limit=args.limit,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        caa_layers=args.caa_layers,
        sae_layers=args.sae_layers,
        extraction_strategies=extraction_strategies,
        bos_features_source=args.bos_features_source,
        run_single_task_fn=run_single_task_fn,
    )
