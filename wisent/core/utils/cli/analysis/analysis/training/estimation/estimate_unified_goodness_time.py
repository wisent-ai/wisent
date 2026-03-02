"""CLI entry point for estimating unified goodness training time."""
from wisent.core.utils.config_tools.constants import DEFAULT_SPLIT_RATIO, BENCHMARK_FAST_LOAD_THRESHOLD, SEPARATOR_WIDTH_WIDE, SEPARATOR_WIDTH_HALF
from wisent.core.utils.cli.analysis.training.estimate_time_functions import (
    BENCHMARK_SIZES, BENCHMARK_LOAD_TIMES, estimate_runtime,
    get_benchmark_load_time, get_benchmark_size, format_time,
)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate runtime for train-unified-goodness command (ALL 327 benchmarks)"
    )
    parser.add_argument(
        "--max-benchmarks", type=int, default=None,
        help="Maximum number of benchmarks to use (default: ALL 327)"
    )
    parser.add_argument(
        "--cap-pairs-per-benchmark", type=int, default=None,
        help="Cap pairs per benchmark (benchmarks with more get randomly sampled down)"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=DEFAULT_SPLIT_RATIO,
        help="Fraction of pairs for training (default: 0.8 = 80%% train, 20%% eval)"
    )
    parser.add_argument(
        "--eval-scales", type=int, required=True,
        help="Number of steering scales to evaluate"
    )
    parser.add_argument(
        "--skip-evaluation", action="store_true",
        help="Skip evaluation phase"
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "mps", "auto"], required=True,
        help="Device for computation"
    )
    parser.add_argument(
        "--show-breakdown", action="store_true",
        help="Show detailed time breakdown"
    )
    parser.add_argument(
        "--show-benchmarks", action="store_true",
        help="List all benchmarks with their estimated sizes"
    )
    
    args = parser.parse_args()
    
    # Import benchmark list - use ALL benchmarks (same as test_all_benchmarks.py)
    try:
        from wisent.core.utils.cli.train_unified_goodness import load_all_benchmarks
        all_benchmark_names, broken = load_all_benchmarks()
    except ImportError:
        print("Error: Cannot import load_all_benchmarks. Run from wisent-open-source directory.")
        sys.exit(1)
    
    # Convert to dict format for estimate_runtime (name -> config with priority)
    # Since we don't have priority info from the JSON files, we'll estimate based on name
    selected = {}
    for name in all_benchmark_names:
        # Estimate priority based on common fast benchmarks
        if name in BENCHMARK_LOAD_TIMES:
            if BENCHMARK_LOAD_TIMES[name] < 15:
                priority = 'high'
            elif BENCHMARK_LOAD_TIMES[name] < BENCHMARK_FAST_LOAD_THRESHOLD:
                priority = 'medium'
            else:
                priority = 'low'
        else:
            priority = 'medium'  # Default
        selected[name] = {'priority': priority}
    
    # Apply max limit
    if args.max_benchmarks and len(selected) > args.max_benchmarks:
        selected = dict(list(selected.items())[:args.max_benchmarks])
    
    # Print header
    print("\n" + "=" * SEPARATOR_WIDTH_WIDE)
    print("  UNIFIED GOODNESS TRAINING - TIME ESTIMATE")
    print("=" * SEPARATOR_WIDTH_WIDE)
    
    # Print configuration
    print(f"\n CONFIGURATION:")
    print(f"   Total available: {len(all_benchmark_names)} benchmarks")
    print(f"   Using: {len(selected)} benchmarks")
    if args.cap_pairs_per_benchmark:
        print(f"   Pairs cap: {args.cap_pairs_per_benchmark} per benchmark (random sampling)")
    else:
        print(f"   Pairs: ALL available (80% train / 20% eval)")
    print(f"   Train ratio: {args.train_ratio}")
    print(f"   Eval scales: {args.eval_scales}")
    print(f"   Skip evaluation: {args.skip_evaluation}")
    print(f"   Device: {args.device}")
    
    # Calculate estimate
    estimate = estimate_runtime(
        benchmarks=selected,
        num_eval_scales=args.eval_scales,
        train_ratio=args.train_ratio,
        skip_evaluation=args.skip_evaluation,
        device=args.device,
        cap_pairs_per_benchmark=args.cap_pairs_per_benchmark,
    )
    
    # Show benchmarks if requested
    if args.show_benchmarks:
        print(f"\n SELECTED BENCHMARKS ({len(selected)}):")
        print(f"   {'#':<4s} {'Benchmark':<30s} {'Priority':<8s} {'Pairs':>8s} {'Train':>8s} {'Eval':>8s}")
        print(f"   {'-'*4} {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for i, detail in enumerate(estimate['benchmark_details'], 1):
            print(f"   {i:<4d} {detail['name']:<30s} {detail['priority']:<8s} "
                  f"{detail['total_pairs']:>8d} {detail['train_pairs']:>8d} {detail['eval_pairs']:>8d}")
        print(f"   {'-'*4} {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        print(f"   {'':4s} {'TOTAL':<30s} {'':8s} {estimate['total_pairs']:>8d} "
              f"{estimate['total_train_pairs']:>8d} {estimate['total_eval_pairs']:>8d}")
    
    # Print summary
    print(f"\n  ESTIMATED TOTAL TIME: {format_time(estimate['total_seconds'])}")
    print(f"   ({estimate['total_minutes']:.1f} minutes / {estimate['total_hours']:.2f} hours)")
    
    # Print breakdown
    if args.show_breakdown:
        print(f"\n TIME BREAKDOWN:")
        print(f"   Model loading:        {format_time(estimate['model_loading']):>10s}")
        print(f"   Benchmark loading:    {format_time(estimate['benchmark_loading']):>10s}")
        print(f"   Benchmark setup:      {format_time(estimate['benchmark_setup']):>10s}")
        print(f"   Pair generation:      {format_time(estimate['pair_generation']):>10s}")
        print(f"   Activation collection:{format_time(estimate['activation_collection']):>10s}")
        print(f"   Vector training:      {format_time(estimate['vector_training']):>10s}")
        print(f"   Evaluation:           {format_time(estimate['evaluation']):>10s}")
        print(f"   " + "-" * SEPARATOR_WIDTH_HALF)
        print(f"   TOTAL:                {format_time(estimate['total_seconds']):>10s}")
    
    # Print data summary
    print(f"\n DATA SUMMARY:")
    print(f"   Total pairs:          {estimate['total_pairs']:,}")
    print(f"   Training pairs (80%): {estimate['total_train_pairs']:,}")
    print(f"   Eval pairs (20%):     {estimate['total_eval_pairs']:,}")
    if not args.skip_evaluation:
        print(f"   Eval samples (x{args.eval_scales} scales): {estimate.get('total_eval_samples', 0):,}")
    
    # Print quick reference table with realistic sizes
    print(f"\n QUICK REFERENCE - OTHER CONFIGURATIONS:")
    print(f"   {'Config':<45s} {'Pairs':>10s} {'Time':>12s}")
    print(f"   {'-'*45} {'-'*10} {'-'*12}")
    
    # Build configs with subset of benchmarks (using full list now)
    def get_subset(max_n=None):
        subset = {}
        for name in all_benchmark_names:
            if name in BENCHMARK_LOAD_TIMES:
                if BENCHMARK_LOAD_TIMES[name] < 15:
                    priority = 'high'
                elif BENCHMARK_LOAD_TIMES[name] < BENCHMARK_FAST_LOAD_THRESHOLD:
                    priority = 'medium'
                else:
                    priority = 'low'
            else:
                priority = 'medium'
            subset[name] = {'priority': priority}
        if max_n and len(subset) > max_n:
            subset = dict(list(subset.items())[:max_n])
        return subset
    
    configs = [
        ('10 benchmarks, skip eval', get_subset(10), True, None),
        ('50 benchmarks, full eval', get_subset(50), False, None),
        (f'ALL ({len(all_benchmark_names)}) benchmarks, no cap', get_subset(), False, None),
        (f'ALL ({len(all_benchmark_names)}) benchmarks, cap 10000', get_subset(), False, 10000),
        (f'ALL ({len(all_benchmark_names)}) benchmarks, cap 5000', get_subset(), False, 5000),
        (f'ALL ({len(all_benchmark_names)}) benchmarks, cap 1000', get_subset(), False, 1000),
    ]
    
    for name, bench_subset, skip_eval, limit in configs:
        est = estimate_runtime(
            benchmarks=bench_subset,
            num_eval_scales=args.eval_scales,
            train_ratio=args.train_ratio,
            skip_evaluation=skip_eval,
            device=args.device,
            cap_pairs_per_benchmark=limit,
        )
        print(f"   {name:<45s} {est['total_pairs']:>10,} {format_time(est['total_seconds']):>12s}")
    
    print("\n" + "=" * SEPARATOR_WIDTH_WIDE)

    return estimate


if __name__ == "__main__":
    main()
