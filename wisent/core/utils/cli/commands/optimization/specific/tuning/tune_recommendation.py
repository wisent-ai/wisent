"""CLI handler for tune-recommendation command.

Two subcommands:
  collect-ground-truth  (GPU)  -- run all methods, record accuracy
  optimize-config       (CPU)  -- Optuna over recommendation params
"""
from __future__ import annotations

from wisent.core.utils.config_tools.constants import (
    DEFAULT_LIMIT, DEFAULT_N_TRIALS, DEFAULT_RANDOM_SEED,
    RECOMMEND_COLLECTOR_PER_TYPE, RECOMMEND_N_TRIALS, RECOMMEND_TOP_K,
)


def execute_tune_recommendation(args) -> None:
    """Dispatch to the appropriate subcommand."""
    sub = getattr(args, "subcommand", None)
    if sub == "collect-ground-truth":
        _collect(args)
    elif sub == "optimize-config":
        _optimize(args)
    else:
        print("Usage: wisent tune-recommendation "
              "{collect-ground-truth,optimize-config}")


def _collect(args) -> None:
    from wisent.core.reading.modules.modules.steering.analysis.recommendation import (
        collect_ground_truth,
    )
    benchmarks = None
    if getattr(args, "benchmarks", None):
        benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    methods = None
    if getattr(args, "methods", None):
        methods = [m.strip() for m in args.methods.split(",")]
    collect_ground_truth(
        model=args.model,
        benchmarks=benchmarks,
        output_path=args.output,
        zwiad_dir=getattr(args, "zwiad_dir", "zwiad_results"),
        limit=getattr(args, "limit", DEFAULT_LIMIT),
        device=getattr(args, "device", None),
        methods=methods,
        n_trials=getattr(args, "n_trials", DEFAULT_N_TRIALS),
        benchmark_start=getattr(args, "benchmark_start", None),
        benchmark_end=getattr(args, "benchmark_end", None),
        use_geometry_selection=getattr(
            args, "use_geometry_selection", False),
        per_type=getattr(args, "per_type", RECOMMEND_COLLECTOR_PER_TYPE),
        fine_geometry=getattr(args, "fine_geometry", False),
    )


def _optimize(args) -> None:
    from wisent.core.reading.modules.modules.steering.analysis.recommendation import (
        GroundTruthDataset, RecommendationOptimizer,
    )
    dataset = GroundTruthDataset.load(args.ground_truth)
    print(f"Loaded {len(dataset)} ground-truth records")
    optimizer = RecommendationOptimizer(
        dataset=dataset,
        objective_type=getattr(args, "objective", "top1"),
        top_k=getattr(args, "top_k", RECOMMEND_TOP_K),
    )
    n_trials = getattr(args, "n_trials", RECOMMEND_N_TRIALS)
    output = getattr(args, "output", None)
    if output is None:
        from pathlib import Path
        output = str(
            Path.home() / ".wisent"
            / "learned_recommendation_config.json")
    optimizer.tune(
        n_trials=n_trials, output_path=output,
        seed=getattr(args, "seed", DEFAULT_RANDOM_SEED))
