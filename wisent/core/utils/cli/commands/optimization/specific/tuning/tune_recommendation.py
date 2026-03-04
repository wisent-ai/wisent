"""CLI handler for tune-recommendation command.

Two subcommands:
  collect-ground-truth  (GPU)  -- run all methods, record accuracy
  optimize-config       (CPU)  -- Optuna over recommendation params
"""
from __future__ import annotations



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
    kwargs = {}
    if getattr(args, "methods", None):
        kwargs["methods"] = tuple(m.strip() for m in args.methods.split(","))
    collect_ground_truth(
        model=args.model, zwiad_dir=args.zwiad_dir, limit=args.limit,
        lr_lower_bound=args.lr_lower_bound, lr_upper_bound=args.lr_upper_bound,
        alpha_lower_bound=args.alpha_lower_bound, alpha_upper_bound=args.alpha_upper_bound,
        optuna_szlak_reg_min=args.optuna_szlak_reg_min,
        optuna_nurt_steps_min=args.optuna_nurt_steps_min,
        optuna_nurt_steps_max=args.optuna_nurt_steps_max,
        optuna_wicher_concept_dims=args.optuna_wicher_concept_dims,
        optuna_wicher_steps_min=args.optuna_wicher_steps_min,
        optuna_wicher_steps_max=args.optuna_wicher_steps_max,
        optuna_przelom_target_modes=args.optuna_przelom_target_modes,
        benchmarks=benchmarks, output_path=args.output,
        device=getattr(args, "device", None), n_trials=args.n_trials,
        **kwargs,
        benchmark_start=getattr(args, "benchmark_start", None),
        benchmark_end=getattr(args, "benchmark_end", None),
        use_geometry_selection=getattr(args, "use_geometry_selection", False),
        per_type=args.per_type, fine_geometry=getattr(args, "fine_geometry", False),
        optuna_grom_gate_dim_min=args.optuna_grom_gate_dim_min,
        optuna_grom_gate_dim_max=args.optuna_grom_gate_dim_max,
        optuna_grom_intensity_dim_min=args.optuna_grom_intensity_dim_min,
        optuna_grom_intensity_dim_max=args.optuna_grom_intensity_dim_max,
        optuna_grom_sparse_weight_min=args.optuna_grom_sparse_weight_min,
        optuna_grom_sparse_weight_max=args.optuna_grom_sparse_weight_max,
        geo_default_score=args.geo_default_score, geo_blend_default=args.geo_blend_default,
        geo_default_scale=args.geo_default_scale,
        zwiad_ranges=args.zwiad_ranges, zwiad_weights=args.zwiad_weights,
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
        top_k=args.top_k,
    )
    n_trials = args.n_trials
    output = getattr(args, "output", None)
    if output is None:
        from pathlib import Path
        output = str(
            Path.home() / ".wisent"
            / "learned_recommendation_config.json")
    optimizer.tune(
        n_trials=n_trials, output_path=output,
        seed=args.seed)
