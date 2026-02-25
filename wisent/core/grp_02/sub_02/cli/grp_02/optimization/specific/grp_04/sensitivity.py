"""CLI handler for the sensitivity command.

Dispatches to four subcommands:
  run       -- SensitivityEngine one-at-a-time sweep
  report    -- Display ranked results from JSON
  optimize  -- Optuna joint optimization of top-N constants
  calibrate -- Threshold calibration from empirical distributions
"""
from __future__ import annotations

import json
from pathlib import Path

from wisent.core.constants import (
    DEFAULT_LIMIT, DEFAULT_N_TRIALS, DISPLAY_TOP_N_SMALL,
    SEPARATOR_WIDTH_SENSITIVITY, SENSITIVITY_DEFAULT_STEPS,
    SENSITIVITY_DEFAULT_THRESHOLD, EVIDENCE_TRAIN_SPLIT_NUM,
    EVIDENCE_TRAIN_SPLIT_DEN,
)


def execute_sensitivity(args) -> None:
    """Dispatch to the appropriate sensitivity subcommand."""
    action = getattr(args, "sensitivity_action", None)
    if action == "run":
        _run_sensitivity(args)
    elif action == "report":
        _show_report(args)
    elif action == "optimize":
        _run_optimize(args)
    elif action == "calibrate":
        _run_calibrate(args)
    else:
        print("Usage: wisent sensitivity "
              "{run,report,optimize,calibrate}")


def _load_resources(model_name, task_name, limit, device):
    """Load model, pairs, and evaluator for sensitivity operations."""
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import (
        build_contrastive_pairs,
    )
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.evaluators.steering_evaluators import (
        SteeringEvaluatorFactory, EvaluatorConfig,
    )

    print(f"Loading model: {model_name}")
    model = WisentModel(model_name, device=device)

    print(f"Generating contrastive pairs for: {task_name}")
    all_pairs = build_contrastive_pairs(
        task_name=task_name, limit=limit,
    )
    split = len(all_pairs) * EVIDENCE_TRAIN_SPLIT_NUM // EVIDENCE_TRAIN_SPLIT_DEN
    train_pairs = ContrastivePairSet(
        name=f"{task_name}_train", pairs=all_pairs[:split],
    )
    test_pairs = ContrastivePairSet(
        name=f"{task_name}_test", pairs=all_pairs[split:],
    )
    print(f"Pairs: {len(train_pairs)} train, {len(test_pairs)} test")

    config = EvaluatorConfig(
        evaluator_type="task", task=task_name,
    )
    evaluator = SteeringEvaluatorFactory.create(
        config=config, model_name=model_name,
    )
    return model, train_pairs, test_pairs, evaluator


def _run_sensitivity(args) -> None:
    """Execute sensitivity sweep across registered constants."""
    from wisent.core.steering_optimizer.constants_registry.sensitivity import (
        SensitivityEngine,
    )

    model_name = args.model
    task_name = args.task
    method_name = args.method
    steps = getattr(args, "steps", SENSITIVITY_DEFAULT_STEPS)
    limit = getattr(args, "limit", DEFAULT_LIMIT)
    device = getattr(args, "device", None)
    output_path = getattr(args, "output", "sensitivity_result.json")
    group = getattr(args, "group", None)
    method_filter = getattr(args, "method_filter", None)

    model, train_pairs, test_pairs, evaluator = _load_resources(
        model_name, task_name, limit, device,
    )

    engine = SensitivityEngine(
        model=model,
        method_name=method_name,
        task_name=task_name,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        evaluator=evaluator,
        steps=steps,
        verbose=True,
    )

    result = engine.run(
        group=group,
        method_filter=method_filter,
        limit=limit,
    )

    out = Path(output_path)
    result.save(out)
    print(f"\nResults saved to: {out}")
    print(f"Total: {result.constants_tested} constants in "
          f"{result.total_time:.1f}s")

    ranked = result.ranked()
    if ranked:
        print(f"\nTop 10 most sensitive constants:")
        for r in ranked[:DISPLAY_TOP_N_SMALL]:
            print(f"  {r.name}: sensitivity={r.sensitivity:.4f} "
                  f"(best={r.best_value:.4g})")


def _show_report(args) -> None:
    """Display ranked sensitivity results from saved JSON."""
    from wisent.core.steering_optimizer.constants_registry.sensitivity import (
        SensitivityResult,
    )

    input_path = args.input
    threshold = getattr(args, "threshold", SENSITIVITY_DEFAULT_THRESHOLD)
    top_n = getattr(args, "top_n", None)

    result = SensitivityResult.load(Path(input_path))

    print(f"Sensitivity Report: {result.model_name}")
    print(f"Task: {result.task_name}, Method: {result.method_name}")
    print(f"Baseline: {result.baseline_score:.4f}, "
          f"Constants tested: {result.constants_tested}")
    print(f"Total time: {result.total_time:.1f}s\n")

    ranked = result.ranked()
    filtered = [r for r in ranked if r.sensitivity >= threshold]

    if top_n is not None:
        filtered = filtered[:top_n]

    print(f"{'Rank':<6}{'Name':<35}{'Group':<6}"
          f"{'Sensitivity':<14}{'Best Value':<14}{'Current':<14}")
    print("-" * SEPARATOR_WIDTH_SENSITIVITY)

    for idx, r in enumerate(filtered):
        print(f"{idx + 1:<6}{r.name:<35}{r.group:<6}"
              f"{r.sensitivity:<14.4f}{r.best_value:<14.4g}"
              f"{r.current_value:<14.4g}")

    if not filtered:
        print("No constants above threshold "
              f"{threshold:.4f}")

    n_above = len([r for r in ranked if r.sensitivity >= threshold])
    print(f"\n{n_above}/{len(ranked)} constants above "
          f"threshold {threshold:.4f}")


def _run_optimize(args) -> None:
    """Jointly optimize top-N sensitive constants via Optuna."""
    from wisent.core.steering_optimizer.constants_registry.sensitivity import (
        SensitivityResult, OptunaConstantOptimizer,
    )

    model_name = args.model
    task_name = args.task
    method_name = args.method
    input_path = args.input
    top_n = getattr(args, "top_n", 20)
    n_trials = getattr(args, "n_trials", DEFAULT_N_TRIALS)
    sens_threshold = getattr(args, "sensitivity_threshold", 0.0)
    output_path = getattr(args, "output", None)
    device = getattr(args, "device", None)
    limit = getattr(args, "limit", DEFAULT_LIMIT)

    sensitivity_result = SensitivityResult.load(Path(input_path))
    print(f"Loaded sensitivity: {sensitivity_result.constants_tested} "
          f"constants from {input_path}")

    # Reconstruct the fixed operating point from sensitivity result
    from wisent.core.cli.optimization.core.method_optimizer_config import (
        OptimizationConfig,
    )
    from wisent.core.activations import ExtractionStrategy

    op = sensitivity_result.operating_point
    fixed_config = OptimizationConfig(
        method_name=op["method_name"],
        layers=op["layers"],
        token_aggregation=ExtractionStrategy(op["token_aggregation"]),
        prompt_strategy=ExtractionStrategy(op["prompt_strategy"]),
        strength=op["strength"],
        strategy=op.get("strategy", "constant"),
        method_params=op.get("method_params", {}),
    )

    model, train_pairs, test_pairs, evaluator = _load_resources(
        model_name, task_name, limit, device,
    )

    optimizer = OptunaConstantOptimizer(
        model=model,
        method_name=method_name,
        task_name=task_name,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        evaluator=evaluator,
        fixed_config=fixed_config,
        verbose=True,
    )

    profile = optimizer.optimize_from_sensitivity(
        sensitivity_result=sensitivity_result,
        top_n=top_n,
        n_trials=n_trials,
        threshold=sens_threshold,
    )

    if profile is None:
        print("No constants to optimize.")
        return

    _save_profile(profile, output_path)


def _run_calibrate(args) -> None:
    """Calibrate Group E thresholds from empirical distributions."""
    from wisent.core.steering_optimizer.constants_registry.calibration import (
        ThresholdCalibrator,
    )

    model_name = args.model
    task_name = args.task
    pairs_file = getattr(args, "pairs_file", None)
    output_path = getattr(args, "output", None)
    device = getattr(args, "device", None)
    limit = getattr(args, "limit", DEFAULT_LIMIT)

    if pairs_file:
        print(f"Loading pairs from: {pairs_file}")
        with open(pairs_file) as f:
            pairs = json.load(f)
        model = model_name
    else:
        model, train_pairs, test_pairs, _ = _load_resources(
            model_name, task_name, limit, device,
        )
        pairs = list(train_pairs) + list(test_pairs)

    calibrator = ThresholdCalibrator(
        model=model,
        task_name=task_name,
        pairs=pairs,
        verbose=True,
    )

    profile = calibrator.calibrate_to_profile()
    _save_profile(profile, output_path)


def _save_profile(profile, output_path):
    """Save a ConstantProfile via the profile manager."""
    from wisent.core.steering_optimizer.constants_registry.profiles import (
        ConstantProfileManager,
    )

    if output_path:
        out_dir = Path(output_path).parent
        manager = ConstantProfileManager(profiles_dir=out_dir)
    else:
        manager = ConstantProfileManager()

    profile_path = manager.save(profile)

    print(f"\nProfile saved to: {profile_path}")
    print(f"Constants: {len(profile.constants)}")
    for name, value in sorted(profile.constants.items()):
        print(f"  {name}: {value:.6g}")
