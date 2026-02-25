"""CLI handler for the evidence command.

Dispatches to four subcommands:
  compare-axis -- Run axis comparison, record to evidence ledger
  list         -- Show evidence records
  report       -- Show computed search-space reductions
  sync         -- Upload / download ledger to/from GCS
"""
from __future__ import annotations

from wisent.core.constants import (
    DEFAULT_LIMIT, DISPLAY_TOP_N_TINY, SEPARATOR_WIDTH_ULTRA,
    EVIDENCE_TRAIN_SPLIT_NUM, EVIDENCE_TRAIN_SPLIT_DEN,
)

# Axis name -> list of string values the search-space uses
_AXIS_DEFAULTS = {
    "extraction_strategy": [
        "last_token", "mean_pooling", "first_token",
        "max_pooling", "continuation_token",
    ],
    "prompt_strategy": [
        "chat_template", "direct_completion", "multiple_choice",
        "role_playing", "instruction_following",
    ],
    "steering_strategy": [
        "constant", "initial_only", "diminishing",
        "increasing", "gaussian",
    ],
    "strength": ["0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "2.0"],
    "num_directions": ["1", "2", "3", "5"],
    "gate_hidden_dim": ["32", "64", "128"],
}


def execute_evidence(args) -> None:
    """Dispatch to the appropriate evidence subcommand."""
    action = getattr(args, "evidence_action", None)
    if action == "compare-axis":
        _run_compare_axis(args)
    elif action == "list":
        _run_list(args)
    elif action == "report":
        _run_report(args)
    elif action == "sync":
        _run_sync(args)
    else:
        print("Usage: wisent evidence "
              "{compare-axis,list,report,sync}")


def _load_resources(model_name, task_name, limit, device):
    """Load model, pairs, and evaluator (same pattern as sensitivity)."""
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
    all_pairs = build_contrastive_pairs(task_name=task_name, limit=limit)
    split = len(all_pairs) * EVIDENCE_TRAIN_SPLIT_NUM // EVIDENCE_TRAIN_SPLIT_DEN
    train_pairs = ContrastivePairSet(
        name=f"{task_name}_train", pairs=all_pairs[:split],
    )
    test_pairs = ContrastivePairSet(
        name=f"{task_name}_test", pairs=all_pairs[split:],
    )
    print(f"Pairs: {len(train_pairs)} train, {len(test_pairs)} test")
    config = EvaluatorConfig(evaluator_type="task", task=task_name)
    evaluator = SteeringEvaluatorFactory.create(
        config=config, model_name=model_name,
    )
    return model, train_pairs, test_pairs, evaluator


def _run_compare_axis(args) -> None:
    """Compare values of one axis and record to ledger."""
    from wisent.core.cli.optimization.core.method_optimizer import (
        MethodOptimizer,
    )
    from wisent.core.steering_optimizer.constants_registry.evidence import (
        EvidenceLedger, AxisEvidence, compute_dominant_values,
    )

    model_name = args.model
    task_name = args.task
    method_name = args.method
    axis = args.axis
    limit = getattr(args, "limit", DEFAULT_LIMIT)
    device = getattr(args, "device", None)

    raw_values = getattr(args, "values", None)
    if raw_values:
        values = [v.strip() for v in raw_values.split(",")]
    elif axis in _AXIS_DEFAULTS:
        values = _AXIS_DEFAULTS[axis]
    else:
        print(f"No default values for axis '{axis}'. "
              f"Supply --values explicitly.")
        return

    model, train_pairs, test_pairs, evaluator = _load_resources(
        model_name, task_name, limit, device,
    )
    optimizer = MethodOptimizer(
        model=model, method_name=method_name, verbose=True,
    )

    scores: dict[str, float] = {}
    print(f"\nComparing axis '{axis}' with values: {values}")
    for val in values:
        print(f"\n--- Testing {axis}={val} ---")
        custom = _axis_value_to_custom(axis, val, model.num_layers)
        configs = optimizer.generate_search_space(
            num_layers=model.num_layers, **custom,
        )
        if not configs:
            print(f"  No configs generated for {axis}={val}, skipping")
            continue
        summary = optimizer.optimize(
            train_pairs=train_pairs, test_pairs=test_pairs,
            evaluator=evaluator, task_name=task_name, configs=configs,
        )
        best = summary.best_result.score if summary.best_result else 0.0
        scores[val] = best
        print(f"  Best score for {val}: {best:.4f}")

    if not scores:
        print("No scores collected. Nothing to record.")
        return

    dominant, margin = compute_dominant_values(scores)
    n_samples = len(train_pairs) + len(test_pairs)

    evidence = AxisEvidence(
        axis_name=axis, model_name=model_name,
        task_name=task_name, method_name=method_name,
        tested_values=values, scores=scores,
        dominant_values=dominant, margin=margin,
        confidence=1.0, n_samples=n_samples,
        source="compare_axis",
    )

    ledger = EvidenceLedger()
    path = ledger.record(evidence)
    print(f"\nEvidence recorded: {evidence.id}")
    print(f"  Dominant: {dominant} (margin={margin:.4f})")
    print(f"  Saved to: {path}")


def _axis_value_to_custom(axis, val, num_layers):
    """Convert an axis+value into generate_search_space kwargs."""
    if axis == "extraction_strategy":
        return {"custom_token_aggregations": [val]}
    if axis == "prompt_strategy":
        return {"custom_prompt_strategies": [val]}
    if axis == "steering_strategy":
        return {}
    if axis == "strength":
        return {"custom_strengths": [float(val)]}
    if axis in ("num_directions", "sensor_layer", "gate_hidden_dim",
                "intensity_hidden_dim", "optimization_steps"):
        return {"custom_method_params": {axis: [_try_numeric(val)]}}
    return {}


def _try_numeric(val):
    """Attempt to cast string to int or float."""
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def _run_list(args) -> None:
    """List evidence records."""
    from wisent.core.steering_optimizer.constants_registry.evidence import (
        EvidenceLedger,
    )
    ledger = EvidenceLedger()
    records = ledger.list_all()
    model_filter = getattr(args, "model", None)
    axis_filter = getattr(args, "axis", None)
    if model_filter:
        records = [r for r in records if r.model_name == model_filter]
    if axis_filter:
        records = [r for r in records if r.axis_name == axis_filter]
    if not records:
        print("No evidence records found.")
        return
    print(f"{'ID':<18}{'Axis':<24}{'Model':<36}"
          f"{'Task':<16}{'Dominant':<28}{'Conf':<6}")
    print("-" * SEPARATOR_WIDTH_ULTRA)
    for ev in records:
        dom = ",".join(ev.dominant_values[:DISPLAY_TOP_N_TINY])
        print(f"{ev.id:<18}{ev.axis_name:<24}"
              f"{ev.model_name:<36}{ev.task_name:<16}"
              f"{dom:<28}{ev.confidence:<6.2f}")
    print(f"\nTotal: {len(records)} records")


def _run_report(args) -> None:
    """Show what reductions the ledger produces for a model."""
    from wisent.core.steering_optimizer.constants_registry.evidence import (
        EvidenceLedger,
    )
    model_name = args.model
    task = getattr(args, "task", None)
    method = getattr(args, "method", None)
    ledger = EvidenceLedger()
    reductions = ledger.get_reductions(
        model_name, task_name=task, method_name=method,
    )
    if not reductions:
        print(f"No reductions for {model_name}")
        return
    print(f"Search-space reductions for: {model_name}")
    if task:
        print(f"  Task filter: {task}")
    print()
    for axis, red in sorted(reductions.items()):
        print(f"  {axis}:")
        print(f"    Keep:    {red.keep_values}")
        print(f"    Removed: {red.removed_values}")
        print(f"    Confidence: {red.confidence:.2f}")
        print(f"    Evidence: {len(red.evidence_ids)} record(s)")


def _run_sync(args) -> None:
    """Upload or download ledger to/from GCS."""
    from wisent.core.steering_optimizer.constants_registry.evidence import (
        EvidenceLedger,
    )
    ledger = EvidenceLedger()
    do_upload = getattr(args, "upload", False)
    do_download = getattr(args, "download", False)
    if not do_upload and not do_download:
        print("Specify --upload and/or --download")
        return
    if do_download:
        n = ledger.download_from_gcs()
        print(f"Downloaded {n} evidence file(s) from GCS")
    if do_upload:
        n = ledger.upload_to_gcs()
        print(f"Uploaded {n} evidence file(s) to GCS")
