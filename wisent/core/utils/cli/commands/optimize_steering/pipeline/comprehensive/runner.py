"""Per-method optimization runner for comprehensive comparison."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from wisent.core.utils.config_tools.constants import (
    JSON_INDENT,
    SCORE_RANGE_MIN,
    TRIALS_PER_DIMENSION_MULTIPLIER,
)

logger = logging.getLogger(__name__)


def run_method_search(
    model: str, task_name: str, method: str,
    train_pairs_file: str, test_pairs_file: str,
    num_layers: int,
    device: Optional[str], verbose: bool,
    backend: str,
    search_overrides: Dict[str, Any],
    early_rejection_config: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run optimization for a single method.

    Uses the BaseOptimizer with the specified backend (hyperopt or
    optuna). The number of trials is derived from the search space
    dimensionality: len(space) * TRIALS_PER_DIMENSION_MULTIPLIER.

    Trains on train_pairs_file and evaluates on test_pairs_file to
    prevent data leakage.
    """
    from wisent.core.utils.cli.optimize_steering.search_space import (
        get_method_space,
    )
    from wisent.core.utils.cli.optimize_steering.pipeline import (
        create_objective,
    )
    from wisent.core.utils.services.optimization.core.atoms import (
        BaseOptimizer,
        HPOConfig,
    )

    space = get_method_space(method, num_layers)
    space = _apply_search_overrides(space, search_overrides)

    n_trials = len(space) * TRIALS_PER_DIMENSION_MULTIPLIER

    if verbose:
        print(f"   {method}: {n_trials} total configurations "
              f"(backend={backend})")

    optimizer = BaseOptimizer()
    optimizer.direction = "maximize"

    with tempfile.TemporaryDirectory() as work_dir:
        raw_objective = create_objective(
            method=method, model=model, task=task_name,
            num_layers=num_layers, limit=None, device=device,
            work_dir=work_dir, train_pairs_file=train_pairs_file,
            test_pairs_file=test_pairs_file,
        )
        if early_rejection_config["enabled"]:
            objective = _wrap_with_early_rejection(
                raw_objective, early_rejection_config,
            )
        else:
            objective = raw_objective
        result = optimizer.optimize_fn(objective, space, n_trials, cfg=HPOConfig(backend=backend))

    if verbose:
        print(f"   {method}: best={result.best_score:.4f} "
              f"({len(result.all_trials)} trials)")

    summary = {
        "method": method, "best_score": result.best_score,
        "best_params": result.best_params,
        "all_trials": result.all_trials,
        "n_trials": n_trials, "backend": backend,
    }

    if output_dir:
        _replay_best_config(
            model, task_name, method, result.best_params,
            train_pairs_file, test_pairs_file,
            device, verbose, output_dir, summary,
        )

    return summary


def _apply_search_overrides(
    space: Dict, overrides: Dict,
) -> Dict:
    """Apply user search overrides to the Optuna search space."""
    from wisent.core.utils.services.optimization.core.parameters import (
        CategoricalParam,
    )
    if "search_layers" in overrides and overrides["search_layers"]:
        layers = [
            int(x) for x in overrides["search_layers"].split(",")
        ]
        space["layer"] = CategoricalParam(choices=layers)
    if "search_strengths" in overrides and overrides["search_strengths"]:
        strengths = [
            float(x) for x in overrides["search_strengths"].split(",")
        ]
        space["strength"] = CategoricalParam(choices=strengths)
    if "search_strategies" in overrides and overrides["search_strategies"]:
        strats = [
            s.strip()
            for s in overrides["search_strategies"].split(",")
        ]
        space["steering_strategy"] = CategoricalParam(choices=strats)
    return space


def _replay_best_config(
    model: str, task_name: str, method: str,
    best_params: Dict[str, Any],
    train_pairs_file: str, test_pairs_file: str,
    device: Optional[str], verbose: bool,
    output_dir: str, summary: Dict[str, Any],
) -> None:
    """Re-run the best config and persist responses + scores."""
    from wisent.core.utils.cli.optimize_steering.pipeline import (
        run_pipeline, _build_config,
    )

    method_dir = os.path.join(output_dir, method.lower())
    os.makedirs(method_dir, exist_ok=True)

    if verbose:
        print(f"   Replaying best {method} config to persist responses...")

    config, strength = _build_config(method, best_params)
    result = run_pipeline(
        model=model, task=task_name, config=config,
        work_dir=method_dir, strength=strength,
        limit=None, device=device,
        train_pairs_file=train_pairs_file,
        test_pairs_file=test_pairs_file,
    )

    summary_path = os.path.join(method_dir, "optimization_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=JSON_INDENT, default=str)

    if verbose:
        print(f"   Saved responses to {method_dir}/responses.json")
        print(f"   Saved scores to {method_dir}/scores.json")


def _wrap_with_early_rejection(objective_fn, config: Dict[str, Any]):
    """Wrap objective to return SCORE_RANGE_MIN for rejected trials."""
    cv_threshold = config["cv_threshold"]

    def wrapped(params: dict) -> float:
        score = objective_fn(params)
        if score < cv_threshold:
            logger.info(
                f"   Early rejection: score {score:.4f} < "
                f"cv_threshold {cv_threshold}",
            )
            return SCORE_RANGE_MIN
        return score

    return wrapped
