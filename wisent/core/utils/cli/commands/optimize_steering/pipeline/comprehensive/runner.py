"""Per-method optimization runner for comprehensive comparison."""
from __future__ import annotations

import logging
import tempfile
from typing import Any, Dict, List, Optional

from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN,
    TRIALS_PER_DIMENSION_MULTIPLIER,
)

logger = logging.getLogger(__name__)


def run_method_search(
    model: str, task_name: str, method: str,
    pairs_file: str, num_layers: int,
    device: Optional[str], verbose: bool,
    backend: str,
    search_overrides: Dict[str, Any],
    early_rejection_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run optimization for a single method.

    Uses the UnifiedOptimizer with the specified backend (hyperopt or
    optuna). The number of trials is derived from the search space
    dimensionality: len(space) * TRIALS_PER_DIMENSION_MULTIPLIER.
    """
    from wisent.core.utils.cli.optimize_steering.search_space import (
        get_method_space,
    )
    from wisent.core.utils.cli.optimize_steering.pipeline import (
        create_objective,
    )
    from wisent.core.utils.services.optimization.core.unified_optimizer import (
        UnifiedOptimizer,
    )

    space = get_method_space(method, num_layers)
    space = _apply_search_overrides(space, search_overrides)

    n_trials = len(space) * TRIALS_PER_DIMENSION_MULTIPLIER

    if verbose:
        print(f"   {method}: {n_trials} total configurations "
              f"(backend={backend})")

    optimizer = UnifiedOptimizer(
        backend=backend, direction="maximize",
    )

    with tempfile.TemporaryDirectory() as work_dir:
        raw_objective = create_objective(
            method=method, model=model, task=task_name,
            num_layers=num_layers, limit=None, device=device,
            work_dir=work_dir, enriched_pairs_file=pairs_file,
        )
        if early_rejection_config["enabled"]:
            objective = _wrap_with_early_rejection(
                raw_objective, early_rejection_config,
            )
        else:
            objective = raw_objective
        result = optimizer.optimize(objective, space, n_trials)

    if verbose:
        print(f"   {method}: best={result.best_score:.4f} "
              f"({len(result.all_trials)} trials)")

    return {
        "method": method, "best_score": result.best_score,
        "best_params": result.best_params,
        "all_trials": result.all_trials,
        "n_trials": n_trials, "backend": backend,
    }


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
