"""Per-method optimization runner for comprehensive comparison."""
from __future__ import annotations

import itertools
import logging
import tempfile
from typing import Any, Dict, List, Optional

from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN, COMBO_OFFSET, OPTUNA_BACKEND_NAME,
)

logger = logging.getLogger(__name__)


def run_method_search(
    model: str, task_name: str, method: str,
    pairs_file: str, num_layers: int, limit: int,
    device: Optional[str], verbose: bool,
    search_strategy: str, n_trials: int, n_startup_trials: int,
    search_overrides: Dict[str, Any],
    early_rejection_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run optimization for a single method.

    Supports grid (exhaustive) and optuna (TPE sampling) strategies.
    """
    if search_strategy == "optuna":
        return _run_optuna_search(
            model=model, task_name=task_name, method=method,
            pairs_file=pairs_file, num_layers=num_layers, limit=limit,
            device=device, verbose=verbose, n_trials=n_trials,
            n_startup_trials=n_startup_trials,
            search_overrides=search_overrides,
            early_rejection_config=early_rejection_config,
        )
    elif search_strategy == "grid":
        return _run_grid_search(
            model=model, task_name=task_name, method=method,
            pairs_file=pairs_file, num_layers=num_layers, limit=limit,
            device=device, verbose=verbose,
            search_overrides=search_overrides,
            early_rejection_config=early_rejection_config,
        )
    raise ValueError(f"Unknown search strategy: {search_strategy}")


def _run_optuna_search(
    model: str, task_name: str, method: str,
    pairs_file: str, num_layers: int, limit: int,
    device: Optional[str], verbose: bool, n_trials: int,
    n_startup_trials: int, search_overrides: Dict[str, Any],
    early_rejection_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run Optuna TPE-based search for a method."""
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

    optimizer = UnifiedOptimizer(
        backend=OPTUNA_BACKEND_NAME, direction="maximize",
    )

    with tempfile.TemporaryDirectory() as work_dir:
        raw_objective = create_objective(
            method=method, model=model, task=task_name,
            num_layers=num_layers, limit=limit, device=device,
            work_dir=work_dir, enriched_pairs_file=pairs_file,
        )
        if early_rejection_config.get("enabled", False):
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
        "n_trials": n_trials, "search_strategy": "optuna",
    }


def _run_grid_search(
    model: str, task_name: str, method: str,
    pairs_file: str, num_layers: int, limit: int,
    device: Optional[str], verbose: bool,
    search_overrides: Dict[str, Any],
    early_rejection_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run exhaustive grid search for a method."""
    from wisent.core.utils.cli.optimize_steering.pipeline.pipeline import (
        run_pipeline, _build_config,
    )

    grid_axes = _build_grid_axes(search_overrides, num_layers, method)
    param_names = list(grid_axes.keys())
    combos = list(itertools.product(*grid_axes.values()))

    if verbose:
        print(f"   {method}: grid search over {len(combos)} combinations")

    best_score = SCORE_RANGE_MIN
    best_params = {}
    all_trials = []

    for combo_idx, combo in enumerate(combos):
        params = dict(zip(param_names, combo))
        try:
            config, strength = _build_config(method, params)
        except (ValueError, KeyError) as exc:
            logger.warning(f"   Skipping invalid config: {exc}")
            continue

        with tempfile.TemporaryDirectory() as work_dir:
            try:
                result = run_pipeline(
                    model=model, task=task_name, config=config,
                    work_dir=work_dir, strength=strength, limit=limit,
                    device=device, enriched_pairs_file=pairs_file,
                )
                score = result.score
            except Exception as exc:
                logger.warning(f"   Pipeline failed: {exc}")
                score = SCORE_RANGE_MIN

        all_trials.append({"params": params, "score": score})
        if score > best_score:
            best_score = score
            best_params = params
        if verbose:
            print(f"   [{combo_idx + COMBO_OFFSET}/{len(combos)}] "
                  f"score={score:.4f}")

    if verbose:
        print(f"   {method}: best={best_score:.4f}")

    return {
        "method": method, "best_score": best_score,
        "best_params": best_params, "all_trials": all_trials,
        "n_trials": len(combos), "search_strategy": "grid",
    }


def _build_grid_axes(
    overrides: Dict[str, Any], num_layers: int, method: str,
) -> Dict[str, List]:
    """Build grid axes from search overrides or defaults."""
    from wisent.core.control.steering_optimizer.types import (
        SteeringApplicationStrategy,
    )
    axes = {}

    if "search_layers" in overrides and overrides["search_layers"]:
        axes["layer"] = [
            int(x) for x in overrides["search_layers"].split(",")
        ]
    else:
        axes["layer"] = list(
            range(COMBO_OFFSET, num_layers + COMBO_OFFSET),
        )

    if "search_strengths" in overrides and overrides["search_strengths"]:
        axes["strength"] = [
            float(x) for x in overrides["search_strengths"].split(",")
        ]
    else:
        import math
        from wisent.core.utils.config_tools.constants import (
            SP_STRENGTH_MU, SP_STRENGTH_SIGMA,
        )
        center = math.exp(SP_STRENGTH_MU)
        axes["strength"] = [
            center * math.exp(-SP_STRENGTH_SIGMA),
            center,
            center * math.exp(SP_STRENGTH_SIGMA),
        ]

    if "search_strategies" in overrides and overrides["search_strategies"]:
        strats = overrides["search_strategies"].split(",")
        axes["steering_strategy"] = [s.strip() for s in strats]
    else:
        axes["steering_strategy"] = [
            s.value for s in SteeringApplicationStrategy
        ]

    method_upper = method.upper()
    if method_upper == "TECZA":
        _add_method_axes(axes, overrides, (
            "search_num_directions:num_directions",
            "search_direction_weighting:direction_weighting",
            "search_retain_weight:retain_weight",
        ))
    elif method_upper in ("TETNO", "GROM"):
        _add_method_axes(axes, overrides, (
            "search_sensor_layer:sensor_layer",
            "search_steering_layers:steering_layers",
            "search_max_alpha:max_alpha",
            "search_threshold:condition_threshold",
            "search_gate_temp:gate_temperature",
            "search_gate_hidden:gate_hidden_dim",
            "search_intensity_hidden:intensity_hidden_dim",
            "search_behavior_weight:behavior_weight",
            "search_sparse_weight:sparse_weight",
        ))
    return axes


def _add_method_axes(
    axes: Dict, overrides: Dict, mappings: tuple,
) -> None:
    """Add method-specific grid axes from override mappings."""
    for mapping in mappings:
        override_key, axis_name = mapping.split(":")
        if override_key in overrides:
            axes[axis_name] = overrides[override_key]


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
