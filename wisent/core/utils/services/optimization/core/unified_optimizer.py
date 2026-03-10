"""Unified optimizer supporting Hyperopt and Optuna backends."""
from __future__ import annotations

import math
from typing import Any, Callable

from wisent.core.utils.config_tools.constants import (
    HYPEROPT_BACKEND_NAME,
    INDEX_FIRST,
    OPTUNA_BACKEND_NAME,
    OPTUNA_SIGMA_SPREAD_FACTOR,
    QUANTIZATION_STEP_DEFAULT,
)
from wisent.core.utils.services.optimization.core.parameters import (
    CategoricalParam,
    FloatParam,
    IntParam,
    OptimizationRun,
    Param,
)


class UnifiedOptimizer:
    """Optimizer supporting Hyperopt and Optuna backends.

    Hyperopt uses distribution-based parameters (hp.normal, hp.lognormal)
    which are unbounded - no arbitrary min/max ranges needed.

    Optuna derives finite bounds from the distribution parameters
    using a sigma spread factor.
    """

    def __init__(
        self,
        backend: str = HYPEROPT_BACKEND_NAME,
        direction: str = "maximize",
        seed: int | None = None,
    ):
        self.backend = backend
        self.direction = direction
        self.seed = seed

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any]], float],
        space: dict[str, Param],
        n_trials: int,
    ) -> OptimizationRun:
        """Run optimization with the configured backend."""
        if self.backend == HYPEROPT_BACKEND_NAME:
            return self._run_hyperopt(objective_fn, space, n_trials)
        if self.backend == OPTUNA_BACKEND_NAME:
            return self._run_optuna(objective_fn, space, n_trials)
        raise ValueError(f"Unknown backend: {self.backend}")

    def _run_hyperopt(
        self,
        objective_fn: Callable,
        space: dict[str, Param],
        n_trials: int,
    ) -> OptimizationRun:
        from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

        hp_space = {}
        for name, param in space.items():
            hp_space[name] = _param_to_hyperopt(name, param)

        maximize = self.direction == "maximize"
        trials = Trials()

        def wrapped(params):
            score = objective_fn(params)
            return {
                "loss": -score if maximize else score,
                "status": STATUS_OK,
            }

        rstate_kwargs = {}
        if self.seed is not None:
            import numpy as _np
            rstate_kwargs["rstate"] = _np.random.default_rng(self.seed)

        best_raw = fmin(
            fn=wrapped,
            space=hp_space,
            algo=tpe.suggest,
            max_evals=n_trials,
            trials=trials,
            show_progressbar=True,
            **rstate_kwargs,
        )

        all_trials = []
        best_score = None
        best_params = None
        for t in trials.trials:
            vals = t["misc"]["vals"]
            params = {k: v[INDEX_FIRST] for k, v in vals.items()}
            params = _resolve_categorical_indices(params, space)
            loss = t["result"]["loss"]
            score = -loss if maximize else loss
            all_trials.append({"params": params, "score": score})
            if best_score is None or (
                (maximize and score > best_score)
                or (not maximize and score < best_score)
            ):
                best_score = score
                best_params = params

        if best_params is None:
            best_params = _resolve_categorical_indices(
                best_raw, space,
            )
            best_loss = trials.best_trial["result"]["loss"]
            best_score = -best_loss if maximize else best_loss

        return OptimizationRun(
            best_params=best_params,
            best_score=best_score,
            all_trials=all_trials,
            n_trials=n_trials,
            backend=HYPEROPT_BACKEND_NAME,
        )

    def _run_optuna(
        self,
        objective_fn: Callable,
        space: dict[str, Param],
        n_trials: int,
    ) -> OptimizationRun:
        import optuna

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for name, param in space.items():
                params[name] = _param_to_optuna(trial, name, param)
            return objective_fn(params)

        study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(
                seed=self.seed,
            ),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        all_trials = [
            {"params": t.params, "score": t.value}
            for t in study.trials
            if t.value is not None
        ]

        return OptimizationRun(
            best_params=study.best_params,
            best_score=study.best_value,
            all_trials=all_trials,
            n_trials=n_trials,
            backend=OPTUNA_BACKEND_NAME,
        )


def _param_to_hyperopt(name: str, param: Param):
    """Map a Param dataclass to a hyperopt search expression."""
    from hyperopt import hp

    if isinstance(param, FloatParam):
        return _float_to_hyperopt(name, param)
    if isinstance(param, IntParam):
        return _int_to_hyperopt(name, param)
    if isinstance(param, CategoricalParam):
        return hp.choice(name, param.choices)
    raise TypeError(f"Unknown param type: {type(param)}")


def _float_to_hyperopt(name: str, p: FloatParam):
    from hyperopt import hp

    d = p.distribution
    if d == "normal":
        return hp.normal(name, p.mu, p.sigma)
    if d == "lognormal":
        return hp.lognormal(name, p.mu, p.sigma)
    if d == "uniform":
        return hp.uniform(name, p.low, p.high)
    raise ValueError(f"Unknown float distribution: {d}")


def _int_to_hyperopt(name: str, p: IntParam):
    from hyperopt import hp
    from hyperopt.pyll import scope

    d = p.distribution
    if d == "randint":
        return scope.int(
            hp.quniform(name, p.low, p.high, QUANTIZATION_STEP_DEFAULT),
        )
    if d == "qnormal":
        return scope.int(hp.qnormal(name, p.mu, p.sigma, p.q))
    if d == "qlognormal":
        return scope.int(hp.qlognormal(name, p.mu, p.sigma, p.q))
    raise ValueError(f"Unknown int distribution: {d}")


def _param_to_optuna(
    trial: "optuna.Trial", name: str, param: Param,
):
    """Map a Param dataclass to an Optuna trial suggestion."""
    if isinstance(param, FloatParam):
        return _float_to_optuna(trial, name, param)
    if isinstance(param, IntParam):
        return _int_to_optuna(trial, name, param)
    if isinstance(param, CategoricalParam):
        return trial.suggest_categorical(name, param.choices)
    raise TypeError(f"Unknown param type: {type(param)}")


def _float_to_optuna(trial, name: str, p: FloatParam):
    d = p.distribution
    if d == "uniform":
        return trial.suggest_float(name, p.low, p.high, log=p.log_scale)
    spread = OPTUNA_SIGMA_SPREAD_FACTOR
    if d == "normal":
        lo = p.mu - spread * p.sigma
        hi = p.mu + spread * p.sigma
        return trial.suggest_float(name, lo, hi)
    if d == "lognormal":
        lo = math.exp(p.mu - spread * p.sigma)
        hi = math.exp(p.mu + spread * p.sigma)
        return trial.suggest_float(name, lo, hi, log=True)
    raise ValueError(f"Unknown float distribution: {d}")


def _int_to_optuna(trial, name: str, p: IntParam):
    d = p.distribution
    if d == "randint":
        return trial.suggest_int(name, p.low, p.high)
    spread = OPTUNA_SIGMA_SPREAD_FACTOR
    if d == "qnormal":
        lo = max(INDEX_FIRST, int(p.mu - spread * p.sigma))
        hi = int(p.mu + spread * p.sigma)
        return trial.suggest_int(name, lo, hi, step=p.q)
    if d == "qlognormal":
        lo = max(
            QUANTIZATION_STEP_DEFAULT,
            int(math.exp(p.mu - spread * p.sigma)),
        )
        hi = int(math.exp(p.mu + spread * p.sigma))
        return trial.suggest_int(name, lo, hi, step=p.q)
    raise ValueError(f"Unknown int distribution: {d}")


def _resolve_categorical_indices(
    params: dict, space: dict[str, Param],
) -> dict:
    """Hyperopt returns indices for hp.choice; resolve to values."""
    resolved = {}
    for k, v in params.items():
        if k in space and isinstance(space[k], CategoricalParam):
            idx = int(v)
            choices = space[k].choices
            resolved[k] = choices[idx] if idx < len(choices) else v
        else:
            resolved[k] = v
    return resolved
