from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import optuna

from wisent.core.utils.infra_tools.errors import UnknownTypeError
from wisent.core.utils.config_tools.constants import (
    BASE_OPTIMIZER_NAME,
    HYPEROPT_BACKEND_NAME,
    OPTUNA_BACKEND_NAME,
)
from wisent.core.utils.services.optimization.core.parameters import (
    OptimizationRun,
    Param,
)

__all__ = [
    "Direction",
    "HPOConfig",
    "HPORun",
    "BaseOptimizer",
]

Direction = Literal["maximize", "minimize"]


@dataclass(slots=True, frozen=True)
class HPOConfig:
    """
    Configuration for hyperparameter optimization (HPO).

    attributes:
        n_trials:
            number of trials to run.
        direction:
            global default direction ("maximize" or "minimize").
        backend:
            optimizer backend ("optuna" or "hyperopt").
        sampler:
            one of {"tpe", "random", "cmaes"} or None for Optuna default.
        pruner:
            one of {"nop", "median", "sha", "asha", "hyperband"} or None for default.
        study_name:
            optional persistent study name.
        storage:
            Optuna storage URL (e.g., sqlite:///file.db) for persistence.
        seed:
            sampler seed for reproducibility.
        load_if_exists:
            reuse persisted study if it already exists (when storage+study_name set).
    """
    n_trials: int = None
    direction: Direction = "maximize"
    backend: str = OPTUNA_BACKEND_NAME
    sampler: str | None = "tpe"
    pruner: str | None = "asha"
    storage: str | None = None
    study_name: str | None = None
    seed: int | None = None
    load_if_exists: bool = True


@dataclass(slots=True, frozen=True)
class HPORun:
    """Result of an HPO run."""
    study: optuna.Study
    best_params: dict[str, Any]
    best_value: float


class BaseOptimizer:
    """
    Base class for task-agnostic optimizers.

    Supports two usage patterns:
    1. OOP: subclass and implement '_objective(trial)' with trial.suggest_*()
    2. Functional: call 'optimize_fn()' with an objective_fn + Param space dict
    """

    name: str = BASE_OPTIMIZER_NAME
    direction: Direction = "maximize"

    def optimize(self, cfg: HPOConfig) -> HPORun:
        """Run optimization using the OOP subclass pattern."""
        if cfg.n_trials is None:
            raise ValueError("n_trials is required in HPOConfig")
        sampler = self._make_sampler(cfg)
        pruner = self._make_pruner(cfg)
        direction: Direction = getattr(self, "direction", cfg.direction)

        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=cfg.storage,
            study_name=cfg.study_name,
            load_if_exists=bool(cfg.storage and cfg.study_name and cfg.load_if_exists),
        )

        study.optimize(self._objective, n_trials=cfg.n_trials, show_progress_bar=False)
        return HPORun(study=study, best_params=study.best_params, best_value=study.best_value)

    def optimize_fn(
        self,
        objective_fn: Callable[[dict[str, Any]], float],
        space: dict[str, Param],
        n_trials: int,
        cfg: HPOConfig | None = None,
        model: str | None = None,
        benchmark: str | None = None,
        method: str | None = None,
    ) -> OptimizationRun:
        """Run optimization using a functional objective + Param space.

        arguments:
            objective_fn: callable that takes a dict of param values and returns a score.
            space: dict mapping param names to Param objects.
            n_trials: number of trials to run.
            cfg: optional HPOConfig for backend/sampler/pruner/storage settings.
            model: model name for study persistence on HF.
            benchmark: benchmark name for study persistence on HF.
            method: method name for study persistence on HF.
        """
        if cfg is None:
            cfg = HPOConfig(n_trials=n_trials, direction=self.direction)
        direction = getattr(self, "direction", cfg.direction)

        from wisent.core.utils.services.optimization.core._backend_converters import (
            run_hyperopt,
            run_optuna_functional,
        )

        if cfg.backend == HYPEROPT_BACKEND_NAME:
            return run_hyperopt(
                objective_fn, space, n_trials, direction, cfg.seed,
                model=model, benchmark=benchmark, method=method,
            )
        if cfg.backend == OPTUNA_BACKEND_NAME:
            sampler = self._make_sampler(cfg)
            pruner = self._make_pruner(cfg)
            return run_optuna_functional(
                objective_fn, space, n_trials, direction,
                sampler, pruner, cfg.storage, cfg.study_name, cfg.load_if_exists,
                model=model, benchmark=benchmark, method=method,
            )
        raise ValueError(f"Unknown backend: {cfg.backend}")

    def _objective(self, trial: optuna.Trial) -> float:
        """Implement one trial; return objective value. Override in subclasses."""
        raise NotImplementedError

    def _make_sampler(self, cfg: HPOConfig) -> optuna.samplers.BaseSampler | None:
        """Create an Optuna sampler based on the config."""
        if cfg.sampler is None:
            return None
        s = cfg.sampler.lower()
        if s == "tpe":
            return optuna.samplers.TPESampler(seed=cfg.seed)
        if s == "random":
            return optuna.samplers.RandomSampler(seed=cfg.seed)
        if s == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=cfg.seed)
        raise UnknownTypeError(entity_type="sampler", value=cfg.sampler, valid_values=["tpe", "random", "cmaes"])

    def _make_pruner(self, cfg: HPOConfig) -> optuna.pruners.BasePruner | None:
        """Create an Optuna pruner based on the config."""
        if cfg.pruner is None:
            return None
        p = cfg.pruner.lower()
        if p == "nop":
            return optuna.pruners.NopPruner()
        if p in {"sha", "asha"}:
            return optuna.pruners.SuccessiveHalvingPruner()
        if p == "median":
            return optuna.pruners.MedianPruner()
        if p == "hyperband":
            return optuna.pruners.HyperbandPruner()
        raise UnknownTypeError(entity_type="pruner", value=cfg.pruner, valid_values=["nop", "sha", "asha", "median", "hyperband"])

    @staticmethod
    def report_and_maybe_prune(trial: optuna.Trial, value: float, step: int) -> None:
        """Report an intermediate metric and prune if the pruner suggests it."""
        trial.report(float(value), step=step)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
