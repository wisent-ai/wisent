from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import optuna

from wisent.core.errors import UnknownTypeError

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
    Configuration for hyperparameter optimization (HPO) using Optuna.

    attributes:
        n_trials:
            number of trials (ignored if timeout is reached).
        direction:
            global default direction ("maximize" or "minimize").
        sampler:
            one of {"tpe", "random", "cmaes"} or None for Optuna default.
        pruner:
            one of {"nop", "median", "sha", "asha", "hyperband"} or None for default.
        timeout:
            optional global seconds budget.
        study_name:
            optional persistent study name.
        storage:
            Optuna storage URL (e.g., sqlite:///file.db) for persistence.
        seed:
            sampler seed for reproducibility.
        load_if_exists:
            reuse persisted study if it already exists (when storage+study_name set).
    """
    n_trials: int = 100
    direction: Direction = "maximize"
    sampler: str | None = "tpe"
    pruner: str | None = "asha"
    timeout: int | None = None
    storage: str | None = None
    study_name: str | None = None
    seed: int | None = 42
    load_if_exists: bool = True


@dataclass(slots=True, frozen=True)
class HPORun:
    """
    Result of an HPO run.
    """
    study: optuna.Study
    best_params: dict[str, Any]
    best_value: float


class BaseOptimizer(ABC):
    """
    Base class for building task-agnostic Optuna optimizers.

    Subclasses must implement '_objective(trial)' and return a float objective.
    This class wires up samplers/pruners and runs 'study.optimize(...)'.
    """

    name: str = "base-optimizer"
    direction: Direction = "maximize"

    def optimize(self, cfg: HPOConfig) -> HPORun:
        """
        Run the optimization process.

        arguments:
            cfg: 
                HPOConfig object with optimization settings.

        returns:
            HPORun object with the results of the optimization.
        """
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

        study.optimize(self._objective, n_trials=cfg.n_trials, timeout=cfg.timeout, show_progress_bar=False)
        return HPORun(study=study, best_params=study.best_params, best_value=study.best_value)

    @abstractmethod
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Implement one trial; return objective value.
        """
        raise NotImplementedError

    def _make_sampler(self, cfg: HPOConfig) -> optuna.samplers.BaseSampler | None:
        """
        Create an Optuna sampler based on the config.
        
        arguments:
            cfg: HPOConfig object.
            
        returns:
            An Optuna sampler instance or None for default.

        raises:
            ValueError if the sampler name is unknown.
        """
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
        """
        Create an Optuna pruner based on the config.

        arguments:
            cfg: HPOConfig object.

        returns:
            An Optuna pruner instance or None for default.
        
        raises:
            ValueError if the pruner name is unknown.
        """
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
        """
        Report an intermediate metric and prune if the pruner suggests it.

        arguments:
            trial:
                Optuna trial object.
            value:
                Metric value to report.
            step:
                Step number (e.g., epoch).
        """
        trial.report(float(value), step=step)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
