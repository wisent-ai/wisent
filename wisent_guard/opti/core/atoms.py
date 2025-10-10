
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import optuna

__all__ = [
    "HPOConfig",
    "HPORun",
    "BaseObjective",
    "HPORunner",
]

Direction = Literal["maximize", "minimize"]


@dataclass
class HPOConfig:
    """
    Configuration for hyperparameter optimization (HPO) using Optuna.

    attributes:
        n_trials:
            number of trials to run (ignored if timeout is set and reached).
        direction:
            maximize or minimize (can be overridden per objective).
        sampler:
            one of {"tpe", "random", "cmaes"} or None for Optuna default.
        pruner:
            one of {"nop", "median", "sha", None}.
        timeout:
            optional global seconds budget.
        study_name:
            optional persistent study name.
        seed:
            sampler seed for reproducibility.
    """
    n_trials: int = 100
    direction: Direction = "maximize"
    sampler: str | None = "tpe"
    pruner: str | None = "asha"
    timeout: int | None = None
    storage: str | None = None
    seed: int | None = 42


@dataclass
class HPORun:
    """
    Result of an HPO run.
    
    attributes:
        study:
            the Optuna study object.
        best_params:
            the best hyperparameters found.
        best_value:
            the best objective value achieved.
    """
    study: optuna.Study
    best_params: dict[str, Any]
    best_value: float


class BaseObjective(ABC):
    """
    Abstract base class for defining an HPO objective.

    attributes:
        name:
            name of the objective.
        direction:
            "maximize" or "minimize".
    """

    name: str = "base-objective"
    direction: Direction = "maximize"

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """
        Suggest hyperparameters for the given trial.
        Override this method to define the search space.
        
        arguments:
            trial:
                the Optuna trial object.
        returns:
            a dictionary of hyperparameter names and their suggested values.
        """
        return {}

    @abstractmethod
    def evaluate(self, trial: optuna.Trial, params: dict[str, Any]) -> float:
        """
        Evaluate the objective function with the given hyperparameters.

        arguments:
            trial:
                the Optuna trial object.
            params:
                the hyperparameters to evaluate.
        returns:
            the objective value to be optimized.
        """
        raise NotImplementedError


class HPORunner:
    """
    Runner for hyperparameter optimization using Optuna.
    """

    def _make_sampler(self, cfg: HPOConfig) -> optuna.samplers.BaseSampler | None:
        """
        Create an Optuna sampler based on the configuration.
        
        arguments:
            cfg:
                the HPO configuration.
        returns:
            an Optuna sampler instance or None.
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
        raise ValueError(f"Unknown sampler: {cfg.sampler}")

    def _make_pruner(self, cfg: HPOConfig) -> optuna.pruners.BasePruner | None:
        """
        Create an Optuna pruner based on the configuration.

        arguments:
            cfg:
                the HPO configuration.
        returns:
            an Optuna pruner instance or None.
        
        raises:
            ValueError: if an unknown pruner type is specified.
        """
        if cfg.pruner is None:
            return None
        p = cfg.pruner.lower()
        if p == "nop":
            return optuna.pruners.NopPruner()
        if p == "sha":
            return optuna.pruners.SuccessiveHalvingPruner()
        if p == "median":
            return optuna.pruners.MedianPruner()
        raise ValueError(f"Unknown pruner: {cfg.pruner}")

    def optimize(self, objective: BaseObjective, cfg: HPOConfig) -> HPORun:
        """
        Optimize the given objective using the specified configuration.
        arguments:
            objective:
                the objective to optimize.
            cfg:
                the HPO configuration.
        returns:
            an HPORun containing the results of the optimization.
        """
        sampler = self._make_sampler(cfg)
        pruner = self._make_pruner(cfg)
        direction: Direction = getattr(objective, "direction", cfg.direction)

        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=cfg.storage,
            study_name=cfg.study_name,
            load_if_exists=bool(cfg.storage and cfg.study_name),
        )

        def _opt_fn(trial: optuna.Trial) -> float:
            """
            Objective function for optimization.
            """
            params = objective.suggest(trial)
            return objective.evaluate(trial, params)

        study.optimize(_opt_fn, n_trials=cfg.n_trials, timeout=cfg.timeout)
        return HPORun(study=study, best_params=study.best_params, best_value=study.best_value)