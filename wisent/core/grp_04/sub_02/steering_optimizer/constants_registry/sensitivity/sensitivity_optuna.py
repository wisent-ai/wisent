"""Joint Optuna optimization of the top-N most sensitive constants.

Takes the output of SensitivityEngine (which includes the operating
point found during phase 1) and jointly optimizes the most impactful
constants using Optuna's TPE sampler. Each trial evaluates at the
fixed operating point — no nested search.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import optuna

from wisent.core.constants import DEFAULT_N_TRIALS, DEFAULT_SCORE, SENSITIVITY_OPTUNA_TOP_N

logger = logging.getLogger(__name__)


class OptunaConstantOptimizer:
    """Joint optimization of multiple constants via Optuna.

    Requires a fixed_config (OptimizationConfig) that defines the
    single operating point at which to evaluate. This comes from the
    SensitivityResult.operating_point found during phase 1.
    """

    def __init__(
        self,
        model,
        method_name: str,
        task_name: str,
        train_pairs,
        test_pairs,
        evaluator,
        fixed_config,
        verbose: bool = True,
    ):
        self.model = model
        self.method_name = method_name.lower()
        self.task_name = task_name
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs
        self.evaluator = evaluator
        self.fixed_config = fixed_config
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _create_objective(self, constant_metas: List):
        """Create an Optuna objective that evaluates at fixed config."""
        from ..registry import ConstantPatcher

        def objective(trial: optuna.Trial) -> float:
            overrides = {}
            for meta in constant_metas:
                if meta.log_scale and meta.low > 0:
                    value = trial.suggest_float(
                        meta.name, meta.low, meta.high, log=True,
                    )
                elif meta.dtype == "int":
                    value = float(trial.suggest_int(
                        meta.name, int(meta.low), int(meta.high),
                    ))
                else:
                    value = trial.suggest_float(
                        meta.name, meta.low, meta.high,
                    )
                overrides[meta.name] = value

            from wisent.core.cli.optimization.core.method_optimizer import (
                MethodOptimizer,
            )
            with ConstantPatcher(overrides):
                optimizer = MethodOptimizer(
                    model=self.model,
                    method_name=self.method_name,
                    verbose=False,
                )
                summary = optimizer.optimize(
                    train_pairs=self.train_pairs,
                    test_pairs=self.test_pairs,
                    evaluator=self.evaluator,
                    task_name=self.task_name,
                    configs=[self.fixed_config],
                )
                if summary.best_result is not None:
                    return summary.best_result.score
                return DEFAULT_SCORE

        return objective

    def optimize_from_sensitivity(
        self,
        sensitivity_result,
        top_n: int = SENSITIVITY_OPTUNA_TOP_N,
        n_trials: int = DEFAULT_N_TRIALS,
        threshold: float = DEFAULT_SCORE,
    ):
        """Optimize top-N most sensitive constants."""
        from ..registry import get_registry

        registry = get_registry()
        ranked = sensitivity_result.ranked()

        if threshold > DEFAULT_SCORE:
            ranked = [r for r in ranked if r.sensitivity >= threshold]

        targets = ranked[:top_n]
        if not targets:
            self._log("No constants above sensitivity threshold.")
            return None

        constant_metas = []
        for target in targets:
            if target.name in registry:
                constant_metas.append(registry[target.name])

        self._log(
            f"\nOptuna joint optimization: "
            f"{len(constant_metas)} constants, {n_trials} trials"
        )
        self._log(
            f"Fixed operating point: "
            f"layers={self.fixed_config.layers}, "
            f"strength={self.fixed_config.strength}"
        )
        for meta in constant_metas:
            self._log(
                f"  {meta.name}: [{meta.low}, {meta.high}]"
                f" (log={meta.log_scale})"
            )

        return self._run_study(constant_metas, n_trials)

    def optimize_constants(
        self,
        constant_names: List[str],
        n_trials: int = DEFAULT_N_TRIALS,
    ):
        """Optimize specific constants by name."""
        from ..registry import get_registry

        registry = get_registry()
        constant_metas = []
        for name in constant_names:
            if name in registry:
                constant_metas.append(registry[name])
            else:
                logger.warning(
                    "Constant %s not in registry, skipping", name,
                )

        if not constant_metas:
            self._log("No valid constants to optimize.")
            return None

        return self._run_study(constant_metas, n_trials)

    def _run_study(self, constant_metas: List, n_trials: int):
        """Run the Optuna study and return a ConstantProfile."""
        from ..profiles import ConstantProfile

        start_time = time.time()

        optuna.logging.set_verbosity(
            optuna.logging.INFO
            if self.verbose
            else optuna.logging.WARNING
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
        )

        objective = self._create_objective(constant_metas)
        study.optimize(objective, n_trials=n_trials)

        total_time = time.time() - start_time

        best_params = study.best_params
        best_score = study.best_value

        self._log(f"\nOptuna complete in {total_time:.1f}s")
        self._log(f"Best score: {best_score:.4f}")
        self._log("Best parameters:")
        for name, value in sorted(best_params.items()):
            self._log(f"  {name}: {value:.6g}")

        model_name = str(
            getattr(self.model, 'model_name', 'unknown'),
        )

        return ConstantProfile(
            model_name=model_name,
            task_name=self.task_name,
            constants={k: float(v) for k, v in best_params.items()},
            source="optuna",
            metrics={
                "best_score": best_score,
                "n_trials": float(n_trials),
                "optimization_time": total_time,
                "n_constants": float(len(constant_metas)),
            },
        )
