
from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import optuna

from core.atoms import BaseObjective

MetricName = Literal["accuracy", "precision", "recall", "f1", "auc"]


class ClassificationObjective(BaseObjective):
    name = "classification"

    def __init__(
        self,
        make_classifier: Callable[[], Any],
        X: Any,
        y: Any,
        metric: MetricName = "f1",
        param_space: Optional[Callable[[optuna.Trial], dict[str, Any]]] = None,
        base_config: Optional[Any] = None,
        direction: str = "maximize",
    ) -> None:
        self._factory = make_classifier
        self.X = X
        self.y = y
        self.metric: MetricName = metric
        self.param_space = param_space
        self.base_config = base_config
        self.direction = direction

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        if self.param_space is None:
            params = {
                "train__learning_rate": trial.suggest_float("train__learning_rate", 1e-5, 1e-1, log=True),
                "threshold": trial.suggest_float("threshold", 0.1, 0.9),
            }
        else:
            params = dict(self.param_space(trial))
        return params

    def evaluate(self, trial: optuna.Trial, params: dict[str, Any]) -> float:
        clf = self._factory()

        train_overrides = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("train__")}
        model_params = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("model__")}
        threshold = params.get("threshold", None)

        cfg = self.base_config
        if cfg is not None:
            try:
                cfg = type(cfg)(**{**cfg.__dict__, **train_overrides})
            except Exception:
                cfg = self.base_config
                for k, v in train_overrides.items():
                    setattr(cfg, k, v)
        else:
            class _Dummy:
                pass
            cfg = _Dummy()
            for k, v in train_overrides.items():
                setattr(cfg, k, v)

        if threshold is not None and hasattr(clf, "set_threshold"):
            clf.set_threshold(float(threshold))

        report = clf.fit(self.X, self.y, config=cfg, **model_params)

        hist = getattr(report, "history", None)
        if isinstance(hist, dict) and self.metric in hist:
            values = hist[self.metric]
            for i, v in enumerate(values, start=1):
                trial.report(float(v), step=i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        final = getattr(report, "final", None)
        score = float(getattr(final, self.metric))
        return score