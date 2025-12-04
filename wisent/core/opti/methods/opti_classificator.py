from __future__ import annotations

from dataclasses import replace
from typing import Callable

import optuna

from wisent.core.opti.core.atoms import BaseOptimizer
from wisent.core.classifiers.classifiers.core.atoms import BaseClassifier, ClassifierTrainConfig 

__all__ = ["ClassificationOptimizer"]

class ClassificationOptimizer(BaseOptimizer):
    """
    Optuna optimizer for binary classifiers.

    arguments:
        make_classifier:
            callable that returns a new instance of a BaseClassifier subclass. This is important
            to ensure each trial gets a fresh model.
        X, Y:
            training data and binary labels (0/1).
        base_config:
            base training configuration; individual trials can override parameters.
        model_space:
            callable that takes an Optuna trial and returns a dictionary of model hyperparameters
            to pass to BaseClassifier.fit(..., **model_params), which in turn passes them to
            BaseClassifier.build_model(...).
        training_space:
            callable that takes an Optuna trial and returns a dictionary of training hyperparameters
            to pass to BaseClassifier.fit(..., **training_params). Supported keys are:
                num_epochs:
                    int, number of training epochs
                batch_size:
                    int, training batch size
                learning_rate:
                    float, learning rate for the optimizer
                monitor:
                    str, metric to monitor for early stopping and pruning
                optimizer:
                    torch.optim.Optimizer subclass or instance
                lr:
                    learning rate scheduler instance
                optimizer_kwargs:
                    dict, extra kwargs to pass to the optimizer constructor
                criterion:
                    loss function instance (subclass of torch.nn.modules.loss._Loss)
        objective_metric:
            str, metric to optimize (must be one of the metrics reported by the classifier).

    returns:
        HPORun with the study, best params, and best value.

    example:
        >>> from wisent.classifiers.models.logistic import LogisticClassifier
        >>> from wisent.classifiers.core.atoms import ClassifierTrainConfig
        >>> from wisent.opti.methods.opti_classificator import ClassificationOptimizer
        >>> import numpy as np
        >>> import torch
        >>> # Create some synthetic data
        >>> rng = np.random.default_rng(42)
        >>> X = rng.normal(size=(1000, 20)).astype(np.float32)
        >>> w = rng.normal(size=(20, 1)).astype(np.float32)
        >>> logits = X @ w + 0.1 * rng.normal(size=(1000, 1)).astype(np.float32)
        >>> Y = (logits > 0).astype(np.float32).squeeze()
        >>> # Define base training configuration
        >>> train_config = ClassifierTrainConfig(
        ...     test_size=0.2,
        ...     num_epochs=20,
        ...     batch_size=32,
        ...     learning_rate=1e-3,
        ...     monitor='accuracy',
        ...     random_state=42
        ... )
        >>> # Define model hyperparameter search space
        >>> def model_space(trial):
        ...     return {
        ...         "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64]),
        ...         "dropout": trial.suggest_float("dropout", 0.0, 0.5)
        ...     }
        >>> # Define training hyperparameter search space
        >>> def training_space(trial):
        ...     return {
        ...         "num_epochs": trial.suggest_int("num_epochs", 10, 50),
        ...         "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        ...         "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e-2),
        ...         "monitor": "accuracy"
        ...     }
        >>> # Create the optimizer
        >>> optimizer = ClassificationOptimizer(
        ...     make_classifier=lambda: LogisticClassifier(threshold=0.5, device='cpu'),
        ...     X=X,
        ...     Y=Y,
        ...     base_config=train_config,
        ...     model_space=model_space,
        ...     training_space=training_space,
        ...     objective_metric="accuracy"
        ... )
        >>> # Run optimization
        >>> result = optimizer.optimize(
        ...     HPOConfig(n_trials=10, direction="maximize", seed=42)
        ... )
        >>> print("Best params:", result.best_params)
        Best params: {'hidden_dim': 16, 'dropout': 0.123456, 'num_epochs': 30, 'batch_size': 32, 'learning_rate': 0.00123456}
        >>> print("Best accuracy:", result.best_value)
        Best accuracy: 0.92

    """

    name = "classification-optimizer"
    direction = "maximize"

    def __init__(
        self,
        make_classifier: Callable[[], BaseClassifier],
        X,
        Y,
        base_config: ClassifierTrainConfig,
        model_space: Callable[[optuna.Trial], dict],
        training_space: Callable[[optuna.Trial], dict] | None = None,
        objective_metric: str = "accuracy",
    ) -> None:
        self._make_classifier = make_classifier
        self._X = X
        self._Y = Y
        self._cfg0 = base_config
        self._model_space = model_space
        self._training_space = training_space or (lambda trial: {})
        self._metric = objective_metric


    def _objective(self, trial: optuna.Trial) -> float:
        """
        One trial: build model, train, and return the objective metric.
        This is called by the parent class 'optimize(...)'.

        arguments:
            trial: Optuna trial object.

        returns:
            float, value of the objective metric to optimize.
        """
        mparams = self._model_space(trial)
        tparams = self._training_space(trial)

        cfg = replace(
            self._cfg0,
            num_epochs=tparams.get("num_epochs", self._cfg0.num_epochs),
            batch_size=tparams.get("batch_size", self._cfg0.batch_size),
            learning_rate=tparams.get("learning_rate", self._cfg0.learning_rate),
            monitor=tparams.get("monitor", self._cfg0.monitor),
        )

        clf = self._make_classifier()

        def on_epoch_end(epoch: int, metrics: dict[str, float]) -> None:
            val = float(metrics.get(self._metric, metrics.get("accuracy", 0.0)))
            BaseOptimizer.report_and_maybe_prune(trial, val, step=epoch)

        report = clf.fit(
            self._X, self._Y,
            config=cfg,
            optimizer=tparams.get("optimizer"),
            lr=tparams.get("lr"),
            optimizer_kwargs=tparams.get("optimizer_kwargs"),
            criterion=tparams.get("criterion"),
            on_epoch_end=on_epoch_end,
            **mparams,  # -> build_model(...)
        )

        final = getattr(report.final, self._metric)
        return float(final)
