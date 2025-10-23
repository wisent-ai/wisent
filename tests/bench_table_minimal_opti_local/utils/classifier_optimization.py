"""
Classifier Optimization Utilities using Optuna.

Provides hyperparameter optimization for each configuration tuple:
(layer, aggregation_method, prompt_strategy, threshold, classifier_type)

Uses existing wisent classifiers and follows bench_table_rewrite patterns.
"""

from __future__ import annotations

import sys
import os
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import optuna
import numpy as np
from typing import Tuple, Dict, List

from wisent.opti.methods.opti_classificator import ClassificationOptimizer
from wisent.opti.core.atoms import HPOConfig
from wisent.classifiers.models.mlp import MLPClassifier
from wisent.classifiers.models.logistic import LogisticClassifier
from wisent.classifiers.core.atoms import ClassifierTrainConfig


def prepare_training_data(positive_activations: list, negative_activations: list):
    """Prepare training data from positive and negative activations.

    Args:
        positive_activations: List of positive activation tensors
        negative_activations: List of negative activation tensors

    Returns:
        (X, y) tensors
    """
    X_pos = torch.stack(positive_activations)
    y_pos = torch.ones(len(positive_activations))

    X_neg = torch.stack(negative_activations)
    y_neg = torch.zeros(len(negative_activations))

    X = torch.cat([X_pos, X_neg], dim=0)
    y = torch.cat([y_pos, y_neg], dim=0)

    return X, y


def mlp_model_space(trial: optuna.Trial) -> dict:
    """Define search space for MLP model hyperparameters."""
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    return {"hidden_dim": hidden_dim, "dropout": dropout}


def logistic_model_space(trial: optuna.Trial) -> dict:
    """Define search space for logistic regression (no model hyperparameters)."""
    return {}


def training_space(trial: optuna.Trial) -> dict:
    """Define search space for training hyperparameters.

    Fixed batch sizes to [16, 32, 64] since we always use plenty of training data.
    """
    opt_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    params = {
        "optimizer": opt_name,
        "lr": lr,
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "num_epochs": trial.suggest_int("num_epochs", 20, 100),
        "criterion": "bce",
        "monitor": "accuracy",
    }
    if opt_name == "SGD":
        params["optimizer_kwargs"] = {"momentum": trial.suggest_float("momentum", 0.0, 0.95)}
    return params


def optimize_classifier_config(
    train_val_pos: List[torch.Tensor],
    train_val_neg: List[torch.Tensor],
    num_train: int,
    classifier_type: str,
    threshold: float,
    n_trials: int = 40,
) -> Tuple[Dict, float]:
    """
    Run Optuna optimization for a specific classifier configuration.

    This function is called inside the loop over (layer, aggregation, prompt_strategy, threshold, classifier_type).

    Args:
        train_val_pos: Combined train+val positive activations (list of tensors)
        train_val_neg: Combined train+val negative activations (list of tensors)
        num_train: Number of training samples (rest are validation)
        classifier_type: "logistic" or "mlp"
        threshold: Classification threshold (passed to classifier __init__)
        n_trials: Number of Optuna trials

    Returns:
        (best_params, best_val_accuracy)
    """
    # Prepare training+validation data for HPO
    X_train_val, y_train_val = prepare_training_data(train_val_pos, train_val_neg)

    num_val = len(train_val_pos) - num_train
    val_size = num_val / len(train_val_pos)

    # Create classifier factory with threshold
    if classifier_type == "mlp":
        make_classifier = lambda: MLPClassifier(
            threshold=threshold,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        model_space = mlp_model_space
    elif classifier_type == "logistic":
        make_classifier = lambda: LogisticClassifier(
            threshold=threshold,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        model_space = logistic_model_space
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    # Base configuration
    base_cfg = ClassifierTrainConfig(
        test_size=val_size,
        num_epochs=50,
        batch_size=32,
        learning_rate=1e-3,
        monitor="accuracy",
        random_state=42,
    )

    # Create optimizer
    tuner = ClassificationOptimizer(
        make_classifier=make_classifier,
        X=X_train_val,
        Y=y_train_val,
        base_config=base_cfg,
        model_space=model_space,
        training_space=training_space,
        objective_metric="accuracy",
    )

    # Run HPO
    hpo_cfg = HPOConfig(
        n_trials=n_trials,
        direction="maximize",
        sampler="tpe",
        pruner="median",
        timeout=None,
        seed=42,
    )

    run = tuner.optimize(hpo_cfg)

    return run.best_params, run.best_value


def evaluate_on_test_set(
    train_val_pos: List[torch.Tensor],
    train_val_neg: List[torch.Tensor],
    test_pos: List[torch.Tensor],
    test_neg: List[torch.Tensor],
    num_train: int,
    classifier_type: str,
    best_params: Dict,
    threshold: float,
    n_runs: int = 10,
) -> Tuple[float, float, List[float]]:
    """
    Evaluate best classifier configuration on test set with multiple runs.

    Args:
        train_val_pos: Combined train+val positive activations
        train_val_neg: Combined train+val negative activations
        test_pos: Test positive activations
        test_neg: Test negative activations
        num_train: Number of training samples (rest are validation, but we use all for final training)
        classifier_type: "logistic" or "mlp"
        best_params: Best hyperparameters from Optuna
        threshold: Classification threshold (passed to classifier __init__)
        n_runs: Number of runs to average over

    Returns:
        (mean_accuracy, std_accuracy, all_accuracies)
    """
    # Use all train+val data for final training
    X_train, y_train = prepare_training_data(train_val_pos, train_val_neg)
    X_test, y_test = prepare_training_data(test_pos, test_neg)

    # Extract model kwargs
    model_kwargs = {k: v for k, v in best_params.items() if k in ["hidden_dim", "dropout"]}

    test_accuracies = []

    for run_idx in range(n_runs):
        seed = 42 + run_idx

        # Create run configuration
        run_cfg = ClassifierTrainConfig(
            test_size=0.0,  # No validation split for final training
            num_epochs=best_params.get("num_epochs", 50),
            batch_size=best_params.get("batch_size", 32),
            learning_rate=best_params.get("lr", 1e-3),
            monitor="accuracy",
            random_state=seed,
        )

        # Create classifier with threshold
        if classifier_type == "mlp":
            clf = MLPClassifier(
                threshold=threshold,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        elif classifier_type == "logistic":
            clf = LogisticClassifier(
                threshold=threshold,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        # Train
        clf.fit(
            X_train, y_train,
            config=run_cfg,
            optimizer=best_params.get("optimizer", "Adam"),
            lr=best_params.get("lr", 1e-3),
            optimizer_kwargs=best_params.get("optimizer_kwargs"),
            criterion="bce",
            layers=model_kwargs,
        )

        # Evaluate on test set - predict() already applies threshold internally
        predictions = clf.predict(X_test)
        if isinstance(predictions, list):
            predictions = torch.tensor(predictions).float()
        elif isinstance(predictions, int):
            predictions = torch.tensor([predictions]).float()

        accuracy = (predictions == y_test).float().mean().item()
        test_accuracies.append(accuracy)

    mean_accuracy = np.mean(test_accuracies)
    std_accuracy = np.std(test_accuracies)

    return mean_accuracy, std_accuracy, test_accuracies
