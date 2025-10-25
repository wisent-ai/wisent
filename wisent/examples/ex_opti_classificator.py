# An example of using Optuna to optimize hyperparameters of a simple classifier.

# First we need to load necessary modules
from wisent.opti.core.atoms import HPOConfig
from wisent.opti.methods.opti_classificator import ClassificationOptimizer
from wisent.classifiers.models.mlp import MLPClassifier
from wisent.classifiers.core.atoms import ClassifierTrainConfig

import numpy as np
import matplotlib.pyplot as plt

import torch

# Create some synthetic nonlinear data
def make_nonlinear_data(n=1000, noise=0.2, seed=24):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:, 0]**2 + X[:, 1]**2) + rng.normal(scale=noise, size=n) > 0.5).astype(int)
    return X.astype(np.float32), y

# Generate data
X, y = make_nonlinear_data(n=1000, noise=0.2, seed=24)

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=20)
plt.title('Synthetic Nonlinear Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class')
plt.show()

# define search spaces for model and training hyperparameters
import optuna

def model_space(trial: optuna.Trial) -> dict:
    hidden_dim = trial.suggest_int("hidden_dim", 8, 128, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    return {"hidden_dim": hidden_dim, "dropout": dropout}

# define search space for training hyperparameters
def training_space(trial: optuna.Trial) -> dict:
    opt_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    params = {
        "optimizer": opt_name,
        "lr": lr,
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "num_epochs": trial.suggest_int("num_epochs", 10, 40),
        "criterion": "bce",
        "monitor": "accuracy",
    }
    if opt_name == "SGD":
        params["optimizer_kwargs"] = {"momentum": trial.suggest_float("momentum", 0.0, 0.95)}
    return params

# configure base training config
base_cfg = ClassifierTrainConfig(
    test_size=0.2,
    num_epochs=25,
    batch_size=64,
    learning_rate=1e-3,
    monitor="accuracy",
)

# build the optimizer
tuner = ClassificationOptimizer(
    make_classifier=lambda: MLPClassifier(device="cuda" if torch.cuda.is_available() else "cpu"),
    X=X, Y=y,
    base_config=base_cfg,
    model_space=model_space,
    training_space=training_space,
    objective_metric="accuracy",
)

# Optuna study config: sampler + pruner + direction ("maximize" for accuracy).
# create_study lets you set direction, sampler, pruner, and persistent storage if you want.

hpo_cfg = HPOConfig(
    n_trials=40,
    direction="maximize",
    sampler="tpe",
    pruner="median",
    timeout=None,
    seed=42,
)

run = tuner.optimize(hpo_cfg)
print("BEST VALUE:", run.best_value)
print("BEST PARAMS:", run.best_params)

# retrain a final model with best params ------------------
# split params back into model vs training bits:
def best_to_layers(params: dict) -> dict:
    return {k: v for k, v in params.items() if k in {"hidden_dim", "dropout"}}

best = run.best_params

final_cfg = ClassifierTrainConfig(
    test_size=0.2,
    num_epochs=50,
    batch_size=best.get("batch_size", base_cfg.batch_size),
    learning_rate=best.get("lr", base_cfg.learning_rate),
    monitor="accuracy",
)

clf = MLPClassifier()
report = clf.fit(
    X, y,
    config=final_cfg,
    optimizer=best.get("optimizer"),
    lr=float(best.get("lr", final_cfg.learning_rate)),
    optimizer_kwargs=best.get("optimizer_kwargs"),
    criterion="bce",
    layers=best_to_layers(best),
)
print("FINAL:", report.final)

# plot decision boundary for the final model

def plot_decision_boundary(clf: MLPClassifier, X: np.ndarray, y: np.ndarray, title: str):
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=20)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    plt.show()

plot_decision_boundary(clf, X, y, "Decision Boundary after HPO")
