
import numpy as np

from wisent_guard.opti.core.atoms import HPOConfig, HPORunner
from wisent_guard.opti.methods.opti_classificator import ClassificationObjective
from wisent_guard.classifiers.core.atoms import ClassifierTrainConfig
from wisent_guard.classifiers.models.mlp import MLPClassifier


def make_xor_blobs(n: int = 1200, noise: float = 0.35, seed: int = 13):
    """
    Nonlinear 2-class data (XOR-style) + two derived features.
    Learnable by a small MLP; tricky for a pure linear model.

    atributes:
        n: total number of samples (will be rounded to multiple of 4)
        noise: Gaussian noise level
        seed: random seed
    returns:
        X: (n, 4) float32 array
        y: (n,) int array with 0/1 labels
    """
    rng = np.random.default_rng(seed)
    n4 = n // 4
    centers = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float)
    X = np.vstack([centers[i] + rng.normal(scale=noise, size=(n4, 2)) for i in range(4)])
    y = np.array([int((cx * cy) < 0) for cx, cy in X], dtype=int)
    f3 = X[:, 0] * X[:, 1] + rng.normal(scale=noise, size=X.shape[0])
    f4 = (X ** 2).sum(axis=1) + rng.normal(scale=noise, size=X.shape[0])
    X = np.column_stack([X, f3, f4]).astype(np.float32)
    return X, y




if __name__ == "__main__":
    # Data
    X, y = make_xor_blobs(n=1200, noise=0.35, seed=13)

    # plot data
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=20)
    plt.title("XOR Blobs")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Class")
    plt.show()

    default_config = ClassifierTrainConfig(
        test_size=0.2,
        num_epochs=30,
        batch_size=32,
        learning_rate=1e-3,
        early_stopping_patience=5,
        random_state=42,
    )

    cls = MLPClassifier() 
    obj = ClassificationObjective(
        clf=MLPClassifier(),   # returns a fresh classifier each trial
        X=X, y=y, metric="f1",
        param_space=lambda t: {
            "train__learning_rate": t.suggest_float("train__learning_rate", 1e-4, 5e-2, log=True),
            "train__batch_size": t.suggest_categorical("train__batch_size", [16, 32, 64]),
            "model__hidden_dim": t.suggest_int("model__hidden_dim", 16, 256, log=True),
            "threshold": t.suggest_float("threshold", 0.2, 0.8),
        },
        base_config=default_config,
    )

    run = HPORunner().optimize(obj, HPOConfig(n_trials=60, pruner="median", sampler="tpe"))
    print("Best params:", run.best_params)
    print("Best f1:", run.best_value)