"""
Train classifiers on varying numbers of training examples (k = 5, 10, 20, 50, 100, 250)
Test on 150 examples
Plot layer vs accuracy curves with binomial error bars
"""

from __future__ import annotations

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List
import optuna
import time
import gc
import json
from datetime import datetime

from wisent_guard.opti.methods.opti_classificator import ClassificationOptimizer
from wisent_guard.opti.core.atoms import HPOConfig
from wisent_guard.classifiers.models.mlp import MLPClassifier
from wisent_guard.classifiers.core.atoms import ClassifierTrainConfig

from tests.bench_table.gsm8k.activation_matrix import load_pairs, create_activations_matrix
from wisent_guard.core.models.wisent_model import WisentModel
from wisent_guard.core.activations.core.activations_collector import ActivationCollector
from wisent_guard.core.activations.core.atoms import ActivationAggregationStrategy


def mlp_model_space(trial: optuna.Trial) -> dict:
    """Define search space for MLP model hyperparameters."""
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    return {"hidden_dim": hidden_dim, "dropout": dropout}


def training_space(trial: optuna.Trial) -> dict:
    """Define search space for training hyperparameters."""
    opt_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    params = {
        "optimizer": opt_name,
        "lr": lr,
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
        "num_epochs": trial.suggest_int("num_epochs", 20, 100),
        "criterion": "bce",
        "monitor": "accuracy",
    }
    if opt_name == "SGD":
        params["optimizer_kwargs"] = {"momentum": trial.suggest_float("momentum", 0.0, 0.95)}
    return params


def prepare_training_data(positive_activations: list, negative_activations: list):
    """Prepare training data from positive and negative activations."""
    X_pos = torch.stack(positive_activations)
    y_pos = torch.ones(len(positive_activations))

    X_neg = torch.stack(negative_activations)
    y_neg = torch.zeros(len(negative_activations))

    X = torch.cat([X_pos, X_neg], dim=0)
    y = torch.cat([y_pos, y_neg], dim=0)

    return X, y


def train_and_evaluate_for_k(
    k: int,
    num_val: int,
    num_test: int,
    model_name: str,
    aggregation_method: ActivationAggregationStrategy,
    n_trials: int = 40,
) -> tuple[Dict[str, float], Dict]:
    """
    Train classifiers with k training examples and evaluate on num_test examples.

    Training data comes from training docs, test data comes from test docs.

    Returns:
        Tuple of (layer_accuracies, layer_metadata)
    """

    print("=" * 80)
    print(f"TRAINING WITH k={k} examples, validating on {num_val}, testing on {num_test} examples")
    print("=" * 80)

    # Step 1: Create training+validation activations matrix from training docs
    print(f"\nCreating TRAINING+VALIDATION activations matrix for {k + num_val} pairs from training docs...")
    train_val_result = create_activations_matrix(
        model_name=model_name,
        aggregation_methods=[aggregation_method],
        num_questions=k + num_val,
        output_path=None,
        preferred_doc="training"
    )

    train_val_matrix = train_val_result["matrix"][aggregation_method.value]
    num_layers = train_val_result["summary"]["num_layers"]

    # Additional GPU cleanup between train+val and test activation collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(10)

    # Step 2: Create test activations matrix from test docs
    print(f"\nCreating TEST activations matrix for {num_test} pairs from test docs...")
    test_result = create_activations_matrix(
        model_name=model_name,
        aggregation_methods=[aggregation_method],
        num_questions=num_test,
        output_path=None,
        preferred_doc="test"
    )

    test_matrix = test_result["matrix"][aggregation_method.value]

    # Calculate validation split size
    # Total training data = k + num_val pairs = (k + num_val) * 2 activations
    # Validation size = num_val * 2 activations
    # test_size = (num_val * 2) / ((k + num_val) * 2) = num_val / (k + num_val)
    val_size = num_val / (k + num_val)

    print(f"\nData split:")
    print(f"  Training: {k} pairs ({k * 2} activations) from training docs")
    print(f"  Validation: {num_val} pairs ({num_val * 2} activations) from training docs")
    print(f"  Test: {num_test} pairs ({num_test * 2} activations) from test docs")
    print(f"  Validation split for HPO: {val_size*100:.1f}%")

    # Step 3: Train and evaluate classifier for each layer
    print(f"\nTraining and evaluating classifiers for each layer...")
    layer_accuracies: Dict[str, float] = {}
    layer_metadata: Dict[str, Dict] = {}  # Store detailed results per layer

    for layer_idx in range(1, num_layers + 1):
        layer_name = str(layer_idx)

        print(f"\n[{layer_idx}/{num_layers}] Layer {layer_name}")
        print("-" * 40)

        # Get training+validation activations for this layer
        train_val_pos = train_val_matrix[layer_name]["positive"]
        train_val_neg = train_val_matrix[layer_name]["negative"]

        # Get test activations for this layer
        test_pos = test_matrix[layer_name]["positive"]
        test_neg = test_matrix[layer_name]["negative"]

        if not train_val_pos or not train_val_neg or not test_pos or not test_neg:
            print(f"  SKIPPED: Missing data")
            layer_accuracies[layer_name] = 0.0
            continue

        # Prepare training+validation data (will be split internally by Optuna)
        X_train_val, y_train_val = prepare_training_data(train_val_pos, train_val_neg)

        print(f"  Training+Validation samples: {len(X_train_val)} (will use {(1-val_size)*100:.1f}% train, {val_size*100:.1f}% val for HPO)")
        print(f"  Test samples: {len(test_pos) + len(test_neg)}")

        # Set up base config and model space
        make_classifier = lambda: MLPClassifier(device="cuda" if torch.cuda.is_available() else "cpu")

        base_cfg = ClassifierTrainConfig(
            test_size=val_size,  # Split to use num_val for validation
            num_epochs=50,
            batch_size=8,
            learning_rate=1e-3,
            monitor="accuracy",
            random_state=42,  # Fixed seed for reproducible train/val split
        )

        # Run Optuna to find best hyperparameters
        try:
            tuner = ClassificationOptimizer(
                make_classifier=make_classifier,
                X=X_train_val,
                Y=y_train_val,
                base_config=base_cfg,
                model_space=mlp_model_space,
                training_space=training_space,
                objective_metric="accuracy",  # Optimize on validation split
            )

            # HPO configuration
            hpo_cfg = HPOConfig(
                n_trials=n_trials,
                direction="maximize",
                sampler="tpe",
                pruner="median",
                timeout=None,
                seed=42,
            )

            # Run optimization
            print(f"  Running Optuna optimization ({n_trials} trials with {val_size*100:.1f}% validation split)...")
            run = tuner.optimize(hpo_cfg)

            print(f"  Best validation accuracy: {run.best_value:.3f}")
            print(f"  Best params: {run.best_params}")

            # Step 4: Train final model 10 times on ONLY training data (first k pairs) with best params
            print(f"  Training final model 10 times on training data only (first {k} pairs)...")

            # Extract only training data (first k pairs)
            train_pos_only = train_val_pos[:k]
            train_neg_only = train_val_neg[:k]
            X_train, y_train = prepare_training_data(train_pos_only, train_neg_only)

            # Prepare test data once
            X_test, y_test = prepare_training_data(test_pos, test_neg)

            best_params = run.best_params

            # Extract model hyperparameters
            model_kwargs = {k: v for k, v in best_params.items() if k in ["hidden_dim", "dropout"]}

            # Step 5: Train 10 times with different random initializations and collect test accuracies
            test_accuracies = []
            for run_idx in range(10):
                # Different seed for each run (used by fit() for weight init, shuffling, dropout)
                seed = 42 + run_idx

                # Create config with seed - fit() will call torch.manual_seed(seed) internally
                run_cfg = ClassifierTrainConfig(
                    test_size=0.0,  # Use all training data
                    num_epochs=best_params.get("num_epochs", 50),
                    batch_size=best_params.get("batch_size", 8),
                    learning_rate=best_params.get("lr", 1e-3),
                    monitor="accuracy",
                    random_state=seed,  # Different seed for each run
                )

                # Create new classifier instance (new random initialization)
                final_clf = MLPClassifier(device="cuda" if torch.cuda.is_available() else "cpu")

                final_clf.fit(
                    X_train, y_train,
                    config=run_cfg,  # Use run-specific config with different seed
                    optimizer=best_params.get("optimizer", "Adam"),
                    lr=best_params.get("lr", 1e-3),
                    optimizer_kwargs=best_params.get("optimizer_kwargs"),
                    criterion="bce",
                    layers=model_kwargs,
                )

                # Evaluate on test set
                with torch.no_grad():
                    predictions = final_clf.predict(X_test)
                    # Convert to tensor if it's a list
                    if isinstance(predictions, list):
                        predictions = torch.tensor(predictions).float()
                    accuracy = (predictions == y_test).float().mean().item()
                    test_accuracies.append(accuracy)

                print(f"    Run {run_idx + 1}/10: {accuracy:.3f}")

            # Calculate mean and std of test accuracies
            mean_accuracy = np.mean(test_accuracies)
            std_accuracy = np.std(test_accuracies)
            min_accuracy = np.min(test_accuracies)
            max_accuracy = np.max(test_accuracies)

            layer_accuracies[layer_name] = mean_accuracy

            # Store detailed metadata for this layer
            layer_metadata[layer_name] = {
                "mean_accuracy": float(mean_accuracy),
                "std_accuracy": float(std_accuracy),
                "min_accuracy": float(min_accuracy),
                "max_accuracy": float(max_accuracy),
                "all_accuracies": [float(acc) for acc in test_accuracies],
                "best_params": best_params,
                "best_val_accuracy": float(run.best_value),
            }

            print(f"  Mean test accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            layer_accuracies[layer_name] = 0.0
            layer_metadata[layer_name] = {
                "mean_accuracy": 0.0,
                "std_accuracy": 0.0,
                "min_accuracy": 0.0,
                "max_accuracy": 0.0,
                "all_accuracies": [],
                "best_params": {},
                "best_val_accuracy": 0.0,
                "error": str(e),
            }

    return layer_accuracies, layer_metadata


def calculate_binomial_error(accuracy: float, n_samples: int) -> float:
    """
    Calculate binomial standard error for accuracy.

    Standard error = sqrt(p * (1 - p) / n)
    where p is the accuracy (proportion correct)
    """
    p = accuracy
    if n_samples == 0:
        return 0.0
    return np.sqrt(p * (1 - p) / n_samples)


def plot_results(
    results: Dict[int, Dict[str, float]],
    num_test: int,
    aggregation_name: str,
    output_path: str = "/workspace/results/gsm8k/gsm8k_train_size_improve_plot.png",
):
    """
    Plot layer vs accuracy curves for different training sizes.

    Args:
        results: Dict[k] -> Dict[layer_name] -> accuracy
        num_test: Number of test samples (for error bar calculation)
        aggregation_name: Name of the aggregation method used
        output_path: Path to save the plot
    """

    print("\n" + "=" * 80)
    print("CREATING PLOT")
    print("=" * 80)

    plt.figure(figsize=(14, 8))

    # Plot each k value
    for k in sorted(results.keys()):
        layer_accuracies = results[k]

        # Sort by layer number
        layers = sorted([int(layer) for layer in layer_accuracies.keys()])
        accuracies = [layer_accuracies[str(layer)] for layer in layers]

        # Calculate binomial standard error
        errors = [calculate_binomial_error(acc, num_test * 2) for acc in accuracies]

        # Convert to percentage
        accuracies_pct = [acc * 100 for acc in accuracies]
        errors_pct = [err * 100 for err in errors]

        # Plot line
        line, = plt.plot(
            layers,
            accuracies_pct,
            marker='o',
            linewidth=2,
            markersize=6,
            label=f'{k*2} training examples',
            alpha=0.8
        )

        # Add confidence band (mean ± std error)
        plt.fill_between(
            layers,
            [acc - err for acc, err in zip(accuracies_pct, errors_pct)],
            [acc + err for acc, err in zip(accuracies_pct, errors_pct)],
            alpha=0.2,
            color=line.get_color()
        )

    plt.xlabel("Layer Number", fontsize=14)
    plt.ylabel("Test Accuracy (%)", fontsize=14)
    plt.title(f"Classifier Performance vs Layer ({aggregation_name}) - GSM8K", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=12)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random baseline (50%)')

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    # Configuration
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    aggregation_method = ActivationAggregationStrategy.CHOICE_TOKEN
    k_values = [5, 10, 20, 50, 100, 250]
    num_val = 50  
    num_test = 150
    n_trials = 40

    # Store results: results[k] = {layer_name: accuracy}
    results: Dict[int, Dict[str, float]] = {}
    all_metadata: Dict[int, Dict] = {}  # Store metadata for each k

    print("=" * 80)
    print("CLASSIFIER PERFORMANCE EXPERIMENT")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Aggregation: {aggregation_method.value}")
    print(f"Training sizes (k): {k_values}")
    print(f"Test size: {num_test}")
    print(f"Optuna trials per layer: {n_trials}")
    print("=" * 80)

    # Train and evaluate for each k
    for k in k_values:
        print(f"\n\n{'='*80}")
        print(f"EXPERIMENT: k={k}")
        print(f"{'='*80}\n")

        layer_accuracies, layer_metadata = train_and_evaluate_for_k(
            k=k,
            num_val = num_val,
            num_test=num_test,
            model_name=model_name,
            aggregation_method=aggregation_method,
            n_trials=n_trials,
        )

        results[k] = layer_accuracies
        all_metadata[k] = layer_metadata

        # Save metadata as JSON
        accuracies_list = list(layer_accuracies.values())
        best_layer = max(layer_accuracies.keys(), key=lambda l: layer_accuracies[l])

        metadata = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_name": model_name,
                "benchmark": "gsm8k",
                "aggregation": aggregation_method.value,
                "train_size_k": k,
                "n_trials": n_trials,
                "n_runs": 10,
            },
            "data_config": {
                "train_size": k,
                "val_size": num_val,
                "test_size": num_test,
                "train_source": "training",
                "val_source": "training",
                "test_source": "test",
            },
            "results_by_layer": layer_metadata,
            "summary": {
                "best_layer": int(best_layer),
                "best_mean_accuracy": float(layer_accuracies[best_layer]),
                "overall_mean_accuracy": float(np.mean(accuracies_list)),
                "overall_std_accuracy": float(np.std(accuracies_list)),
            }
        }

        # Save to JSON file
        output_dir = "/workspace/results/gsm8k"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/gsm8k_train_size_k{k}_{aggregation_method.value}_results.json"

        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nMetadata saved to: {output_file}")

        # Print summary for this k
        print(f"\nSummary for k={k}:")
        print(f"  Mean accuracy: {np.mean(list(layer_accuracies.values())):.3f}")
        print(f"  Max accuracy: {np.max(list(layer_accuracies.values())):.3f}")
        print(f"  Min accuracy: {np.min(list(layer_accuracies.values())):.3f}")

        # Clean up GPU memory between k iterations
        print(f"\nCleaning up GPU memory...")
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(10)
        print(f"Ready for next k value.")

    # Plot results
    plot_results(results, num_test=num_test, aggregation_name=aggregation_method.value)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)


