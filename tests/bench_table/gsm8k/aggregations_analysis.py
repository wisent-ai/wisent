"""
Analyze classifier performance across different aggregation strategies.

For each aggregation method:
- Train: 250 pairs (500 activations) from training docs
- Validation: 50 pairs (100 activations) from training docs
- Test: 150 pairs (300 activations) from test docs

Creates individual plots for each aggregation method and one combined plot.
"""

from __future__ import annotations

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import optuna
import time
import gc

from wisent_guard.opti.methods.opti_classificator import ClassificationOptimizer
from wisent_guard.opti.core.atoms import HPOConfig
from wisent_guard.classifiers.models.mlp import MLPClassifier
from wisent_guard.classifiers.core.atoms import ClassifierTrainConfig

from activation_martix import create_activations_matrix
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
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),  # Larger batches for 500 activations
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


def train_and_evaluate_aggregation(
    aggregation_method: ActivationAggregationStrategy,
    num_train: int,
    num_val: int,
    num_test: int,
    model_name: str,
    n_trials: int = 40,
) -> Dict[str, float]:
    """
    Train classifiers for one aggregation method and evaluate across all layers.

    Args:
        aggregation_method: Aggregation strategy to use
        num_train: Number of training pairs
        num_val: Number of validation pairs
        num_test: Number of test pairs
        model_name: HuggingFace model name
        n_trials: Number of Optuna trials

    Returns:
        Dict mapping layer_name -> test accuracy
    """

    print("=" * 80)
    print(f"AGGREGATION: {aggregation_method.value}")
    print(f"TRAIN: {num_train} pairs, VAL: {num_val} pairs, TEST: {num_test} pairs")
    print("=" * 80)

    # Step 1: Create training+validation activations matrix from training docs
    print(f"\nCreating TRAINING+VALIDATION activations matrix for {num_train + num_val} pairs...")
    train_val_result = create_activations_matrix(
        model_name=model_name,
        aggregation_methods=[aggregation_method],
        num_questions=num_train + num_val,
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
    print(f"\nCreating TEST activations matrix for {num_test} pairs...")
    test_result = create_activations_matrix(
        model_name=model_name,
        aggregation_methods=[aggregation_method],
        num_questions=num_test,
        output_path=None,
        preferred_doc="test"
    )

    test_matrix = test_result["matrix"][aggregation_method.value]

    # Calculate validation split size
    val_size = num_val / (num_train + num_val)

    print(f"\nData split:")
    print(f"  Training: {num_train} pairs ({num_train * 2} activations)")
    print(f"  Validation: {num_val} pairs ({num_val * 2} activations)")
    print(f"  Test: {num_test} pairs ({num_test * 2} activations)")
    print(f"  Validation split for HPO: {val_size*100:.1f}%")

    # Step 3: Train and evaluate classifier for each layer
    print(f"\nTraining and evaluating classifiers for each layer...")
    layer_accuracies: Dict[str, float] = {}

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

        print(f"  Training+Validation samples: {len(X_train_val)}")
        print(f"  Test samples: {len(test_pos) + len(test_neg)}")

        # Set up base config and model space
        make_classifier = lambda: MLPClassifier(device="cuda" if torch.cuda.is_available() else "cpu")

        base_cfg = ClassifierTrainConfig(
            test_size=val_size,
            num_epochs=50,
            batch_size=32,  # Fixed batch size for 600 activations (Optuna will search over [16, 32, 64])
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
                objective_metric="accuracy",
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
            print(f"  Running Optuna optimization ({n_trials} trials)...")
            run = tuner.optimize(hpo_cfg)

            print(f"  Best validation accuracy: {run.best_value:.3f}")
            print(f"  Best params: {run.best_params}")

            # Step 4: Train final model on ONLY training data (first num_train pairs) with best params
            print(f"  Training final model on training data only (first {num_train} pairs)...")

            # Extract only training data (first num_train pairs)
            train_pos_only = train_val_pos[:num_train]
            train_neg_only = train_val_neg[:num_train]
            X_train, y_train = prepare_training_data(train_pos_only, train_neg_only)

            best_params = run.best_params
            final_clf = MLPClassifier(device="cuda" if torch.cuda.is_available() else "cpu")

            # Train on training data only (no validation)
            final_cfg = ClassifierTrainConfig(
                test_size=0.0,  # Use all training data
                num_epochs=best_params.get("num_epochs", 50),
                batch_size=best_params.get("batch_size", 8),
                learning_rate=best_params.get("lr", 1e-3),
                monitor="accuracy",
            )

            # Extract model hyperparameters
            model_kwargs = {k: v for k, v in best_params.items() if k in ["hidden_dim", "dropout"]}

            final_clf.fit(
                X_train, y_train,
                config=final_cfg,
                optimizer=best_params.get("optimizer", "Adam"),
                lr=best_params.get("lr", 1e-3),
                optimizer_kwargs=best_params.get("optimizer_kwargs"),
                criterion="bce",
                layers=model_kwargs,
            )

            # Step 5: Evaluate on separate test set
            X_test, y_test = prepare_training_data(test_pos, test_neg)

            with torch.no_grad():
                predictions = final_clf.predict(X_test)
                # Convert to tensor if it's a list
                if isinstance(predictions, list):
                    predictions = torch.tensor(predictions)
                predicted_labels = (predictions > 0.5).float()
                test_accuracy = (predicted_labels == y_test).float().mean().item()

            layer_accuracies[layer_name] = test_accuracy

            print(f"  Final test accuracy: {test_accuracy:.3f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            layer_accuracies[layer_name] = 0.0

    return layer_accuracies


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


def plot_individual_aggregation(
    aggregation_name: str,
    layer_accuracies: Dict[str, float],
    num_test: int,
    output_path: str,
):
    """Plot results for a single aggregation method."""

    print(f"\nCreating plot for {aggregation_name}...")

    plt.figure(figsize=(12, 7))

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
        alpha=0.8,
        color='#2E86AB'
    )

    # Add confidence band
    plt.fill_between(
        layers,
        [acc - err for acc, err in zip(accuracies_pct, errors_pct)],
        [acc + err for acc, err in zip(accuracies_pct, errors_pct)],
        alpha=0.2,
        color=line.get_color()
    )

    plt.xlabel("Layer Number", fontsize=14)
    plt.ylabel("Test Accuracy (%)", fontsize=14)
    plt.title(f"Classifier Performance: {aggregation_name}", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random baseline (50%)')
    plt.legend(loc='best', fontsize=12)

    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Plot saved to {output_path}")


def plot_combined_aggregations(
    results: Dict[str, Dict[str, float]],
    num_test: int,
    output_path: str = "tests/bench_table/gsm8k/plots/aggregations_combined_plot.png",
):
    """Plot all aggregation methods on the same chart."""

    print("\n" + "=" * 80)
    print("CREATING COMBINED PLOT")
    print("=" * 80)

    plt.figure(figsize=(14, 8))

    # Plot each aggregation method
    for agg_name in sorted(results.keys()):
        layer_accuracies = results[agg_name]

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
            label=agg_name,
            alpha=0.8
        )

        # Add confidence band
        plt.fill_between(
            layers,
            [acc - err for acc, err in zip(accuracies_pct, errors_pct)],
            [acc + err for acc, err in zip(accuracies_pct, errors_pct)],
            alpha=0.15,
            color=line.get_color()
        )

    plt.xlabel("Layer Number", fontsize=14)
    plt.ylabel("Test Accuracy (%)", fontsize=14)
    plt.title("Classifier Performance: All Aggregation Methods", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=11)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random baseline (50%)')

    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Combined plot saved to {output_path}")


if __name__ == "__main__":
    # Configuration
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    num_train = 250  # 500 activations
    num_val = 50     # 100 activations
    num_test = 150   # 300 activations
    n_trials = 40

    # All aggregation strategies
    agg_strategies = [
        ActivationAggregationStrategy.CONTINUATION_TOKEN,
        ActivationAggregationStrategy.LAST_TOKEN,
        ActivationAggregationStrategy.FIRST_TOKEN,
        ActivationAggregationStrategy.MEAN_POOLING,
        ActivationAggregationStrategy.CHOICE_TOKEN,
        ActivationAggregationStrategy.MAX_POOLING,
    ]

    # Store results: results[agg_name] = {layer_name: accuracy}
    results: Dict[str, Dict[str, float]] = {}

    print("=" * 80)
    print("AGGREGATION METHODS ANALYSIS")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Training: {num_train} pairs ({num_train * 2} activations)")
    print(f"Validation: {num_val} pairs ({num_val * 2} activations)")
    print(f"Test: {num_test} pairs ({num_test * 2} activations)")
    print(f"Optuna trials per layer: {n_trials}")
    print(f"Aggregation methods: {len(agg_strategies)}")
    print("=" * 80)

    # Train and evaluate for each aggregation method
    for agg_method in agg_strategies:
        agg_name = agg_method.value

        print(f"\n\n{'='*80}")
        print(f"ANALYZING: {agg_name}")
        print(f"{'='*80}\n")

        layer_accuracies = train_and_evaluate_aggregation(
            aggregation_method=agg_method,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            model_name=model_name,
            n_trials=n_trials,
        )

        results[agg_name] = layer_accuracies

        # Print summary for this aggregation
        accuracies_list = list(layer_accuracies.values())
        print(f"\nSummary for {agg_name}:")
        print(f"  Mean accuracy: {np.mean(accuracies_list):.3f}")
        print(f"  Max accuracy: {np.max(accuracies_list):.3f}")
        print(f"  Min accuracy: {np.min(accuracies_list):.3f}")

        # Create individual plot
        plot_individual_aggregation(
            aggregation_name=agg_name,
            layer_accuracies=layer_accuracies,
            num_test=num_test,
            output_path=f"tests/bench_table/gsm8k/plots/aggregation_{agg_name}_plot.png",
        )

        # Clean up GPU memory between aggregations
        print(f"\nCleaning up GPU memory...")
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(10)
        print(f"Ready for next aggregation method.")

    # Create combined plot with all aggregations
    plot_combined_aggregations(results, num_test=num_test)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated plots:")
    for agg_name in results.keys():
        print(f"  - aggregation_{agg_name}_plot.png")
    print(f"  - aggregations_combined_plot.png")
