"""Optuna objective function and classifier training helpers."""
import logging
import time
from typing import Any, Optional
import numpy as np
import optuna
import torch
from wisent.core.reading.classifiers.core.atoms import Classifier
from wisent.core.utils import resolve_default_device, preferred_dtype
from wisent.core.utils.config_tools.optuna.classifier._optuna_config import get_model_dtype

logger = logging.getLogger(__name__)


class OptunaObjectiveMixin:
    """Mixin providing objective function and training helpers."""

    def _objective_function(self, trial: optuna.Trial, task_name: str, model_name: str) -> float:
        """
        Optuna objective function for a single trial.

        Args:
            trial: Optuna trial object
            task_name: Task name
            model_name: Model name

        Returns:
            Objective value to maximize
        """
        # Sample hyperparameters directly (following steering pattern)
        model_type = trial.suggest_categorical("model_type", self.opt_config.model_types)

        # Layer and aggregation from pre-generated activation data
        available_layers = set()
        available_aggregations = set()

        for key in self.activation_data.keys():
            parts = key.split("_")
            if len(parts) >= 4:  # layer_X_agg_Y
                layer = int(parts[1])
                agg = parts[3]
                available_layers.add(layer)
                available_aggregations.add(agg)

        layer = trial.suggest_categorical("layer", sorted(available_layers))
        aggregation = trial.suggest_categorical("aggregation", sorted(available_aggregations))

        # Classification threshold
        threshold = trial.suggest_float(
            "threshold", self.opt_config.threshold_range[0], self.opt_config.threshold_range[1]
        )

        # Training hyperparameters
        num_epochs = trial.suggest_int(
            "num_epochs", self.opt_config.num_epochs_range[0], self.opt_config.num_epochs_range[1]
        )

        learning_rate = trial.suggest_float(
            "learning_rate", self.opt_config.learning_rate_range[0], self.opt_config.learning_rate_range[1], log=True
        )

        batch_size = trial.suggest_categorical("batch_size", self.opt_config.batch_size_options)

        # Model-specific hyperparameters (conditional logic like steering)
        hyperparams = {"num_epochs": num_epochs, "learning_rate": learning_rate, "batch_size": batch_size}

        if model_type == "mlp":
            # MLP-specific parameters
            hyperparams["hidden_dim"] = trial.suggest_int(
                "hidden_dim", self.opt_config.hidden_dim_range[0], self.opt_config.hidden_dim_range[1], step=32
            )

        # Combine all parameters
        params = {
            "model_type": model_type,
            "layer": layer,
            "aggregation": aggregation,
            "threshold": threshold,
            **hyperparams,
        }

        # Get activation data for this configuration
        activation_key = f"layer_{params['layer']}_agg_{params['aggregation']}"

        if activation_key not in self.activation_data:
            self.logger.warning(f"No activation data for {activation_key}")
            raise optuna.TrialPruned()

        activation_data = self.activation_data[activation_key]
        X, y = activation_data.to_tensors(device=self.gen_config.device, dtype=self.model_dtype)
        print(f"DEBUG: Training data shape: X.shape={X.shape}, y.shape={y.shape}, dtype={X.dtype}")

        # Generate cache key
        data_hash = self.classifier_cache.compute_data_hash(X, y)
        cache_key = self.classifier_cache.get_cache_key(
            model_name=model_name,
            task_name=task_name,
            model_type=params["model_type"],
            layer=params["layer"],
            aggregation=params["aggregation"],
            threshold=params["threshold"],
            hyperparameters={
                k: v for k, v in params.items() if k not in ["model_type", "layer", "aggregation", "threshold"]
            },
            data_hash=data_hash,
        )

        # Try to load from cache
        cached_classifier = self.classifier_cache.load_classifier(cache_key)
        if cached_classifier is not None:
            self.cache_hits += 1
            # Evaluate cached classifier
            return self._evaluate_classifier(cached_classifier, X, y, params["threshold"])

        self.cache_misses += 1

        # Train new classifier
        classifier = self._train_classifier(params, X, y, log_frequency=self.opt_config.log_frequency, trial=trial)

        if classifier is None:
            raise optuna.TrialPruned()

        # Evaluate classifier
        score = self._evaluate_classifier(classifier, X, y, params["threshold"])

        # Save to cache if training was successful
        if score > 0:
            try:
                performance_metrics = {self.opt_config.primary_metric: score}

                self.classifier_cache.save_classifier(
                    cache_key=cache_key,
                    classifier=classifier,
                    model_name=model_name,
                    task_name=task_name,
                    layer=params["layer"],
                    aggregation=params["aggregation"],
                    threshold=params["threshold"],
                    hyperparameters={
                        k: v for k, v in params.items() if k not in ["model_type", "layer", "aggregation", "threshold"]
                    },
                    performance_metrics=performance_metrics,
                    training_samples=len(X),
                    data_hash=data_hash,
                )
            except Exception as e:
                self.logger.warning(f"Failed to cache classifier: {e}")

        return score

    def _train_classifier(
        self, params: dict[str, Any], X: np.ndarray, y: np.ndarray, log_frequency: int, trial: Optional[optuna.Trial] = None
    ) -> Optional[Classifier]:
        """
        Train a classifier with the given parameters.

        Args:
            params: Hyperparameters
            X: Training features
            y: Training labels
            trial: Optuna trial for pruning

        Returns:
            Trained classifier or None if training failed
        """
        try:
            # Create classifier (don't pass hidden_dim to constructor)
            classifier_kwargs = {
                "model_type": params["model_type"],
                "threshold": params["threshold"],
                "device": self.gen_config.device if self.gen_config.device else "auto",
                "dtype": self.model_dtype,
            }

            print(
                f"Preparing to train {params['model_type']} classifier with {len(X)} samples (dtype: {self.model_dtype})"
            )
            classifier = Classifier(**classifier_kwargs)

            # Train classifier
            training_kwargs = {
                "log_frequency": log_frequency,
                "num_epochs": params["num_epochs"],
                "learning_rate": params["learning_rate"],
                "batch_size": params["batch_size"],
                "test_size": self.opt_config.test_size,
                "random_state": self.opt_config.random_state,
            }

            if params["model_type"] == "mlp":
                training_kwargs["hidden_dim"] = params["hidden_dim"]

            # Add pruning callback if trial is provided
            if trial and self.opt_config.enable_pruning:
                training_kwargs["pruning_callback"] = self._create_pruning_callback(trial)

            print(f"About to fit classifier with kwargs: {training_kwargs}")
            results = classifier.fit(X, y, **training_kwargs)
            print(f"Training results: {results}")

            accuracy = results.get("accuracy", 0)
            if accuracy <= self.opt_config.prune_accuracy_threshold:  # Only prune very poor performance
                self.logger.debug(f"Classifier performance too low ({accuracy:.3f}), pruning")
                print(f"Classifier pruned - accuracy too low: {accuracy:.3f}")
                return None

            self.logger.debug(f"Classifier training successful - accuracy: {accuracy:.3f}")
            print(f"Classifier training successful - accuracy: {accuracy:.3f}")

            return classifier

        except Exception as e:
            print(f"EXCEPTION during classifier training: {e}")
            import traceback

            traceback.print_exc()
            self.logger.debug(f"Training failed with params {params}: {e}")
            return None

    def _evaluate_classifier(self, classifier: Classifier, X: np.ndarray, y: np.ndarray, threshold: float) -> float:
        """
        Evaluate classifier performance.

        Args:
            classifier: Trained classifier
            X: Features
            y: Labels
            threshold: Classification threshold

        Returns:
            Performance score based on primary metric
        """
        try:
            print(f"DEBUG: Evaluation data shape: X.shape={X.shape}, y.shape={y.shape}, dtype={X.dtype}")

            # Set threshold
            classifier.set_threshold(threshold)

            # Get predictions
            results = classifier.evaluate(X, y)
            print(f"Evaluation results: {results}")
            print(f"Looking for primary metric '{self.opt_config.primary_metric}' in results")

            # Return primary metric
            score = results.get(self.opt_config.primary_metric, 0.0)
            print(f"Score extracted: {score}")
            return float(score)

        except Exception as e:
            print(f"EXCEPTION during evaluation: {e}")
            import traceback

            traceback.print_exc()
            self.logger.debug(f"Evaluation failed: {e}")
            return 0.0

    def _train_final_classifier(self, best_params: dict[str, Any], task_name: str, model_name: str) -> Classifier:
        """Train the final classifier with best parameters."""
        # Get activation data
        activation_key = f"layer_{best_params['layer']}_agg_{best_params['aggregation']}"
        activation_data = self.activation_data[activation_key]
        X, y = activation_data.to_tensors(device=self.gen_config.device, dtype=self.model_dtype)

        # Try cache first
        data_hash = self.classifier_cache.compute_data_hash(X, y)
        cache_key = self.classifier_cache.get_cache_key(
            model_name=model_name,
            task_name=task_name,
            model_type=best_params["model_type"],
            layer=best_params["layer"],
            aggregation=best_params["aggregation"],
            threshold=best_params["threshold"],
            hyperparameters={
                k: v for k, v in best_params.items() if k not in ["model_type", "layer", "aggregation", "threshold"]
            },
            data_hash=data_hash,
        )

        cached_classifier = self.classifier_cache.load_classifier(cache_key)
        if cached_classifier is not None:
            self.logger.info("Using cached classifier for final model")
            return cached_classifier

        # Train new classifier
        self.logger.info("Training final classifier with best parameters")
        classifier = self._train_classifier(best_params, X, y)

        if classifier is None:
            raise ClassifierCreationError(issue_type="optimization")

        return classifier

