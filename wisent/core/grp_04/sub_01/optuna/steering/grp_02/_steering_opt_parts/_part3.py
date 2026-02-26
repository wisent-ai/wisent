"""_SteeringOptimizerCore - init, register, optimize, and hyperparameter generation.

Split from steering_optimization.py to meet 300-line limit.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from wisent.core.optuna.classifier import (
    CacheConfig,
    ClassifierCache,
    ClassifierOptimizationConfig,
)
from wisent.core.optuna.steering import data_utils, metrics
from wisent.core.steering_methods import SteeringMethodRegistry
from wisent.core.errors import (
    SteeringTrainerNotFoundError,
    ClassifierLoadError,
)

from ._part2 import SteeringMethodTrainer, SteeringTrainer
from wisent.core.constants import CLASSIFIER_BATCH_SIZE


class _SteeringOptimizerCore:
    """Core mixin: __init__, register_trainer, optimize, hyperparameter generation."""

    def __init__(self, cache_config: Optional[CacheConfig] = None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize trainers from centralized registry
        self.trainers = {}
        for method_name in SteeringMethodRegistry.list_methods():
            self.trainers[method_name] = SteeringTrainer(method_name)

        # Initialize classifier cache for reusing trained classifiers
        if cache_config is None:
            cache_config = CacheConfig(cache_dir="./steering_classifier_cache")
        self.classifier_cache = ClassifierCache(cache_config)

        # Session-level classifier caching for current optimization run
        self._session_classifier = None  # Best classifier for current session
        self._session_classifier_metadata = {}  # Layer, model_type, performance, etc.
        self._session_cache_key = None  # Track current session

    def register_trainer(self, method_name: str, trainer: SteeringMethodTrainer):
        """Register a new steering method trainer."""
        self.trainers[method_name] = trainer
        self.logger.info(f"Registered trainer for steering method: {method_name}")

    def optimize_steering_hyperparameters(
        self,
        config,
        classifier_optimization_config: ClassifierOptimizationConfig,
        train_samples: List[Dict],
        validation_samples: List[Dict],
        model,
        tokenizer,
        device: str,
        batch_size: int = CLASSIFIER_BATCH_SIZE,
        max_length: int | None = None,
        task_name: str = "gsm8k",
        max_new_tokens: int = None,
    ) -> Tuple[Dict[str, Any], List]:
        """
        Optimize hyperparameters for a steering method using grid search.

        Returns:
            Tuple of (best_config, all_results)
        """
        if max_length is None:
            max_length = tokenizer.model_max_length
        from wisent.core.optuna.steering.steering_optimization import SteeringResult

        method_name = config.method_name

        if method_name not in self.trainers:
            raise SteeringTrainerNotFoundError(method=method_name)

        trainer = self.trainers[method_name]

        # Load best classifier once at the start of optimization
        self.logger.info("Loading/training classifier for evaluation...")
        contrastive_pairs = data_utils.get_task_contrastive_pairs(train_samples, task_name)

        classifier = self.load_or_find_best_classifier(
            model=model, optimization_config=classifier_optimization_config, contrastive_pairs=contrastive_pairs
        )

        if classifier is None:
            raise ClassifierLoadError()

        self.logger.info(f"Using classifier: {self._session_classifier_metadata}")

        # Collect baseline predictions once for all trials
        self.logger.info("Collecting baseline predictions for comparison...")
        baseline_predictions, ground_truths = self.collect_baseline_predictions(
            validation_samples, model, tokenizer, classifier, device, batch_size, max_length, task_name, max_new_tokens
        )

        # Calculate baseline metrics with integrated classifier scoring
        classifier_scorer = lambda predictions, description: self.score_predictions_with_classifier(
            predictions, model, tokenizer, device, max_length, description
        )
        baseline_benchmark_metrics = metrics.evaluate_benchmark_performance(
            baseline_predictions, ground_truths, task_name, classifier_scorer=classifier_scorer
        )
        self.logger.info(f"Baseline performance: {baseline_benchmark_metrics}")

        # Generate all hyperparameter combinations
        hyperparameter_combinations = self._generate_hyperparameter_combinations(config)

        self.logger.info(f"Starting {method_name} optimization with {len(hyperparameter_combinations)} configurations")

        best_config = None
        best_score = -1
        all_results = []

        for i, (layer, strength, hyperparams) in enumerate(
            tqdm(hyperparameter_combinations, desc="Optimizing steering hyperparameters")
        ):
            self.logger.debug(
                f"Testing {method_name} config {i + 1}/{len(hyperparameter_combinations)}: "
                f"layer={layer}, strength={strength}, hyperparams={hyperparams}"
            )

            try:
                result = self._evaluate_single_config(
                    trainer, method_name, layer, strength, hyperparams, i,
                    train_samples, validation_samples, model, tokenizer, device,
                    batch_size, max_length, task_name, max_new_tokens,
                    baseline_predictions, ground_truths, baseline_benchmark_metrics,
                )
                all_results.append(result)

                if not result.training_success:
                    continue

                steered_accuracy = result.benchmark_metrics.get("accuracy", 0.0)
                if steered_accuracy > best_score:
                    best_score = steered_accuracy
                    baseline_acc = result.baseline_metrics.get("accuracy", 0.0) if result.baseline_metrics else 0.0
                    best_config = {
                        "method": method_name,
                        "layer": layer,
                        "strength": strength,
                        **hyperparams,
                        "benchmark_metrics": result.benchmark_metrics,
                        "baseline_metrics": result.baseline_metrics,
                        "method_instance": result.training_stats.get("method_instance"),
                    }

                    self.logger.debug(
                        f"Config {i + 1} - Baseline: {baseline_acc:.3f}, "
                        f"Steered: {steered_accuracy:.3f}, Delta: {steered_accuracy - baseline_acc:+.3f}"
                    )

            except Exception as e:
                self.logger.error(f"Failed to evaluate config {i + 1}: {e}")
                result = SteeringResult(
                    method_name=method_name,
                    layer=layer,
                    hyperparameters={**hyperparams, "strength": strength},
                    benchmark_metrics={"accuracy": 0.0},
                    baseline_metrics=baseline_benchmark_metrics,
                    comparative_metrics={"accuracy_delta": 0.0, "improvement_rate": 0.0},
                    training_success=False,
                    training_stats={"error": str(e)},
                )
                all_results.append(result)
                continue

        if best_config is None:
            self.logger.warning("No successful steering configuration found")
            best_config = {
                "method": method_name,
                "layer": config.layers[0] if config.layers else 0,
                "strength": config.strengths[0] if config.strengths else 1.0,
                "benchmark_metrics": {"accuracy": 0.0},
                "method_instance": None,
            }
        else:
            steered_acc = best_config["benchmark_metrics"]["accuracy"]
            baseline_acc = best_config.get("baseline_metrics", {}).get("accuracy", 0.0)
            improvement = steered_acc - baseline_acc

            self.logger.info(
                f"Best {method_name} config (optimized for steered accuracy): "
                f"layer={best_config['layer']}, steered={steered_acc:.3f} "
                f"(baseline={baseline_acc:.3f}, delta={improvement:+.3f})"
            )

        return best_config, all_results

    def _evaluate_single_config(
        self, trainer, method_name, layer, strength, hyperparams, index,
        train_samples, validation_samples, model, tokenizer, device,
        batch_size, max_length, task_name, max_new_tokens,
        baseline_predictions, ground_truths, baseline_benchmark_metrics,
    ):
        """Evaluate a single steering configuration. Returns SteeringResult."""
        from wisent.core.optuna.steering.steering_optimization import SteeringResult

        method_instance = trainer.create_method_instance(hyperparams, device)

        training_success, training_stats = trainer.train_method(
            method_instance, train_samples, layer, model, tokenizer, device, task_name, max_new_tokens
        )

        if not training_success:
            self.logger.warning(f"Training failed for config {index + 1}")
            return SteeringResult(
                method_name=method_name,
                layer=layer,
                hyperparameters={**hyperparams, "strength": strength},
                benchmark_metrics={"accuracy": 0.0},
                training_success=False,
                training_stats=training_stats,
            )

        steered_predictions, steered_ground_truths = trainer.apply_steering_and_evaluate(
            method_instance, validation_samples, layer, strength,
            model, tokenizer, device, batch_size, max_length, task_name, max_new_tokens,
        )

        enhanced_metrics = self.compare_predictions(
            baseline_predictions, steered_predictions, ground_truths,
            model, tokenizer, device, max_length, task_name,
        )

        benchmark_metrics = enhanced_metrics["steered"]
        baseline_metrics_for_result = enhanced_metrics["baseline"]
        comparative_metrics = enhanced_metrics["improvement"]

        training_stats["method_instance"] = method_instance

        return SteeringResult(
            method_name=method_name,
            layer=layer,
            hyperparameters={**hyperparams, "strength": strength},
            benchmark_metrics=benchmark_metrics,
            baseline_metrics=baseline_metrics_for_result,
            comparative_metrics=comparative_metrics,
            training_success=True,
            training_stats=training_stats,
        )

    def _generate_hyperparameter_combinations(self, config) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Generate all combinations of hyperparameters for grid search using registry."""
        combinations = []

        # Get default params from registry for this method
        try:
            definition = SteeringMethodRegistry.get(config.method_name)
            default_params = definition.get_default_params()
        except ValueError:
            default_params = {}

        for layer in config.layers:
            for strength in config.strengths:
                combinations.append((layer, strength, default_params))

        return combinations
