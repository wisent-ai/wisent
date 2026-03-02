"""_SteeringOptimizerClassifier - comparison, classifier loading, cache management.

Split from steering_optimization.py to meet 300-line limit.
"""

import logging
import traceback
from typing import Any, Dict, List, Optional

from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
from wisent.core.reading.classifiers.core.atoms import Classifier
from wisent.core.utils.config_tools.constants import (
    BLEND_DEFAULT,
    CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_LAYER_RANGE_END,
    DEFAULT_LIMIT,
    DEFAULT_SCORE,
    KEEP_RECENT_HOURS_DEFAULT,
    OPTUNA_N_TRIALS_SMALL,
)
from wisent.core.utils.config_tools.optuna.classifier import (
    ClassifierOptimizationConfig,
    GenerationConfig,
    OptunaClassifierOptimizer,
)
from wisent.core.utils.config_tools.optuna.steering import metrics
from wisent.core.utils.infra_tools.errors import MissingParameterError

logger = logging.getLogger(__name__)


class _SteeringOptimizerClassifier:
    """Classifier mixin: compare predictions, load/find classifier, cache management."""

    def compare_predictions(
        self,
        baseline_predictions: List[str],
        steered_predictions: List[str],
        ground_truths: List[str],
        model,
        tokenizer,
        device: str,
        max_length: int | None = None,
        task_name: str = "gsm8k",
    ) -> Dict[str, Any]:
        """
        Compare baseline vs steered predictions using benchmark metrics and classifier scores.

        Returns:
            Enhanced metrics with baseline vs steered comparison including classifier scores
        """
        if max_length is None:
            max_length = tokenizer.model_max_length
        # Create classifier scorer function for metrics integration
        classifier_scorer = lambda predictions, description: self.score_predictions_with_classifier(
            predictions, model, tokenizer, device, max_length, description
        )

        # Calculate standard benchmark metrics with integrated classifier confidence scores
        baseline_metrics = metrics.evaluate_benchmark_performance(
            baseline_predictions, ground_truths, task_name, classifier_scorer=classifier_scorer
        )
        steered_metrics = metrics.evaluate_benchmark_performance(
            steered_predictions, ground_truths, task_name, classifier_scorer=classifier_scorer
        )

        # Extract classifier scores from integrated metrics
        baseline_scores = [
            detail.get("classifier_confidence", BLEND_DEFAULT) for detail in baseline_metrics.get("evaluation_details", [])
        ]
        steered_scores = [
            detail.get("classifier_confidence", BLEND_DEFAULT) for detail in steered_metrics.get("evaluation_details", [])
        ]

        # Calculate improvement metrics
        accuracy_delta = steered_metrics.get("accuracy", DEFAULT_SCORE) - baseline_metrics.get("accuracy", DEFAULT_SCORE)
        f1_delta = steered_metrics.get("f1", DEFAULT_SCORE) - baseline_metrics.get("f1", DEFAULT_SCORE)

        # Calculate classifier score improvements
        avg_baseline_score = sum(baseline_scores) / len(baseline_scores) if baseline_scores else DEFAULT_SCORE
        avg_steered_score = sum(steered_scores) / len(steered_scores) if steered_scores else DEFAULT_SCORE
        classifier_score_delta = avg_steered_score - avg_baseline_score

        return {
            "baseline": {
                "accuracy": baseline_metrics.get("accuracy", DEFAULT_SCORE),
                "f1": baseline_metrics.get("f1", DEFAULT_SCORE),
                "classifier_scores": baseline_scores,
                "avg_classifier_score": avg_baseline_score,
                "predictions": baseline_predictions,
            },
            "steered": {
                "accuracy": steered_metrics.get("accuracy", DEFAULT_SCORE),
                "f1": steered_metrics.get("f1", DEFAULT_SCORE),
                "classifier_scores": steered_scores,
                "avg_classifier_score": avg_steered_score,
                "predictions": steered_predictions,
            },
            "improvement": {
                "accuracy_delta": accuracy_delta,
                "f1_delta": f1_delta,
                "classifier_score_delta": classifier_score_delta,
            },
        }

    def load_or_find_best_classifier(
        self,
        model,
        optimization_config: Optional[ClassifierOptimizationConfig] = None,
        model_name: Optional[str] = None,
        task_name: Optional[str] = None,
        contrastive_pairs: Optional[List] = None,
        force_reoptimize: bool = False,
    ) -> Optional[Classifier]:
        """
        Load or train the best classifier for current steering session.

        On first call: Run full classifier optimization and cache result for session
        On subsequent calls: Return cached classifier from current session

        Returns:
            Best trained classifier or None if optimization failed
        """
        # Extract configuration
        if optimization_config is not None:
            model_name = optimization_config.model_name
            task_name = getattr(optimization_config, "task_name", task_name)
            limit = getattr(optimization_config, "data_limit", DEFAULT_LIMIT)
        else:
            limit = DEFAULT_LIMIT

        if not model_name or not task_name:
            raise MissingParameterError(params=["model_name", "task_name"])

        # Create session cache key
        session_cache_key = f"{model_name}_{task_name}"

        # Check if we already have a classifier for this session
        if (
            not force_reoptimize
            and self._session_classifier is not None
            and self._session_cache_key == session_cache_key
        ):
            self.logger.info("Using cached classifier from current session")
            return self._session_classifier

        # First call or forced reoptimization - run classifier optimization
        self.logger.info("Running classifier optimization (first trial in session)")

        if not contrastive_pairs:
            self.logger.error("contrastive_pairs required for classifier optimization")
            return None

        try:
            # Create configuration for classifier optimization if not provided
            if optimization_config is None:
                optimization_config = ClassifierOptimizationConfig(
                    model_name=model_name,
                    device="auto",
                    n_trials=OPTUNA_N_TRIALS_SMALL,
                    model_types=["logistic", "mlp"],
                    primary_metric="f1",
                )

            # Create generation config for activation pre-generation
            generation_config = GenerationConfig(
                layer_search_range=(0, CLASSIFIER_LAYER_RANGE_END),
                aggregation_methods=[
                    ExtractionStrategy.CHAT_MEAN,
                    ExtractionStrategy.CHAT_LAST,
                    ExtractionStrategy.CHAT_FIRST,
                    ExtractionStrategy.CHAT_MAX_NORM,
                ],
                cache_dir="./cache/steering_activations",
                device=optimization_config.device,
                batch_size=CLASSIFIER_BATCH_SIZE,
            )

            # Create classifier optimizer
            classifier_optimizer = OptunaClassifierOptimizer(
                optimization_config=optimization_config,
                generation_config=generation_config,
                cache_config=self.classifier_cache.config,
            )

            # Run classifier optimization
            self.logger.info(f"Optimizing classifier for {model_name}/{task_name} with {len(contrastive_pairs)} pairs")
            result = classifier_optimizer.optimize(
                model=model,
                contrastive_pairs=contrastive_pairs,
                task_name=task_name,
                model_name=model_name,
                limit=limit,
            )

            if result.best_value > 0:
                # Get the best configuration and classifier
                best_config = result.get_best_config()
                best_classifier = result.best_classifier

                # Cache for current session
                self._session_classifier = best_classifier
                self._session_classifier_metadata = {
                    "layer": best_config["layer"],
                    "aggregation": best_config["aggregation"],
                    "model_type": best_config["model_type"],
                    "threshold": best_config["threshold"],
                    "f1_score": result.best_value,
                    "hyperparameters": best_config.get("hyperparameters", {}),
                }
                self._session_cache_key = session_cache_key

                self.logger.info(
                    f"Cached best classifier for session: layer_{best_config['layer']} "
                    f"{best_config['model_type']} (F1: {result.best_value:.3f})"
                )

                return best_classifier
            self.logger.warning("Classifier optimization failed - no successful trials")
            return None

        except Exception as e:
            self.logger.error(f"Failed to run classifier optimization: {e}")
            traceback.print_exc()
            return None

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached classifiers."""
        return self.classifier_cache.get_cache_info()

    def clear_classifier_cache(self, keep_recent_hours: float = KEEP_RECENT_HOURS_DEFAULT) -> int:
        """Clear old cached classifiers."""
        return self.classifier_cache.clear_cache(keep_recent_hours=keep_recent_hours)
