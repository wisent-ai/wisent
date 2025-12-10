"""
Evaluation methods for the steering optimization pipeline.
"""

import logging
from typing import Any, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from wisent.core.optuna.steering import metrics
from wisent.core.task_interface import get_task

from .config import OptimizationConfig
from .generation import GenerationHelper
from .results import ResultsSaver


logger = logging.getLogger(__name__)


class EvaluationHelper:
    """Helper class for evaluating steering performance."""

    def __init__(
        self,
        config: OptimizationConfig,
        generation_helper: GenerationHelper,
        results_saver: ResultsSaver,
    ):
        self.config = config
        self.generation_helper = generation_helper
        self.results_saver = results_saver
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.evaluator_type: str = "task"
        self.steering_evaluator: Optional[Any] = None
        self.val_samples: list[dict] = []
        self.val_task_docs: list[dict] = []

    def set_context(
        self,
        evaluator_type: str,
        steering_evaluator: Optional[Any],
        val_samples: list[dict],
        val_task_docs: list[dict],
    ):
        """Set evaluation context."""
        self.evaluator_type = evaluator_type
        self.steering_evaluator = steering_evaluator
        self.val_samples = val_samples
        self.val_task_docs = val_task_docs

    def evaluate_steering_on_validation(
        self,
        steering_instance: Any,
        method_name: str,
        layer_id: int,
        hyperparams: dict[str, Any],
        trial_number: int = 0,
        trial=None,
    ) -> float:
        """Evaluate steering method on validation data by re-running forward passes."""
        if steering_instance is None:
            return 0.0

        if self.evaluator_type == "refusal":
            return self._evaluate_refusal(steering_instance, method_name, layer_id, hyperparams)
        elif self.evaluator_type == "personalization":
            return self._evaluate_personalization(steering_instance, method_name, layer_id, hyperparams)
        else:
            return self._evaluate_task(steering_instance, method_name, layer_id, hyperparams, trial_number, trial)

    def _evaluate_task(
        self,
        steering_instance: Any,
        method_name: str,
        layer_id: int,
        hyperparams: dict[str, Any],
        trial_number: int = 0,
        trial=None,
    ) -> float:
        """Task-based evaluation using benchmark metrics."""
        predictions = []
        ground_truths = []
        task_docs = []

        task = get_task(self.config.val_dataset)
        extractor = task.get_extractor()

        questions = []
        ground_truths = []
        valid_samples = []

        for sample in tqdm(self.val_samples, desc="Extracting validation QA pairs", leave=False):
            qa_pair = extractor.extract_qa_pair(sample, task)
            if not qa_pair:
                continue

            question = qa_pair["formatted_question"]
            ground_truth = qa_pair["correct_answer"]
            questions.append(question)
            ground_truths.append(ground_truth)
            valid_samples.append(sample)

        if questions:
            if steering_instance is None:
                predictions = self.generation_helper.generate_baseline_batched(questions)
            else:
                strength = hyperparams["steering_alpha"]
                predictions = self.generation_helper.generate_with_steering_batched(
                    steering_instance, questions, strength, layer_id
                )

            for i, (pred, gt) in enumerate(zip(predictions[:3], ground_truths[:3])):
                self.logger.debug(f"{method_name.upper()} Sample {i} - Model: ...{pred[-50:] if pred else 'None'}")
                self.logger.debug(f"{method_name.upper()} Sample {i} - Ground truth: {gt}")
        else:
            predictions = []

        if not predictions:
            return 0.0

        self.results_saver.save_detailed_validation_results(
            questions,
            ground_truths,
            predictions,
            trial_number,
            self.val_task_docs,
            self.config.val_dataset,
            trial=trial,
            steering_method=method_name,
            layer_id=layer_id,
            hyperparams=hyperparams,
        )

        task_docs = valid_samples[: len(predictions)] if valid_samples else []

        benchmark_metrics = metrics.evaluate_benchmark_performance(
            predictions, ground_truths, self.config.val_dataset, task_docs=task_docs
        )

        return benchmark_metrics.get("accuracy", 0.0)

    def _evaluate_with_steering_evaluator(
        self,
        steering_instance: Any,
        layer_id: int,
        hyperparams: dict[str, Any],
        evaluator_type: str,
    ) -> float:
        """Evaluate steering using the shared steering evaluator (refusal or personalization)."""
        if self.steering_evaluator is None:
            self.logger.warning(f"No steering evaluator setup for {evaluator_type} evaluation")
            return 0.0
        
        prompts = self.steering_evaluator.get_prompts()
        if not prompts:
            return 0.0
        
        strength = hyperparams["steering_alpha"]
        responses = self.generation_helper.generate_with_steering_batched(
            steering_instance, prompts, strength, layer_id
        )
        results = self.steering_evaluator.evaluate_responses(responses)
        return results.get("score", results.get("compliance_rate", 0.0))

    def _evaluate_refusal(
        self, steering_instance: Any, method_name: str, layer_id: int, hyperparams: dict[str, Any]
    ) -> float:
        """Evaluate steering using shared refusal evaluator."""
        return self._evaluate_with_steering_evaluator(steering_instance, layer_id, hyperparams, "refusal")

    def _evaluate_personalization(
        self, steering_instance: Any, method_name: str, layer_id: int, hyperparams: dict[str, Any]
    ) -> float:
        """Evaluate steering using shared personalization evaluator."""
        return self._evaluate_with_steering_evaluator(steering_instance, layer_id, hyperparams, "personalization")

    def evaluate_personalization_metrics(self, responses: list[str]) -> dict[str, Any]:
        """Evaluate personalization metrics for a list of responses using the shared evaluator."""
        if self.steering_evaluator is None:
            self.logger.warning("No steering evaluator for personalization metrics")
            return {"accuracy": 0.0, "score": 0.0}
        
        if not responses:
            return {"accuracy": 0.0, "score": 0.0}
        
        results = self.steering_evaluator.evaluate_responses(responses)
        score = results.get("score", 0.0)
        
        return {
            "accuracy": score,
            "score": score,
            "trait": self.config.trait,
            "num_responses": len(responses),
            "evaluation_method": "personalization_evaluator",
        }

    def evaluate_probe_metrics(self, probe, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Evaluate probe metrics."""
        y_pred = probe.predict(X_test)
        y_pred_proba = probe.predict_proba(X_test)[:, 1]

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
        }
