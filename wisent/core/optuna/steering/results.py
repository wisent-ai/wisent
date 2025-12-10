"""
Result saving and metadata utilities for the steering optimization pipeline.
"""

import json
import logging
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import optuna

from wisent.core.optuna.steering import metrics

from .config import OptimizationConfig


logger = logging.getLogger(__name__)


class ResultsSaver:
    """Handles saving optimization results and metadata."""

    def __init__(self, config: OptimizationConfig, run_dir: Path, run_timestamp: str):
        self.config = config
        self.run_dir = run_dir
        self.run_timestamp = run_timestamp
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def create_experiment_metadata(
        self, trial=None, steering_method: str = None, layer_id: int = None, hyperparams: dict = None
    ) -> dict[str, Any]:
        """Create comprehensive experiment metadata for detailed results."""
        metadata = {
            "trial_info": {
                "trial_number": trial.number if trial else None,
                "trial_params": dict(trial.params) if trial else {},
                "trial_state": str(getattr(trial, "state", "RUNNING")) if trial else None,
            },
            "model_config": {
                "model_name": self.config.model_name,
                "device": self.config.device,
            },
            "dataset_config": {
                "train_dataset": self.config.train_dataset,
                "val_dataset": self.config.val_dataset,
                "test_dataset": self.config.test_dataset,
                "train_limit": self.config.train_limit,
                "val_limit": self.config.val_limit,
                "test_limit": self.config.test_limit,
                "contrastive_pairs_limit": self.config.contrastive_pairs_limit,
            },
            "steering_config": {
                "steering_method": steering_method,
                "layer_id": layer_id,
                "hyperparams": hyperparams or {},
                "layer_search_range": self.config.layer_search_range,
                "probe_type": self.config.probe_type,
                "available_steering_methods": self.config.steering_methods,
            },
            "optimization_config": {
                "study_name": self.config.study_name,
                "sampler": self.config.sampler,
                "pruner": self.config.pruner,
                "n_trials": self.config.n_trials,
                "n_startup_trials": self.config.n_startup_trials,
            },
            "generation_config": {
                "batch_size": self.config.batch_size,
                "max_length": self.config.max_length,
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "do_sample": self.config.do_sample,
            },
            "run_info": {
                "timestamp": datetime.now().isoformat(),
                "run_dir": str(self.run_dir),
                "output_dir": self.config.output_dir,
                "cache_dir": self.config.cache_dir,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            },
            "wandb_config": {
                "use_wandb": self.config.use_wandb,
                "wandb_project": self.config.wandb_project,
            }
            if hasattr(self.config, "use_wandb")
            else {},
        }

        return metadata

    def save_detailed_validation_results(
        self,
        questions: list[str],
        ground_truths: list[str],
        predictions: list[str],
        trial_number: int,
        val_task_docs: list[dict],
        val_dataset: str,
        trial=None,
        steering_method: str = None,
        layer_id: int = None,
        hyperparams: dict = None,
    ):
        """Save detailed validation results to JSON file with experiment metadata."""
        detailed_results = []

        eval_results = metrics.evaluate_benchmark_performance(
            predictions, ground_truths, task_name=val_dataset, task_docs=val_task_docs
        )

        eval_details = eval_results.get("evaluation_details", [])

        for i, (question, correct_answer, model_answer) in enumerate(zip(questions, ground_truths, predictions)):
            is_correct = eval_details[i]["is_correct"] if i < len(eval_details) else False

            result_entry = {
                "row": i,
                "question": question,
                "correct_answer": correct_answer,
                "model_answer": model_answer,
                "is_correct": is_correct,
                "evaluation_method": eval_results.get("evaluation_method", "unknown"),
            }

            if self._should_use_multiple_choice_evaluation():
                mc_fields = self._extract_mc_fields(question, correct_answer, model_answer)
                result_entry.update(mc_fields)

            detailed_results.append(result_entry)

        experiment_metadata = self.create_experiment_metadata(trial, steering_method, layer_id, hyperparams)
        final_results = {"experiment_metadata": experiment_metadata, "results": detailed_results}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_detailed_results_trial_{trial_number:03d}_{timestamp}.json"
        filepath = self.run_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved detailed validation results to: {filename}")

    def save_detailed_test_results(
        self,
        questions: list[str],
        ground_truths: list[str],
        baseline_predictions: list[str],
        steered_predictions: list[str],
        test_task_docs: list[dict],
        test_dataset: str,
        best_trial=None,
        best_params: dict = None,
        layer_id: int = None,
        steering_method: str = None,
    ) -> str:
        """Save detailed test results to JSON file with both baseline and steered answers."""
        detailed_results = []

        baseline_eval_results = metrics.evaluate_benchmark_performance(
            baseline_predictions, ground_truths, task_name=test_dataset, task_docs=test_task_docs
        )
        steered_eval_results = metrics.evaluate_benchmark_performance(
            steered_predictions, ground_truths, task_name=test_dataset, task_docs=test_task_docs
        )

        baseline_details = baseline_eval_results.get("evaluation_details", [])
        steered_details = steered_eval_results.get("evaluation_details", [])

        for i, (question, correct_answer, baseline_answer, steered_answer) in enumerate(
            zip(questions, ground_truths, baseline_predictions, steered_predictions)
        ):
            is_baseline_correct = baseline_details[i]["is_correct"] if i < len(baseline_details) else False
            is_correct = steered_details[i]["is_correct"] if i < len(steered_details) else False

            result_entry = {
                "row": i,
                "question": question,
                "correct_answer": correct_answer,
                "baseline_model_answer": baseline_answer,
                "model_answer": steered_answer,
                "is_baseline_correct": is_baseline_correct,
                "is_correct": is_correct,
                "evaluation_method": steered_eval_results.get("evaluation_method", "unknown"),
            }

            if self._should_use_multiple_choice_evaluation():
                mc_fields = self._extract_mc_fields(question, correct_answer, steered_answer)
                mc_fields["baseline_model_selected_letter"] = self._extract_letter_from_answer(baseline_answer)
                result_entry.update(mc_fields)

            detailed_results.append(result_entry)

        experiment_metadata = self.create_experiment_metadata(
            trial=best_trial,
            steering_method=steering_method or best_params.get("steering_method") if best_params else None,
            layer_id=layer_id,
            hyperparams=best_params,
        )

        final_results = {"experiment_metadata": experiment_metadata, "results": detailed_results}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_detailed_results_{timestamp}.json"
        filepath = self.run_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved detailed test results to: {filename}")
        return filename

    def save_all_trials_metrics(self, study: optuna.Study, baseline_accuracy: float):
        """Save all trials' quality metrics and scores to a central JSON file."""
        trials_data = {
            "model": self.config.model_name,
            "task": self.config.val_dataset,
            "timestamp": self.run_timestamp,
            "baseline_accuracy": baseline_accuracy,
            "n_trials": len(study.trials),
            "trials": []
        }
        
        for trial in study.trials:
            trial_entry = {
                "trial_number": trial.number,
                "validation_accuracy": trial.value,
                "baseline_accuracy": trial.user_attrs.get("baseline_accuracy", baseline_accuracy),
                "steering_delta": trial.user_attrs.get("steering_delta", (trial.value - baseline_accuracy) if trial.value else None),
                "state": str(trial.state),
                "params": trial.params,
                "quality_metrics": trial.user_attrs.get("quality_metrics"),
                "early_rejected": trial.user_attrs.get("early_rejected", False),
            }
            trials_data["trials"].append(trial_entry)
        
        metrics_path = self.run_dir / f"all_trials_metrics_{self.run_timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(trials_data, f, indent=2)
        
        self.logger.info(f"Saved all trials metrics to: {metrics_path}")

    def save_reproducibility_bundle(self, study: optuna.Study, final_results: dict[str, Any]):
        """Save complete reproducibility bundle."""
        study_path = self.run_dir / f"optuna_study_{self.run_timestamp}.db"
        study.study_name = str(study_path)

        config_path = self.run_dir / f"config_{self.run_timestamp}.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        results_path = self.run_dir / f"final_results_{self.run_timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2, default=str)

        best_config = {
            "best_params": study.best_trial.params,
            "best_value": study.best_trial.value,
            "model_name": self.config.model_name,
            "random_seed": self.config.seed,
            "commit_hash": self._get_git_commit_hash(),
            "timestamp": self.run_timestamp,
        }

        best_config_path = self.run_dir / f"best_configuration_{self.run_timestamp}.json"
        with open(best_config_path, "w") as f:
            json.dump(best_config, f, indent=2)

        trials_df = study.trials_dataframe()
        trials_path = self.run_dir / f"study_trials_{self.run_timestamp}.csv"
        trials_df.to_csv(trials_path, index=False)

        self.logger.info(f"Reproducibility bundle saved to: {self.run_dir}")
        self.logger.info(f"Study database: {study_path}")
        self.logger.info(f"Configuration: {config_path}")
        self.logger.info(f"Results: {results_path}")
        self.logger.info(f"Best config: {best_config_path}")

        safetensors_path = self.run_dir / "best_steering_vector.safetensors"
        pt_path = self.run_dir / "best_steering_vector.pt"

        if safetensors_path.exists():
            self.logger.info(f"Steering vector: {safetensors_path.name}")
        elif pt_path.exists():
            self.logger.info(f"Steering vector: {pt_path.name}")

    def _get_git_commit_hash(self) -> Optional[str]:
        """Get current git commit hash for reproducibility."""
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None

    def _should_use_multiple_choice_evaluation(self) -> bool:
        """Determine if we should use multiple choice evaluation for this dataset."""
        return self.config.test_dataset.lower() in ["truthfulqa_mc1", "truthfulqa", "mmlu"]

    def _extract_letter_from_answer(self, answer: str) -> str:
        """Extract the first A-E letter from an answer string."""
        match = re.search(r"\b([A-E])\b", answer.upper())
        return match.group(1) if match else "?"

    def _extract_mc_fields(self, question: str, correct_answer: str, model_answer: str) -> dict:
        """Extract multiple choice fields from question and answers."""
        available_answers = []
        choice_pattern = r"([A-E])\.\s+(.+?)(?=\n[A-E]\.|$)"
        matches = re.findall(choice_pattern, question, re.MULTILINE | re.DOTALL)
        for letter, choice_text in matches:
            available_answers.append(f"{letter}. {choice_text.strip()}")
        return {
            "available_answers": available_answers,
            "correct_choice_letter": correct_answer,
            "model_selected_letter": self._extract_letter_from_answer(model_answer),
        }
