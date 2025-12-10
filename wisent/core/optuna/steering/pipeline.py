"""
Core Optimization Pipeline for steering vector hyperparameter search.

This module provides the main OptimizationPipeline class that orchestrates
the Optuna-based hyperparameter optimization process.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from safetensors.torch import save_file as safetensors_save
from tqdm import tqdm

from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations.core.atoms import ActivationAggregationStrategy
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.contrastive_pairs.diagnostics import run_vector_quality_diagnostics, VectorQualityConfig
from wisent.core.errors import (
    InvalidLayerIdError,
    MissingParameterError,
    ModelArchitectureUnknownError,
    OptimizationError,
    SteeringMethodUnknownError,
)
from wisent.core.models.core.atoms import SteeringPlan
from wisent.core.models.wisent_model import WisentModel
from wisent.core.optuna.steering import metrics
from wisent.core.steering_methods import SteeringMethodRegistry, get_steering_method
from wisent.core.task_interface import get_task
from wisent.core.utils.device import empty_device_cache, resolve_device

from .cache import ActivationCache
from .config import OptimizationConfig
from .evaluation import EvaluationHelper
from .generation import GenerationHelper
from .results import ResultsSaver
from .tracking import WandBTracker


logger = logging.getLogger(__name__)


class OptimizationPipeline:
    """Main optimization pipeline using Optuna for hyperparameter search."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache = ActivationCache(config.cache_dir)

        self.wandb_tracker = WandBTracker(config, enabled=config.use_wandb)
        if config.use_wandb:
            self.wandb_tracker.init()

        self.model = None
        self.tokenizer = None
        self._wisent_model: Optional[WisentModel] = None
        self._activation_collector: Optional[ActivationCollector] = None
        self.train_samples = None
        self.val_samples = None
        self.test_samples = None
        self.train_task_docs = None
        self.val_task_docs = None
        self.test_task_docs = None
        self.tokenization_config = {
            "max_length": config.max_length,
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        
        self.evaluator_type = "task"
        self.steering_evaluator = None
        self.baseline_accuracy = 0.0
        
        self._generation_helper: Optional[GenerationHelper] = None
        self._results_saver: Optional[ResultsSaver] = None
        self._evaluation_helper: Optional[EvaluationHelper] = None

    def run_optimization(self) -> dict[str, Any]:
        """Run the complete optimization pipeline."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING OPTIMIZATION PIPELINE WITH OPTUNA")
        self.logger.info("=" * 80)

        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.run_timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Run directory: {self.run_dir}")

        self._setup_experiment()
        self._compute_baseline_accuracy()
        
        study = self._create_optuna_study()
        study.optimize(self._objective_function, n_trials=self.config.n_trials)
        best_trial = study.best_trial
        
        self._results_saver.save_all_trials_metrics(study, self.baseline_accuracy)
        
        final_results = self._final_evaluation(best_trial)
        self._results_saver.save_reproducibility_bundle(study, final_results)

        self.wandb_tracker.log_final_results(study, final_results)

        self.logger.info("Optimization completed successfully!")
        return final_results

    def _setup_experiment(self):
        """Setup model, tokenizer, and load datasets."""
        self.logger.info("Setting up experiment...")

        self._wisent_model = WisentModel(model_name=self.config.model_name, device=str(self.device))
        self.model = self._wisent_model.hf_model
        self.tokenizer = self._wisent_model.tokenizer
        self.model.eval()
        self.tokenizer.padding_side = "left"
        
        self._activation_collector = ActivationCollector(
            model=self._wisent_model,
            store_device="cpu",
            dtype=torch.float32
        )

        self._setup_evaluator()

        if self.config.trait:
            self._setup_synthetic_data()
        else:
            self._setup_task_data()

        if not self.config.trait:
            self._precache_activations()

        self._generation_helper = GenerationHelper(self._wisent_model, self.config)
        self._results_saver = ResultsSaver(self.config, self.run_dir, self.run_timestamp)
        self._evaluation_helper = EvaluationHelper(self.config, self._generation_helper, self._results_saver)
        self._evaluation_helper.set_context(
            self.evaluator_type,
            self.steering_evaluator,
            self.val_samples,
            self.val_task_docs,
        )

    def _setup_evaluator(self):
        """Setup the evaluator based on --task argument."""
        task = (self.config.train_dataset or "").lower()
        
        if task == "refusal":
            evaluator_type = "refusal"
        elif task == "personalization":
            if not self.config.trait:
                raise ValueError("--trait is required when --task personalization")
            evaluator_type = "personalization"
        elif task == "custom":
            if not getattr(self.config, 'custom_evaluator', None):
                raise ValueError("--custom-evaluator is required when --task custom")
            evaluator_type = "custom"
        else:
            evaluator_type = "task"
        
        self.evaluator_type = evaluator_type
        self.logger.info(f"Using evaluator: {evaluator_type}")
        
        if evaluator_type == "refusal":
            self._setup_refusal_evaluator()
        elif evaluator_type == "personalization":
            self._setup_personalization_evaluator()
        elif evaluator_type == "custom":
            self._setup_custom_evaluator()

    def _setup_refusal_evaluator(self):
        """Setup refusal evaluator for compliance/refusal measurement."""
        self._setup_steering_evaluator("refusal")
        self.eval_prompts = self.steering_evaluator.get_prompts()
        self.logger.info(f"Loaded {len(self.eval_prompts)} evaluation prompts for refusal evaluation")

    def _setup_personalization_evaluator(self):
        """Setup personalization evaluator for trait steering."""
        self._setup_steering_evaluator("personalization")
        self.personalization_test_prompts = self.steering_evaluator.get_prompts()
        self.logger.info(f"Setup personalization evaluator for trait: {self.config.trait}")

    def _setup_steering_evaluator(self, evaluator_type: str):
        """Setup a steering evaluator (refusal or personalization)."""
        from wisent.core.evaluators.steering_evaluators import RefusalEvaluator, PersonalizationEvaluator, EvaluatorConfig
        config = EvaluatorConfig(
            evaluator_type=evaluator_type,
            trait=self.config.trait if evaluator_type == "personalization" else None,
            eval_prompts_path=self.config.eval_prompts,
            num_eval_prompts=self.config.num_eval_prompts,
        )
        if evaluator_type == "refusal":
            self.steering_evaluator = RefusalEvaluator(config, self.config.model_name)
        else:
            self.steering_evaluator = PersonalizationEvaluator(
                config, self.config.model_name, wisent_model=self._wisent_model
            )

    def _setup_custom_evaluator(self):
        """Setup custom evaluator from user-provided module/function."""
        from wisent.core.evaluators.custom import create_custom_evaluator
        
        custom_kwargs = {}
        if getattr(self.config, 'custom_evaluator_kwargs', None):
            if isinstance(self.config.custom_evaluator_kwargs, str):
                custom_kwargs = json.loads(self.config.custom_evaluator_kwargs)
            else:
                custom_kwargs = self.config.custom_evaluator_kwargs
        
        self.custom_evaluator = create_custom_evaluator(
            self.config.custom_evaluator, **custom_kwargs
        )
        
        if self.config.trait:
            from wisent.core.evaluators.steering_evaluators import PersonalizationEvaluator
            self.custom_test_prompts = PersonalizationEvaluator._generate_test_prompts(
                self.config.num_eval_prompts or 30
            )
        else:
            self.custom_test_prompts = [
                "Tell me about yourself.",
                "What do you think about artificial intelligence?",
                "How would you solve world hunger?",
                "Explain quantum computing in simple terms.",
                "What's the meaning of life?",
            ]
        
        self.logger.info(f"Setup custom evaluator: {self.config.custom_evaluator}")

    def _setup_synthetic_data(self):
        """Setup synthetic contrastive pairs for trait-based optimization."""
        self.logger.info(f"Generating synthetic pairs for trait: {self.config.trait}")
        
        from wisent.core.cli.agent.generate_synthetic_pairs import generate_synthetic_pairs
        
        pair_set, report = generate_synthetic_pairs(
            model=self._wisent_model,
            prompt=self.config.trait,
            time_budget=5.0,
            num_pairs=self.config.contrastive_pairs_limit,
            verbose=False,
        )
        
        self.synthetic_pairs = []
        for pair in pair_set.pairs:
            self.synthetic_pairs.append({
                "prompt": pair.prompt,
                "positive_response": pair.positive_response.model_response,
                "negative_response": pair.negative_response.model_response,
            })
        
        self.train_samples = self.synthetic_pairs
        self.val_samples = []
        self.test_samples = []
        
        self.train_task_docs = self.train_samples
        self.val_task_docs = self.val_samples
        self.test_task_docs = self.test_samples
        
        self.logger.info(f"Generated {len(self.synthetic_pairs)} synthetic pairs")

    def _setup_task_data(self):
        """Setup task-based data for benchmark optimization."""
        from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
        
        loader = LMEvalDataLoader()
        result = loader._load_one_task(
            task_name=self.config.train_dataset,
            split_ratio=0.8,
            seed=42,
            limit=self.config.train_limit + self.config.val_limit + self.config.test_limit,
            training_limit=None,
            testing_limit=None,
        )
        
        self.train_pairs = result["train_qa_pairs"]
        self.test_pairs = result["test_qa_pairs"]
        
        all_test = self.test_pairs.pairs
        val_count = min(self.config.val_limit, len(all_test) // 2)
        self.val_pairs_list = all_test[:val_count]
        self.test_pairs_list = all_test[val_count:val_count + self.config.test_limit]
        
        self.train_samples = [
            {
                "prompt": p.prompt, 
                "positive": p.positive_response.model_response, 
                "negative": p.negative_response.model_response,
                "metadata": p.metadata if hasattr(p, 'metadata') else None,
            }
            for p in self.train_pairs.pairs
        ]
        self.val_samples = [
            {
                "prompt": p.prompt, 
                "positive": p.positive_response.model_response, 
                "negative": p.negative_response.model_response,
                "metadata": p.metadata if hasattr(p, 'metadata') else None,
            }
            for p in self.val_pairs_list
        ]
        self.test_samples = [
            {
                "prompt": p.prompt, 
                "positive": p.positive_response.model_response, 
                "negative": p.negative_response.model_response,
                "metadata": p.metadata if hasattr(p, 'metadata') else None,
            }
            for p in self.test_pairs_list
        ]

        self.train_task_docs = self.train_samples
        self.val_task_docs = self.val_samples
        self.test_task_docs = self.test_samples

        self.logger.info(
            f"Loaded {len(self.train_samples)} train, {len(self.val_samples)} val, {len(self.test_samples)} test samples"
        )

    def _precache_activations(self):
        """Pre-cache activations for all layers and splits to improve efficiency."""
        self.logger.info("Pre-caching activations for efficiency...")

        layer_range = range(self.config.layer_search_range[0], self.config.layer_search_range[1] + 1)
        splits_data = [("train", self.train_samples), ("val", self.val_samples), ("test", self.test_samples)]

        for split_name, samples in splits_data:
            for layer_id in layer_range:
                if not self.cache.has_cached_activations(split_name, layer_id, self.tokenization_config):
                    self.logger.info(f"Caching activations for {split_name} split, layer {layer_id}")
                    dataset_name = {
                        "train": self.config.train_dataset,
                        "val": self.config.val_dataset,
                        "test": self.config.test_dataset,
                    }[split_name]
                    activations, labels = self._create_probe_data(samples, layer_id, dataset_name)
                    self.cache.save_activations(activations, labels, split_name, layer_id, self.tokenization_config)
                else:
                    self.logger.info(f"Activations already cached for {split_name} split, layer {layer_id}")

    def _compute_baseline_accuracy(self):
        """Compute baseline accuracy on validation set (no steering) once for all trials."""
        self.logger.info("Computing baseline accuracy (no steering)...")
        
        if self.evaluator_type in ("refusal", "personalization"):
            self.baseline_accuracy = self._compute_baseline_with_evaluator()
        else:
            self.baseline_accuracy = self._compute_baseline_task()
        
        self.logger.info(f"Baseline accuracy: {self.baseline_accuracy:.4f}")

    def _compute_baseline_with_evaluator(self) -> float:
        """Compute baseline using the steering evaluator (refusal or personalization)."""
        if self.steering_evaluator is None:
            self.logger.warning(f"No steering evaluator setup for {self.evaluator_type} baseline")
            return 0.0
        
        prompts = self.steering_evaluator.get_prompts()
        if not prompts:
            return 0.0
        
        responses = self._generation_helper.generate_baseline_batched(prompts)
        
        if self.evaluator_type == "personalization":
            self.steering_evaluator._baseline_responses = responses
            return 0.0
        
        results = self.steering_evaluator.evaluate_responses(responses)
        return results.get("score", results.get("compliance_rate", 0.0))

    def _compute_baseline_task(self) -> float:
        """Compute baseline for task-based evaluation."""
        questions = []
        ground_truths = []
        
        for sample in self.val_samples:
            if "prompt" in sample and "positive" in sample:
                questions.append(sample["prompt"])
                ground_truths.append(sample["positive"])
            else:
                try:
                    task = get_task(self.config.val_dataset)
                    extractor = task.get_extractor()
                    qa_pair = extractor.extract_qa_pair(sample, task)
                    if not qa_pair:
                        continue
                    questions.append(qa_pair["formatted_question"])
                    ground_truths.append(qa_pair["correct_answer"])
                except Exception:
                    continue
        
        if not questions:
            self.logger.warning("No valid QA pairs for baseline computation")
            return 0.0
        
        predictions = self._generation_helper.generate_baseline_batched(questions)
        task_docs = self.val_samples[:len(predictions)]
        baseline_metrics = metrics.evaluate_benchmark_performance(
            predictions, ground_truths, self.config.val_dataset, task_docs=task_docs
        )
        return baseline_metrics.get("accuracy", 0.0)

    def _create_optuna_study(self) -> optuna.Study:
        """Create and configure Optuna study."""
        if self.config.sampler == "TPE":
            sampler = TPESampler(
                seed=self.config.seed,
                n_startup_trials=self.config.n_startup_trials,
            )
        else:
            sampler = optuna.samplers.RandomSampler(seed=self.config.seed)

        if self.config.pruner == "MedianPruner":
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        elif self.config.pruner == "SuccessiveHalving":
            pruner = SuccessiveHalvingPruner()
        else:
            pruner = optuna.pruners.NopPruner()

        study = optuna.create_study(
            study_name=f"{self.config.study_name}_{self.run_timestamp}",
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        return study

    def _objective_function(self, trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization."""
        layer_id = trial.suggest_int(
            "layer_id",
            self.config.layer_search_range[0],
            self.config.layer_search_range[1],
        )

        steering_method = trial.suggest_categorical("steering_method", self.config.steering_methods)
        steering_alpha = trial.suggest_float("steering_alpha", 0.1, 2.0, log=True)

        hyperparams = {
            "layer_id": layer_id,
            "steering_method": steering_method,
            "steering_alpha": steering_alpha,
        }

        self.logger.info(f"Trial {trial.number}: layer={layer_id}, method={steering_method}, alpha={steering_alpha:.3f}")

        try:
            steering_instance, quality_metrics = self._train_steering_method(
                trial, steering_method, layer_id, hyperparams, early_reject=self.config.enable_early_rejection
            )
            
            if quality_metrics:
                trial.set_user_attr("quality_metrics", quality_metrics)
            
            if steering_instance is None:
                trial.set_user_attr("early_rejected", True)
                return 0.0

            val_accuracy = self._evaluation_helper.evaluate_steering_on_validation(
                steering_instance, steering_method, layer_id, hyperparams, trial.number, trial
            )

            trial.set_user_attr("baseline_accuracy", self.baseline_accuracy)
            trial.set_user_attr("steering_delta", val_accuracy - self.baseline_accuracy)

            self.wandb_tracker.log_trial(trial, {"validation_accuracy": val_accuracy})

            self.logger.info(
                f"Trial {trial.number}: val_accuracy={val_accuracy:.4f}, "
                f"baseline={self.baseline_accuracy:.4f}, delta={val_accuracy - self.baseline_accuracy:+.4f}"
            )

            return val_accuracy

        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0

    def _train_steering_method(
        self,
        trial: optuna.Trial,
        method_name: str,
        layer_id: int,
        hyperparams: dict[str, Any],
        early_reject: bool = True,
    ) -> tuple[Any, Optional[dict]]:
        """Train steering method and return trained instance with quality metrics."""
        if self.config.trait:
            contrastive_pairs = self._create_synthetic_contrastive_pairs(layer_id, self.config.contrastive_pairs_limit)
        else:
            contrastive_pairs = self._create_contrastive_pairs(
                self.train_samples, layer_id, self.config.train_dataset, self.config.contrastive_pairs_limit
            )
        
        quality_metrics = None
        if early_reject:
            try:
                quality_config = VectorQualityConfig(
                    min_snr=self.config.early_rejection_snr_threshold,
                    min_cv_score=self.config.early_rejection_cv_threshold,
                )
                quality_report = run_vector_quality_diagnostics(contrastive_pairs, quality_config)
                
                quality_metrics = {
                    "snr": quality_report.snr,
                    "cv_score_mean": quality_report.cv_score_mean,
                    "overall_quality": quality_report.overall_quality,
                    "convergence": getattr(quality_report, 'convergence', None),
                    "held_out_transfer": getattr(quality_report, 'held_out_transfer', None),
                    "cv_classification_accuracy": getattr(quality_report, 'cv_classification_accuracy', None),
                    "cohens_d": getattr(quality_report, 'cohens_d', None),
                    "pca_pc1_variance": getattr(quality_report, 'pca_pc1_variance', None),
                    "silhouette_score": getattr(quality_report, 'silhouette_score', None),
                }
                
                if quality_report.overall_quality == "poor":
                    if quality_report.snr < self.config.early_rejection_snr_threshold or \
                       quality_report.cv_score_mean < self.config.early_rejection_cv_threshold:
                        self.logger.info(
                            f"Early rejecting trial (layer={layer_id}): SNR={quality_report.snr:.2f}, "
                            f"CV={quality_report.cv_score_mean:.3f}, quality={quality_report.overall_quality}"
                        )
                        return None, quality_metrics
                        
            except Exception as e:
                self.logger.debug(f"Quality check failed, continuing without early rejection: {e}")

        if not SteeringMethodRegistry.validate_method(method_name):
            raise SteeringMethodUnknownError(method=method_name)
        
        steering_instance = get_steering_method(method_name, device=self.device, **hyperparams)
        trained_vectors = steering_instance.train(contrastive_pairs)
        steering_instance._trained_vectors = trained_vectors.to_dict() if hasattr(trained_vectors, 'to_dict') else dict(trained_vectors)
        
        return steering_instance, quality_metrics

    def _add_activations_to_pair_set(self, pair_set: ContrastivePairSet, layer_id: int) -> ContrastivePairSet:
        """Extract and attach activations to all pairs in a ContrastivePairSet."""
        if not pair_set.pairs:
            return pair_set

        all_texts = []
        text_to_pair_mapping = []

        for pair_idx, pair in enumerate(pair_set.pairs):
            pos_text = f"{pair.prompt} {pair.positive_response.model_response}"
            neg_text = f"{pair.prompt} {pair.negative_response.model_response}"
            all_texts.extend([pos_text, neg_text])
            text_to_pair_mapping.extend([(pair_idx, "positive"), (pair_idx, "negative")])

        layer_name = str(layer_id + 1)
        all_activations = self._activation_collector.collect_batched(
            texts=all_texts,
            layers=[layer_name],
            aggregation=ActivationAggregationStrategy.LAST_TOKEN,
            batch_size=self.config.batch_size,
            show_progress=False,
        )

        pair_activations = {}
        for text_idx, (pair_idx, response_type) in enumerate(text_to_pair_mapping):
            activation_dict = all_activations[text_idx]
            activation = activation_dict.get(layer_name)
            if activation is not None:
                activation = activation.unsqueeze(0)
            if pair_idx not in pair_activations:
                pair_activations[pair_idx] = {"positive": None, "negative": None}
            pair_activations[pair_idx][response_type] = {f"layer_{layer_id}": activation}

        updated_pairs = []
        for pair_idx, pair in enumerate(pair_set.pairs):
            if pair_idx in pair_activations:
                updated_pair = pair.with_activations(
                    positive=pair_activations[pair_idx]["positive"],
                    negative=pair_activations[pair_idx]["negative"]
                )
                updated_pairs.append(updated_pair)
            else:
                updated_pairs.append(pair)

        return ContrastivePairSet(name=pair_set.name, pairs=updated_pairs)

    def _create_contrastive_pairs(
        self, samples: list[dict], layer_id: int, dataset_name: str, limit: Optional[int] = None
    ) -> ContrastivePairSet:
        """Create contrastive pairs with activations for steering training."""
        contrastive_pairs = []
        task = get_task(dataset_name)
        extractor = task.get_extractor()

        samples_to_use = samples[:limit] if limit else samples

        for sample in samples_to_use:
            contrastive_pair = extractor.extract_contrastive_pair(sample, task)
            if contrastive_pair:
                self.logger.debug(f"Creating contrastive pair - Question: ...{contrastive_pair['question'][-50:]}")
                self.logger.debug(
                    f"Creating contrastive pair - Correct: {contrastive_pair['correct_answer']}, Incorrect: {contrastive_pair['incorrect_answer']}"
                )
                pair = ContrastivePair(
                    prompt=contrastive_pair["question"],
                    positive_response=PositiveResponse(model_response=contrastive_pair["correct_answer"], label="1"),
                    negative_response=NegativeResponse(model_response=contrastive_pair["incorrect_answer"], label="0"),
                )
                contrastive_pairs.append(pair)

        pair_set = ContrastivePairSet(name=f"{dataset_name}_training", pairs=contrastive_pairs)
        return self._add_activations_to_pair_set(pair_set, layer_id)

    def _create_synthetic_contrastive_pairs(self, layer_id: int, limit: Optional[int] = None) -> ContrastivePairSet:
        """Create contrastive pairs from pre-generated synthetic pairs for trait mode."""
        if not hasattr(self, 'synthetic_pairs') or not self.synthetic_pairs:
            self.logger.warning("No synthetic pairs available")
            return ContrastivePairSet(name=f"synthetic_{self.config.trait}", pairs=[])
        
        pairs_to_use = self.synthetic_pairs[:limit] if limit else self.synthetic_pairs
        contrastive_pairs = []
        
        for pair_data in pairs_to_use:
            if isinstance(pair_data, dict):
                prompt = pair_data.get("prompt", pair_data.get("question", ""))
                positive_text = pair_data.get("positive_response", pair_data.get("positive", ""))
                negative_text = pair_data.get("negative_response", pair_data.get("negative", ""))
            elif hasattr(pair_data, 'prompt'):
                contrastive_pairs.append(pair_data)
                continue
            else:
                continue
            
            pair = ContrastivePair(
                prompt=prompt,
                positive_response=PositiveResponse(model_response=positive_text, label="1"),
                negative_response=NegativeResponse(model_response=negative_text, label="0"),
            )
            contrastive_pairs.append(pair)
        
        pair_set = ContrastivePairSet(name=f"synthetic_{self.config.trait}", pairs=contrastive_pairs)
        pair_set = self._add_activations_to_pair_set(pair_set, layer_id)
        self.logger.info(f"Created {len(pair_set.pairs)} synthetic contrastive pairs for layer {layer_id}")
        return pair_set

    def _create_probe_data(
        self, samples: list[dict], layer_id: int, dataset_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create contrastive probe training data for a specific layer."""
        self.logger.info(f"Creating probe data from {len(samples)} samples for {dataset_name} on layer {layer_id}")

        texts = []
        labels = []
        success_count = 0
        fail_count = 0

        for i, sample in enumerate(samples):
            try:
                if "prompt" in sample and "positive" in sample and "negative" in sample:
                    question = sample["prompt"]
                    correct_answer = sample["positive"]
                    incorrect_answer = sample["negative"]
                    success_count += 1
                else:
                    task = get_task(dataset_name)
                    extractor = task.get_extractor()
                    contrastive_pair = extractor.extract_contrastive_pair(sample, task)

                    if not contrastive_pair:
                        self.logger.debug(f"Sample {i + 1}: No contrastive pair extracted from keys: {list(sample.keys())}")
                        fail_count += 1
                        continue

                    question = contrastive_pair["question"]
                    correct_answer = contrastive_pair["correct_answer"]
                    incorrect_answer = contrastive_pair["incorrect_answer"]
                    success_count += 1

            except Exception as e:
                self.logger.error(f"Sample {i + 1}: Exception during contrastive pair extraction: {e}")
                fail_count += 1
                continue

            self.logger.debug(f"Contrastive pair - Question: ...{question[-50:]}")
            self.logger.debug(f"Contrastive pair - Correct: {correct_answer[:50]}..., Incorrect: {incorrect_answer[:50]}...")

            correct_text = f"{question} {correct_answer}"
            texts.append(correct_text)
            labels.append(1)

            incorrect_text = f"{question} {incorrect_answer}"
            texts.append(incorrect_text)
            labels.append(0)

        self.logger.info(
            f"Probe data creation: {success_count} successful, {fail_count} failed. Generated {len(texts)} texts."
        )

        if len(texts) == 0:
            return np.array([]), np.array([])

        layer_name = str(layer_id + 1)
        all_activations = self._activation_collector.collect_batched(
            texts=texts,
            layers=[layer_name],
            aggregation=ActivationAggregationStrategy.LAST_TOKEN,
            batch_size=self.config.batch_size,
            show_progress=True,
        )

        activations_list = []
        for act_dict in all_activations:
            if layer_name in act_dict:
                activations_list.append(act_dict[layer_name].numpy())

        if not activations_list:
            return np.array([]), np.array([])

        return np.stack(activations_list), np.array(labels)

    def _final_evaluation(self, best_trial: optuna.Trial) -> dict[str, Any]:
        """Run final evaluation on test split with best configuration."""
        self.logger.info("Running final evaluation with best configuration...")

        if hasattr(best_trial, "params") and best_trial.params:
            best_params = best_trial.params
        elif hasattr(best_trial, "_params"):
            best_params = best_trial._params
        else:
            raise MissingParameterError(params=["trial.params"])
        layer_id = best_params["layer_id"]

        self.logger.info(f"Best configuration: {best_params}")

        steering_method = best_params.get("steering_method", "caa")
        steering_instance, _ = self._train_steering_method(
            best_trial, steering_method, layer_id, best_params, early_reject=False
        )

        if steering_instance and hasattr(steering_instance, "save_steering_vector"):
            pt_path = self.run_dir / "best_steering_vector.pt"
            safetensors_path = self.run_dir / "best_steering_vector.safetensors"
            self._save_steering_vector_dual_format(steering_instance, pt_path, safetensors_path)

        probe = None
        if self.evaluator_type not in ["personalization", "refusal"]:
            from sklearn.linear_model import LogisticRegression
            try:
                X_train, y_train = self.cache.load_activations("train", layer_id, self.tokenization_config)
                probe = LogisticRegression(C=1.0, random_state=self.config.seed, max_iter=1000)
                probe.fit(X_train, y_train)
            except FileNotFoundError:
                self.logger.warning("Could not load cached activations for probe training")

        self.logger.info("Generating baseline predictions...")
        baseline_predictions, test_ground_truths, test_questions, test_task_docs = self._generation_helper.generate_test_predictions(
            None, None, layer_id, 0.0, self.test_samples, self.evaluator_type,
            self.steering_evaluator, self.config.trait, self.config.test_dataset
        )

        self.logger.info("Generating steered predictions...")
        method_name = best_params.get("steering_method", "caa")
        strength = best_params["steering_alpha"]

        steered_predictions, _, _, _ = self._generation_helper.generate_test_predictions(
            steering_instance, method_name, layer_id, strength, self.test_samples, self.evaluator_type,
            self.steering_evaluator, self.config.trait, self.config.test_dataset
        )

        if test_questions and test_ground_truths and baseline_predictions and steered_predictions:
            self._results_saver.save_detailed_test_results(
                test_questions,
                test_ground_truths,
                baseline_predictions,
                steered_predictions,
                self.test_task_docs,
                self.config.test_dataset,
                best_trial=best_trial,
                best_params=best_params,
                layer_id=layer_id,
                steering_method=method_name,
            )

        if self.evaluator_type == "personalization":
            baseline_benchmark_metrics = self._evaluation_helper.evaluate_personalization_metrics(baseline_predictions)
            steered_benchmark_metrics = self._evaluation_helper.evaluate_personalization_metrics(steered_predictions)
        else:
            baseline_benchmark_metrics = metrics.evaluate_benchmark_performance(
                baseline_predictions, test_ground_truths, self.config.test_dataset, task_docs=test_task_docs
            )
            steered_benchmark_metrics = metrics.evaluate_benchmark_performance(
                steered_predictions, test_ground_truths, self.config.test_dataset, task_docs=test_task_docs
            )

        test_probe_metrics = {"auc": 0.5, "accuracy": 0.0}
        if probe is not None and self.evaluator_type not in ["personalization", "refusal"]:
            try:
                X_test, y_test = self.cache.load_activations("test", layer_id, self.tokenization_config)
                test_probe_metrics = self._evaluation_helper.evaluate_probe_metrics(probe, X_test, y_test)
            except FileNotFoundError:
                self.logger.warning("Could not load cached test activations for probe evaluation")

        accuracy_improvement = steered_benchmark_metrics.get("accuracy", 0.0) - baseline_benchmark_metrics.get("accuracy", 0.0)

        final_results = {
            "best_trial_params": best_params,
            "best_validation_score": getattr(best_trial, "value", None),
            "baseline_benchmark_metrics": baseline_benchmark_metrics,
            "steered_benchmark_metrics": steered_benchmark_metrics,
            "accuracy_improvement": accuracy_improvement,
            "test_probe_metrics": test_probe_metrics,
            "config": self.config.to_dict(),
            "num_test_samples": len(test_ground_truths),
        }

        self.logger.info("=" * 60)
        self.logger.info("FINAL TEST RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Baseline accuracy: {baseline_benchmark_metrics.get('accuracy', 0.0):.4f}")
        self.logger.info(f"Steered accuracy: {steered_benchmark_metrics.get('accuracy', 0.0):.4f}")
        self.logger.info(f"Improvement: {accuracy_improvement:+.4f}")
        if self.evaluator_type not in ["personalization", "refusal"]:
            self.logger.info(f"Probe AUC: {test_probe_metrics.get('auc', 0.5):.4f}")
        self.logger.info(f"Test samples: {len(test_ground_truths)}")
        
        quality_metrics = best_trial.user_attrs.get("quality_metrics") if hasattr(best_trial, "user_attrs") else None
        if quality_metrics:
            self.logger.info("-" * 60)
            self.logger.info("VECTOR QUALITY METRICS")
            self.logger.info("-" * 60)
            self.logger.info(f"Overall quality: {quality_metrics.get('overall_quality', 'N/A')}")
            if quality_metrics.get('snr') is not None:
                self.logger.info(f"Signal-to-noise ratio: {quality_metrics['snr']:.2f}")
            if quality_metrics.get('cv_score_mean') is not None:
                self.logger.info(f"Cross-validation score: {quality_metrics['cv_score_mean']:.3f}")
            final_results["quality_metrics"] = quality_metrics
        self.logger.info("=" * 60)

        return final_results

    def _save_steering_vector_dual_format(self, steering_instance: Any, pt_path: Path, safetensors_path: Path):
        """Save steering vector in both .pt and .safetensors formats."""
        try:
            if hasattr(steering_instance, '_trained_vectors') and steering_instance._trained_vectors:
                vectors_dict = {}
                for k, v in steering_instance._trained_vectors.items():
                    if isinstance(v, torch.Tensor):
                        vectors_dict[k] = v
                    elif isinstance(v, np.ndarray):
                        vectors_dict[k] = torch.from_numpy(v)
                
                if vectors_dict:
                    torch.save(vectors_dict, pt_path)
                    safetensors_save(vectors_dict, safetensors_path)
                    self.logger.info(f"Saved steering vector to {pt_path.name} and {safetensors_path.name}")
        except Exception as e:
            self.logger.warning(f"Failed to save steering vector: {e}")

    def evaluate_only(self, best_params: dict[str, Any]) -> dict[str, Any]:
        """Run evaluation only with provided parameters."""
        self.logger.info("Running evaluation-only mode with provided parameters")
        self.logger.info(f"Parameters: {best_params}")

        if self.model is None:
            self._setup_experiment()

        if not hasattr(self, "run_dir"):
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.output_dir / f"evaluate_only_{self.run_timestamp}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Evaluation directory: {self.run_dir}")

        from optuna.trial import FixedTrial

        complete_params = {
            "layer_id": best_params.get("layer_id", 15),
            "probe_type": best_params.get("probe_type", "logistic_regression"),
            "probe_c": best_params.get("probe_c", 1.0),
            "steering_method": best_params.get("steering_method", "caa"),
            "steering_alpha": best_params.get("steering_alpha", 0.5),
        }

        fixed_trial = FixedTrial(complete_params)

        if not hasattr(fixed_trial, "params"):
            fixed_trial.params = complete_params

        return self._final_evaluation(fixed_trial)

    @classmethod
    def from_saved_study(
        cls, study_path: str, config_path: Optional[str] = None, override_config: Optional[dict[str, Any]] = None
    ):
        """Create pipeline from saved study and optionally saved config."""
        if config_path:
            with open(config_path) as f:
                config_dict = json.load(f)
                if override_config:
                    config_dict.update(override_config)
                config = OptimizationConfig(**config_dict)
        else:
            config = OptimizationConfig(**(override_config or {}))

        study_name = Path(study_path).stem
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_path}")

        pipeline = cls(config)
        return pipeline, study

    def evaluate_on_dataset(
        self, best_params: dict[str, Any], dataset_name: str, dataset_limit: Optional[int] = None
    ) -> dict[str, Any]:
        """Evaluate best parameters on a different dataset."""
        original_test_dataset = self.config.test_dataset
        original_test_limit = self.config.test_limit

        self.config.test_dataset = dataset_name
        self.config.test_limit = dataset_limit or self.config.test_limit

        self.logger.info(f"Evaluating on {dataset_name} with {self.config.test_limit} samples")

        from . import data_utils
        self.test_samples = data_utils.load_dataset_samples(self.config.test_dataset, self.config.test_limit)

        results = self.evaluate_only(best_params)

        self.config.test_dataset = original_test_dataset
        self.config.test_limit = original_test_limit

        return results

    def cleanup_memory(self):
        """Clean up GPU/MPS memory."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.wandb_tracker.finish()

        empty_device_cache(self.device.type)

        import gc
        gc.collect()
