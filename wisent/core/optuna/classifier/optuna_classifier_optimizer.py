"""
Optuna-based classifier optimization for efficient hyperparameter search.

This module provides a modern, efficient optimization system that pre-generates
activations once and uses intelligent caching to avoid redundant training.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from wisent.core.classifier.classifier import Classifier
from wisent.core.utils import resolve_default_device, preferred_dtype
from wisent.core.errors import NoActivationDataError, ClassifierCreationError

from .activation_generator import ActivationData, ActivationGenerator, GenerationConfig
from .classifier_cache import CacheConfig, ClassifierCache


from wisent.core.optuna.classifier._optuna_config import (
    get_model_dtype,
    ClassifierOptimizationConfig,
    OptimizationResult,
)
from wisent.core.optuna.classifier._optuna_objective import OptunaObjectiveMixin
from wisent.core.optuna.classifier._optuna_pruning import OptunaPruningMixin


class OptunaClassifierOptimizer(OptunaObjectiveMixin, OptunaPruningMixin):
    """
    Optuna-based classifier optimizer with efficient caching and pre-generation.

    Key features:
    - Pre-generates activations once for all trials
    - Uses intelligent model caching to avoid retraining
    - Supports both logistic and MLP classifiers
    - Multi-objective optimization with pruning
    - Cross-validation for robust evaluation
    """

    def __init__(
        self,
        optimization_config: ClassifierOptimizationConfig,
        generation_config: GenerationConfig,
        cache_config: CacheConfig,
    ):
        self.opt_config = optimization_config
        self.gen_config = generation_config
        self.cache_config = cache_config

        self.activation_generator = ActivationGenerator(generation_config)
        self.classifier_cache = ClassifierCache(cache_config)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Statistics tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.activation_data: dict[str, ActivationData] = {}

    def optimize(
        self, model, contrastive_pairs: list, task_name: str, model_name: str, limit: int
    ) -> OptimizationResult:
        """
        Run Optuna-based classifier optimization.

        Args:
            model: Language model
            contrastive_pairs: Training contrastive pairs
            task_name: Name of the task
            model_name: Name of the model
            limit: Data limit used

        Returns:
            OptimizationResult with best configuration and classifier
        """
        self.logger.info(f"Starting Optuna classifier optimization for {task_name}")
        layer_range = self.gen_config.layer_search_range[1] - self.gen_config.layer_search_range[0] + 1
        self.logger.info(
            f"Configuration: {self.opt_config.n_trials} trials, layers {self.gen_config.layer_search_range[0]}-{self.gen_config.layer_search_range[1]} ({layer_range} layers)"
        )

        # Detect or use configured model dtype
        detected_dtype = get_model_dtype(model)
        self.model_dtype = self.opt_config.model_dtype if self.opt_config.model_dtype is not None else detected_dtype
        self.logger.info(f"Using model dtype: {self.model_dtype} (detected: {detected_dtype})")

        start_time = time.time()

        # Step 1: Pre-generate all activations
        self.logger.info("Pre-generating activations for all layers and aggregation methods...")
        self.activation_data = self.activation_generator.generate_from_contrastive_pairs(
            model=model, contrastive_pairs=contrastive_pairs, task_name=task_name, model_name=model_name, limit=limit
        )

        if not self.activation_data:
            raise NoActivationDataError()

        self.logger.info(f"Generated {len(self.activation_data)} activation datasets")

        # Step 2: Set up Optuna study
        sampler = TPESampler(seed=self.opt_config.sampler_seed)
        pruner = (
            MedianPruner(n_startup_trials=5, n_warmup_steps=self.opt_config.pruning_patience)
            if self.opt_config.enable_pruning
            else None
        )

        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        # Step 3: Run optimization
        self.logger.info("Starting Optuna trials...")

        def objective(trial):
            return self._objective_function(trial, task_name, model_name)

        study.optimize(
            objective,
            n_trials=self.opt_config.n_trials,
            timeout=self.opt_config.timeout,
            n_jobs=self.opt_config.n_jobs,
            show_progress_bar=True,
        )

        # Step 4: Get best results
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        if not completed_trials:
            self.logger.warning("No trials completed successfully - all trials were pruned or failed")
            # Show trial states for debugging
            trial_states = {}
            for trial in study.trials:
                state = trial.state.name
                trial_states[state] = trial_states.get(state, 0) + 1
            self.logger.warning(f"Trial states: {trial_states}")

            # Return a dummy result for debugging
            dummy_result = OptimizationResult(
                best_params={},
                best_value=0.0,
                best_classifier=None,
                study=study,
                trial_results=[],
                optimization_time=time.time() - start_time,
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
            )
            return dummy_result

        best_params = study.best_params
        best_value = study.best_value

        self.logger.info(f"Best trial: {best_params} -> {self.opt_config.primary_metric}={best_value:.4f}")

        # Step 5: Train final model with best parameters
        best_classifier = self._train_final_classifier(best_params, task_name, model_name)

        optimization_time = time.time() - start_time

        # Step 6: Collect trial results
        trial_results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_results.append(
                    {
                        "trial_number": trial.number,
                        "params": trial.params,
                        "value": trial.value,
                        "duration": trial.duration.total_seconds() if trial.duration else None,
                    }
                )

        result = OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            best_classifier=best_classifier,
            study=study,
            trial_results=trial_results,
            optimization_time=optimization_time,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
        )

        self.logger.info(
            f"Optimization completed in {optimization_time:.1f}s "
            f"({self.cache_hits} cache hits, {self.cache_misses} cache misses)"
        )

        return result

