"""
Optuna-based optimization for abliteration parameters.

Finds optimal abliteration parameters (max_weight, strength, num_pairs, etc.)
to maximize task performance while preserving model capabilities.

This fills the gap identified in the codebase:
- Wisent has multi-objective optimization for steering (CAA)
- Wisent has NO optimization for abliteration parameters
- This module provides that missing optimization

Results are persisted to ~/.wisent/configs/{model_name}.json via the unified
WisentConfigManager so they can be automatically loaded on subsequent runs.

Usage:
    optimizer = AbliterationOptimizer(
        model_name="meta-llama/Llama-3.2-1B",
        task="hellaswag",
        evaluate_fn=lambda model_path: accuracy,
    )
    result = optimizer.optimize(n_trials=50)
    print(f"Best parameters: max_weight={result.best_max_weight}")
"""

from __future__ import annotations

import optuna
from optuna.samplers import TPESampler
from optuna.study import StudyDirection
from typing import TYPE_CHECKING, Callable, Optional
from dataclasses import dataclass
import subprocess
import json
import os

from wisent.core.config_manager import (
    get_weight_modification_cache,
    store_weight_modification,
    get_cached_weight_modification,
    WeightModificationResult,
)

if TYPE_CHECKING:
    from optuna import Trial

__all__ = [
    "AbliterationOptimizer",
    "AbliterationParameters",
    "OptimizationResult",
]


@dataclass
class AbliterationParameters:
    """
    Parameters for abliteration weight modification.

    Attributes:
        max_weight: Peak abliteration strength at center layer
        min_weight: Minimum abliteration strength at edge layers
        max_weight_position: Layer index for peak strength (as ratio of total layers)
        min_weight_distance: Distance over which strength decays (as ratio of total layers)
        strength: Global multiplier for all layer weights
        num_pairs: Number of contrastive pairs for training
    """
    max_weight: float
    min_weight: float
    max_weight_position: float  # 0.0 to 1.0
    min_weight_distance: float  # 0.0 to 1.0
    strength: float
    num_pairs: int

    def __repr__(self) -> str:
        return (
            f"AbliterationParameters("
            f"max_weight={self.max_weight:.2f}, "
            f"min_weight={self.min_weight:.2f}, "
            f"strength={self.strength:.2f}, "
            f"num_pairs={self.num_pairs})"
        )


@dataclass
class OptimizationResult:
    """
    Result from abliteration parameter optimization.

    Attributes:
        study: Optuna study object
        best_trial: Best trial found
        best_params: Best AbliterationParameters
        best_score: Best evaluation score achieved
    """
    study: optuna.Study
    best_trial: optuna.trial.FrozenTrial
    best_params: AbliterationParameters
    best_score: float


class AbliterationOptimizer:
    """
    Optuna-based optimizer for abliteration parameters.

    Finds optimal parameters to maximize task performance via abliteration.

    Usage:
        optimizer = AbliterationOptimizer(
            model_name="meta-llama/Llama-3.2-1B",
            task="hellaswag",
            trait_label="correctness",
            base_output_dir="./data/modified_models",
            evaluate_fn=evaluate_model,
        )
        result = optimizer.optimize(n_trials=50)
    """

    def __init__(
        self,
        model_name: str,
        task: str,
        trait_label: str,
        base_output_dir: str,
        evaluate_fn: Callable[[str], float],
        num_layers: int = 16,  # Llama-3.2-1B has 16 layers
        components: list[str] | None = None,
        direction: str = "maximize",
    ):
        """
        Initialize abliteration optimizer.

        Args:
            model_name: HuggingFace model name
            task: Task name for pair generation (e.g., "hellaswag")
            trait_label: Trait label for pairs (e.g., "correctness")
            base_output_dir: Base directory for saving modified models
            evaluate_fn: Function that takes model path and returns score
            num_layers: Number of transformer layers in model
            components: Components to abliterate (default: ["self_attn.o_proj", "mlp.down_proj"])
            direction: "maximize" or "minimize" the evaluation score
        """
        self.model_name = model_name
        self.task = task
        self.trait_label = trait_label
        self.base_output_dir = base_output_dir
        self.evaluate_fn = evaluate_fn
        self.num_layers = num_layers
        self.components = components or ["self_attn.o_proj", "mlp.down_proj"]
        self.direction = direction

        # Trial counter for unique output directories
        self.trial_counter = 0

    def suggest_parameters(self, trial: Trial) -> AbliterationParameters:
        """
        Suggest abliteration parameters for this trial.

        Args:
            trial: Optuna trial

        Returns:
            AbliterationParameters for this trial
        """
        # Max weight: peak strength (most important parameter)
        max_weight = trial.suggest_float("max_weight", 0.5, 5.0, log=False)

        # Min weight: minimum strength at edges
        min_weight = trial.suggest_float("min_weight", 0.0, max_weight * 0.8)

        # Max weight position: where to apply peak strength (as ratio)
        max_weight_position = trial.suggest_float("max_weight_position", 0.3, 0.7)

        # Min weight distance: how wide the kernel is (as ratio)
        min_weight_distance = trial.suggest_float("min_weight_distance", 0.3, 0.8)

        # Global strength multiplier
        strength = trial.suggest_float("strength", 0.5, 2.0)

        # Number of contrastive pairs
        num_pairs = trial.suggest_int("num_pairs", 50, 500, step=50)

        return AbliterationParameters(
            max_weight=max_weight,
            min_weight=min_weight,
            max_weight_position=max_weight_position,
            min_weight_distance=min_weight_distance,
            strength=strength,
            num_pairs=num_pairs,
        )

    def apply_abliteration(self, params: AbliterationParameters, output_dir: str) -> bool:
        """
        Apply abliteration with given parameters.

        Args:
            params: AbliterationParameters to use
            output_dir: Directory to save modified model

        Returns:
            True if successful, False otherwise
        """
        # Convert ratio positions to actual layer indices
        max_weight_position_idx = params.max_weight_position * (self.num_layers - 1)
        min_weight_distance_layers = params.min_weight_distance * (self.num_layers - 1)

        # Build command
        cmd = [
            "python", "-m", "wisent.core.main", "modify-weights",
            "--task", self.task,
            "--trait-label", self.trait_label,
            "--output-dir", output_dir,
            "--model", self.model_name,
            "--num-pairs", str(params.num_pairs),
            "--method", "abliteration",
            "--strength", str(params.strength),
            "--components", *self.components,
            "--use-kernel",
            "--max-weight", str(params.max_weight),
            "--max-weight-position", str(max_weight_position_idx),
            "--min-weight", str(params.min_weight),
            "--min-weight-distance", str(min_weight_distance_layers),
            "--normalize-vectors",
        ]

        try:
            # Run modification
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode != 0:
                print(f"Error in abliteration: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            print(f"Abliteration timed out for params: {params}")
            return False
        except Exception as e:
            print(f"Exception during abliteration: {e}")
            return False

    def objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial

        Returns:
            Evaluation score (higher is better if direction="maximize")
        """
        # Suggest parameters
        params = self.suggest_parameters(trial)

        # Create unique output directory for this trial
        output_dir = os.path.join(
            self.base_output_dir,
            f"{self.task}_optuna_trial_{trial.number}"
        )

        # Apply abliteration
        success = self.apply_abliteration(params, output_dir)

        if not success:
            # Return worst possible score
            return -1.0 if self.direction == "maximize" else 1e9

        # Evaluate modified model
        try:
            score = self.evaluate_fn(output_dir)
        except Exception as e:
            print(f"Evaluation failed for trial {trial.number}: {e}")
            score = -1.0 if self.direction == "maximize" else 1e9

        # Log intermediate result
        print(f"\nTrial {trial.number}: {params}")
        print(f"Score: {score:.4f}")

        return score

    def get_cached_result(self) -> Optional[WeightModificationResult]:
        """
        Check if there's a cached optimization result for this model/task/trait.

        Returns:
            WeightModificationResult if found, None otherwise
        """
        return get_cached_weight_modification(
            model=self.model_name,
            task=self.task,
            trait_label=self.trait_label,
            method="abliteration"
        )

    def optimize(
        self,
        n_trials: int = 50,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        show_progress: bool = True,
        use_cached: bool = False,
        save_to_cache: bool = True,
        set_as_default: bool = False,
    ) -> OptimizationResult:
        """
        Run optimization to find best abliteration parameters.

        Args:
            n_trials: Total number of trials
            n_startup_trials: Number of random startup trials
            n_ei_candidates: Number of Expected Improvement candidates
            show_progress: Whether to show progress bar
            use_cached: If True, return cached result if available
            save_to_cache: If True, save result to cache after optimization
            set_as_default: If True, set result as the default for this model/task

        Returns:
            OptimizationResult with best parameters
        """
        # Check for cached result if requested
        if use_cached:
            cached = self.get_cached_result()
            if cached:
                print(f"\n📦 Found cached optimization result for {self.task}/{self.trait_label}")
                print(f"   Score: {cached.score:.4f}")
                print(f"   max_weight: {cached.max_weight:.3f}")
                print(f"   strength: {cached.strength:.3f}")
                print(f"   num_pairs: {cached.num_pairs}")

                # Convert cached result to OptimizationResult format
                best_params = AbliterationParameters(
                    max_weight=cached.max_weight,
                    min_weight=cached.min_weight,
                    max_weight_position=cached.max_weight_position,
                    min_weight_distance=cached.min_weight_distance,
                    strength=cached.strength,
                    num_pairs=cached.num_pairs,
                )

                # Create a dummy study/trial for compatibility
                return OptimizationResult(
                    study=None,  # No study for cached results
                    best_trial=None,  # No trial for cached results
                    best_params=best_params,
                    best_score=cached.score,
                )

        # Create study with TPE sampler
        study = optuna.create_study(
            sampler=TPESampler(
                n_startup_trials=n_startup_trials,
                n_ei_candidates=n_ei_candidates,
                multivariate=True,
            ),
            direction="maximize" if self.direction == "maximize" else "minimize",
        )

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=show_progress,
        )

        # Get best trial
        best_trial = study.best_trial

        # Reconstruct best parameters
        best_params = AbliterationParameters(
            max_weight=best_trial.params["max_weight"],
            min_weight=best_trial.params["min_weight"],
            max_weight_position=best_trial.params["max_weight_position"],
            min_weight_distance=best_trial.params["min_weight_distance"],
            strength=best_trial.params["strength"],
            num_pairs=best_trial.params["num_pairs"],
        )

        result = OptimizationResult(
            study=study,
            best_trial=best_trial,
            best_params=best_params,
            best_score=best_trial.value,
        )

        # Save to cache if requested
        if save_to_cache:
            print(f"\n💾 Saving optimization result to cache...")
            cache_key = store_weight_modification(
                model=self.model_name,
                task=self.task,
                trait_label=self.trait_label,
                method="abliteration",
                max_weight=best_params.max_weight,
                min_weight=best_params.min_weight,
                max_weight_position=best_params.max_weight_position,
                min_weight_distance=best_params.min_weight_distance,
                strength=best_params.strength,
                num_pairs=best_params.num_pairs,
                components=self.components,
                normalize_vectors=True,
                norm_preserve=True,
                use_biprojection=True,
                use_kernel=True,
                score=result.best_score,
                metric="accuracy",
                output_dir=os.path.join(self.base_output_dir, f"{self.task}_best"),
                set_as_default=set_as_default,
                metadata={
                    "n_trials": n_trials,
                    "direction": self.direction,
                    "num_layers": self.num_layers,
                }
            )
            print(f"   ✓ Saved to cache: {cache_key}")
            if set_as_default:
                print(f"   ✓ Set as default for {self.model_name}/{self.task}/{self.trait_label}")

        return result

    def print_results(self, result: OptimizationResult) -> None:
        """
        Print optimization results.

        Args:
            result: OptimizationResult from optimize()
        """
        print("\n" + "=" * 80)
        print("ABLITERATION PARAMETER OPTIMIZATION RESULTS")
        print("=" * 80)
        print(f"\nBest Trial: #{result.best_trial.number}")
        print(f"Best Score: {result.best_score:.4f}")
        print(f"\nBest Parameters:")
        print(f"  max_weight: {result.best_params.max_weight:.3f}")
        print(f"  min_weight: {result.best_params.min_weight:.3f}")
        print(f"  max_weight_position: {result.best_params.max_weight_position:.3f} (ratio)")
        print(f"  min_weight_distance: {result.best_params.min_weight_distance:.3f} (ratio)")
        print(f"  strength: {result.best_params.strength:.3f}")
        print(f"  num_pairs: {result.best_params.num_pairs}")

        # Show top 5 trials
        print(f"\nTop 5 Trials:")
        print(f"{'Trial':>6} | {'Score':>8} | {'max_weight':>11} | {'strength':>9} | {'num_pairs':>10}")
        print("-" * 80)

        trials = sorted(result.study.trials, key=lambda t: t.value, reverse=(self.direction == "maximize"))
        for trial in trials[:5]:
            print(
                f"{trial.number:>6} | "
                f"{trial.value:>8.4f} | "
                f"{trial.params['max_weight']:>11.3f} | "
                f"{trial.params['strength']:>9.3f} | "
                f"{trial.params['num_pairs']:>10}"
            )

        print("=" * 80 + "\n")
