"""
Weight modification optimizer using BaseOptimizer.

Optimizes weight modification parameters (strength, max_weight, min_weight, position)
to achieve target metrics like compliance rate or task accuracy.

Supports checkpointing for long-running optimizations:
- Save checkpoint after each trial (or every N trials)
- Resume from checkpoint if interrupted
- Save best model periodically
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Literal

import optuna
import torch

from wisent.core.opti.core.atoms import BaseOptimizer, Direction, HPOConfig, HPORun

__all__ = [
    "WeightsOptimizer",
    "WeightsOptimizerConfig",
]


@dataclass(slots=True)
class WeightsOptimizerConfig:
    """
    Configuration for weight modification optimization.

    attributes:
        strength_range:
            Range for strength parameter (min, max).
        max_weight_range:
            Range for max_weight parameter (min, max).
        min_weight_range:
            Range for min_weight parameter (min, max).
        position_range:
            Range for max_weight_position as ratio of layers (min, max).
        method:
            Weight modification method: "directional" or "bake".
        components:
            Number of components for weight modification.
        norm_preserve:
            Whether to preserve norms during directional projection.
        optimize_direction_index:
            Whether to optimize direction index as well.
        target_metric:
            Metric to optimize (e.g., "compliance_rate", "accuracy").
        target_value:
            Target value to achieve (for early stopping).
    """
    strength_range: tuple[float, float] = (0.5, 2.0)
    max_weight_range: tuple[float, float] = (0.1, 1.0)
    min_weight_range: tuple[float, float] = (0.0, 0.3)
    position_range: tuple[float, float] = (0.3, 0.7)
    method: Literal["directional", "bake"] = "directional"
    components: int = 1
    norm_preserve: bool = True
    optimize_direction_index: bool = False
    target_metric: str = "compliance_rate"
    target_value: float | None = None


class WeightsOptimizer(BaseOptimizer):
    """
    Optuna optimizer for weight modification parameters.

    Finds optimal parameters (strength, max_weight, min_weight, position) for
    applying steering vectors as permanent weight modifications.

    arguments:
        model:
            The model to optimize (will be modified in-place during trials).
        base_state_dict:
            Original model state dict for restoration between trials.
        steering_vectors:
            Dict mapping layer indices to steering vector tensors.
        evaluator:
            Callable that takes (model, tokenizer) and returns dict with metrics.
        tokenizer:
            Tokenizer for the model.
        config:
            WeightsOptimizerConfig with search space and method settings.
        num_layers:
            Number of layers in the model.

    example:
        >>> from wisent.opti.methods.opti_weights import WeightsOptimizer, WeightsOptimizerConfig
        >>> from wisent.opti.core.atoms import HPOConfig
        >>>
        >>> config = WeightsOptimizerConfig(
        ...     strength_range=(0.5, 2.0),
        ...     max_weight_range=(0.1, 1.0),
        ...     target_metric="compliance_rate",
        ... )
        >>>
        >>> optimizer = WeightsOptimizer(
        ...     model=model,
        ...     base_state_dict=base_state_dict,
        ...     steering_vectors=steering_vectors,
        ...     evaluator=my_evaluator,
        ...     tokenizer=tokenizer,
        ...     config=config,
        ...     num_layers=32,
        ... )
        >>>
        >>> result = optimizer.optimize(HPOConfig(n_trials=50, direction="maximize"))
        >>> print("Best params:", result.best_params)
        >>> print("Best score:", result.best_value)
    """

    name = "weights-optimizer"
    direction: Direction = "maximize"

    def __init__(
        self,
        model: Any,
        base_state_dict: dict[str, torch.Tensor],
        steering_vectors: dict[int, torch.Tensor],
        evaluator: Callable[[Any, Any], dict[str, float]],
        tokenizer: Any,
        config: WeightsOptimizerConfig,
        num_layers: int,
    ) -> None:
        self.model = model
        self.base_state_dict = base_state_dict
        self.steering_vectors = steering_vectors
        self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.config = config
        self.num_layers = num_layers

        # Set direction based on target metric
        if config.target_metric in ["refusal_rate", "kl_divergence"]:
            self.direction = "minimize"
        else:
            self.direction = "maximize"

    def _objective(self, trial: optuna.Trial) -> float:
        """
        One trial: apply weight modification with suggested params and evaluate.

        arguments:
            trial: Optuna trial object.

        returns:
            float, value of the target metric.
        """
        # Suggest parameters
        params = {
            "strength": trial.suggest_float(
                "strength",
                self.config.strength_range[0],
                self.config.strength_range[1],
            ),
            "max_weight": trial.suggest_float(
                "max_weight",
                self.config.max_weight_range[0],
                self.config.max_weight_range[1],
            ),
            "min_weight": trial.suggest_float(
                "min_weight",
                self.config.min_weight_range[0],
                self.config.min_weight_range[1],
            ),
            "max_weight_position": trial.suggest_float(
                "max_weight_position",
                self.config.position_range[0],
                self.config.position_range[1],
            ),
        }

        # Optional direction index optimization
        if self.config.optimize_direction_index:
            params["direction_index"] = trial.suggest_float(
                "direction_index",
                0.0,
                self.num_layers - 1,
            )

        # Restore base model weights
        # Use strict=False because bake_steering may add bias parameters
        # that didn't exist in the original model
        self._restore_base_weights()

        # Apply weight modification
        self._apply_weight_modification(params)

        # Evaluate
        eval_result = self.evaluator(self.model, self.tokenizer)

        # Get target metric score
        score = eval_result.get(
            self.config.target_metric,
            eval_result.get("score", 0.0),
        )

        # Check for early stopping if target is set
        if self.config.target_value is not None:
            if self.direction == "maximize" and score >= self.config.target_value:
                trial.study.stop()
            elif self.direction == "minimize" and score <= self.config.target_value:
                trial.study.stop()

        return float(score)

    def _apply_weight_modification(self, params: dict[str, float]) -> None:
        """
        Apply weight modification with given parameters.

        arguments:
            params: Dict with strength, max_weight, min_weight, max_weight_position.
        """
        from wisent.core.weight_modification import (
            project_with_kernel,
            bake_steering_with_kernel,
        )

        # Convert position ratio to layer index
        max_weight_position = params["max_weight_position"] * (self.num_layers - 1)

        # Compute min_weight_distance from position
        min_weight_distance = 0.6 * (self.num_layers - 1)

        if self.config.method == "directional":
            project_with_kernel(
                self.model,
                self.steering_vectors,
                max_weight=params["max_weight"] * params["strength"],
                max_weight_position=max_weight_position,
                min_weight=params["min_weight"],
                min_weight_distance=min_weight_distance,
                components=self.config.components,
                norm_preserve=self.config.norm_preserve,
                verbose=False,
            )
        elif self.config.method in ("additive", "titan", "prism", "pulse"):
            # Direct additive: add steering vector directly to weight matrices
            # This modifies weights directly, not biases, so it persists when saved
            # Used for additive and multi-direction methods (titan/prism/pulse)
            self._apply_direct_additive(params)
        else:
            # Default fallback - use bake_steering_with_kernel
            # Note: This adds biases which may not load correctly for some architectures
            bake_steering_with_kernel(
                self.model,
                self.steering_vectors,
                max_alpha=params["max_weight"] * params["strength"],
                max_alpha_position=max_weight_position,
                min_alpha=params["min_weight"],
                components=self.config.components,
                verbose=False,
            )

    def _apply_direct_additive(self, params: dict[str, float]) -> None:
        """
        Apply direct additive weight modification.
        
        This directly adds steering vectors to weight matrices:
        W' = W + strength * steering_vector
        
        This is the simplest approach and worked in manual humanization tests.
        """
        strength = params["strength"] * params["max_weight"]
        max_weight_position = params["max_weight_position"] * (self.num_layers - 1)
        min_weight = params["min_weight"]
        min_weight_distance = 0.6 * (self.num_layers - 1)
        
        # Get model layers
        if hasattr(self.model, "model"):
            layers = self.model.model.layers
        elif hasattr(self.model, "transformer"):
            layers = self.model.transformer.h
        else:
            layers = self.model.layers
        
        components = self.config.components or ["self_attn.o_proj", "mlp.down_proj"]
        
        for layer_idx, steering_vector in self.steering_vectors.items():
            if layer_idx >= len(layers):
                continue
            
            # Compute layer-specific strength using kernel
            distance = abs(layer_idx - max_weight_position)
            if distance > min_weight_distance:
                layer_strength = min_weight
            else:
                layer_strength = strength + (distance / min_weight_distance) * (min_weight - strength)
            
            if layer_strength <= 0:
                continue
            
            layer = layers[layer_idx]
            
            for component_name in components:
                try:
                    component = layer
                    for attr in component_name.split("."):
                        component = getattr(component, attr)
                    
                    if hasattr(component, "weight"):
                        vec = steering_vector.to(component.weight.device, dtype=component.weight.dtype)
                        with torch.no_grad():
                            # Direct addition: add steering vector to each column of weight matrix
                            # Weight shape: [out_dim, in_dim], vec shape: [out_dim]
                            # Result: each output dimension gets shifted by layer_strength * vec[i]
                            # This is equivalent to adding a bias toward the steering direction
                            component.weight.data += layer_strength * vec.unsqueeze(1)
                except AttributeError:
                    continue

    def _restore_base_weights(self) -> None:
        """
        Restore model to base weights.
        
        Uses strict=False because bake_steering may add bias parameters
        that didn't exist in the original model. Also removes any bias
        parameters that were added during weight modification.
        """
        # First, remove any bias parameters that were added
        # (bake_steering_with_kernel may have added these)
        if hasattr(self.model, "model"):
            layers = self.model.model.layers
        elif hasattr(self.model, "transformer"):
            layers = self.model.transformer.h
        else:
            layers = getattr(self.model, "layers", [])
        
        components_to_check = self.config.components or ["self_attn.o_proj", "mlp.down_proj"]
        
        for layer in layers:
            for component_name in components_to_check:
                try:
                    component = layer
                    for attr in component_name.split("."):
                        component = getattr(component, attr)
                    
                    # Check if bias was added (not in base_state_dict)
                    if hasattr(component, "bias") and component.bias is not None:
                        # Check if this bias exists in base state dict
                        bias_key = None
                        for key in self.base_state_dict.keys():
                            if component_name in key and key.endswith(".bias"):
                                bias_key = key
                                break
                        
                        if bias_key is None:
                            # Bias was added, remove it
                            component.bias = None
                except AttributeError:
                    continue
        
        # Now load state dict with strict=False
        self.model.load_state_dict(self.base_state_dict, strict=False)

    def apply_best_params(self, best_params: dict[str, float]) -> None:
        """
        Apply the best parameters found during optimization.

        Restores base weights first, then applies weight modification
        with the optimal parameters.

        arguments:
            best_params: Best parameters from optimization result.
        """
        self._restore_base_weights()
        self._apply_weight_modification(best_params)

    def optimize_with_checkpointing(
        self,
        cfg: HPOConfig,
        checkpoint_path: str | None = None,
        checkpoint_interval: int = 5,
        output_dir: str | None = None,
        tokenizer: Any = None,
        s3_bucket: str | None = None,
        s3_key_prefix: str | None = None,
    ) -> HPORun:
        """
        Run optimization with checkpointing support.

        Saves checkpoint after every checkpoint_interval trials and can resume
        from an existing checkpoint file.

        arguments:
            cfg: HPOConfig object with optimization settings.
            checkpoint_path: Path to save/load checkpoint file.
            checkpoint_interval: Save checkpoint every N trials (default: 5).
            output_dir: Directory to save best model periodically.
            tokenizer: Tokenizer to save with model.

        returns:
            HPORun object with the results of the optimization.
        """
        # Try to load existing checkpoint
        start_trial = 0
        existing_trials = []
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\n   Loading checkpoint from {checkpoint_path}...")
            checkpoint = self._load_checkpoint(checkpoint_path)
            if checkpoint:
                existing_trials = checkpoint.get("trials", [])
                start_trial = len(existing_trials)
                print(f"   Resuming from trial {start_trial}")
                print(f"   Previous best: {checkpoint.get('best_value', 'N/A')}")

        # Create sampler and pruner
        sampler = self._make_sampler(cfg)
        pruner = self._make_pruner(cfg)
        direction: Direction = getattr(self, "direction", cfg.direction)

        # Create study
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )

        # Enqueue existing trials if resuming
        for trial_data in existing_trials:
            study.enqueue_trial(trial_data["params"])

        # Calculate remaining trials
        remaining_trials = cfg.n_trials - start_trial
        if remaining_trials <= 0:
            print(f"   Optimization already complete ({start_trial} trials)")
            return HPORun(study=study, best_params=study.best_params, best_value=study.best_value)

        # Create checkpoint callback
        def checkpoint_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            trial_num = trial.number + 1  # 1-indexed for display
            
            # Save checkpoint at intervals
            if checkpoint_path and trial_num % checkpoint_interval == 0:
                self._save_checkpoint(study, checkpoint_path)
                print(f"   [Checkpoint saved at trial {trial_num}]")
                
                # Upload checkpoint to S3
                if s3_bucket and s3_key_prefix:
                    self._upload_to_s3(checkpoint_path, s3_bucket, f"{s3_key_prefix}/checkpoint.json")

            # Save best model at intervals
            if output_dir and trial_num % checkpoint_interval == 0:
                if study.best_trial is not None:
                    self._save_best_model_checkpoint(study, output_dir, tokenizer)
                    
                    # Upload best model checkpoint to S3
                    if s3_bucket and s3_key_prefix:
                        checkpoint_dir = os.path.join(output_dir, "checkpoint_best")
                        self._upload_to_s3(checkpoint_dir, s3_bucket, f"{s3_key_prefix}/checkpoint_best/")

        # Run optimization with callback
        study.optimize(
            self._objective,
            n_trials=remaining_trials,
            timeout=cfg.timeout,
            show_progress_bar=False,
            callbacks=[checkpoint_callback],
        )

        # Save final checkpoint
        if checkpoint_path:
            self._save_checkpoint(study, checkpoint_path)
            print(f"   [Final checkpoint saved]")

        return HPORun(study=study, best_params=study.best_params, best_value=study.best_value)

    def _save_checkpoint(self, study: optuna.Study, checkpoint_path: str) -> None:
        """Save optimization checkpoint to file."""
        checkpoint = {
            "trials": [
                {
                    "number": t.number,
                    "params": t.params,
                    "value": t.value,
                    "state": str(t.state),
                }
                for t in study.trials
            ],
            "best_params": study.best_params if study.best_trial else None,
            "best_value": study.best_value if study.best_trial else None,
            "n_trials": len(study.trials),
        }
        
        # Write to temp file first, then rename (atomic)
        temp_path = checkpoint_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        os.replace(temp_path, checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str) -> dict | None:
        """Load optimization checkpoint from file."""
        try:
            with open(checkpoint_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"   Warning: Could not load checkpoint: {e}")
            return None

    def _save_best_model_checkpoint(
        self,
        study: optuna.Study,
        output_dir: str,
        tokenizer: Any = None,
    ) -> None:
        """Save the current best model as a checkpoint."""
        if study.best_trial is None:
            return

        # Apply best params
        best_params = study.best_params
        self._restore_base_weights()
        self._apply_weight_modification(best_params)

        # Save model
        checkpoint_dir = os.path.join(output_dir, "checkpoint_best")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        if tokenizer:
            tokenizer.save_pretrained(checkpoint_dir)

        # Save metadata
        metadata = {
            "best_params": best_params,
            "best_value": study.best_value,
            "trial_number": study.best_trial.number,
            "total_trials": len(study.trials),
        }
        with open(os.path.join(checkpoint_dir, "checkpoint_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def _upload_to_s3(self, local_path: str, s3_bucket: str, s3_key: str) -> bool:
        """Upload a file or directory to S3."""
        import subprocess
        try:
            if os.path.isdir(local_path):
                cmd = ["aws", "s3", "sync", local_path, f"s3://{s3_bucket}/{s3_key}", "--quiet"]
            else:
                cmd = ["aws", "s3", "cp", local_path, f"s3://{s3_bucket}/{s3_key}", "--quiet"]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except Exception:
            return False
