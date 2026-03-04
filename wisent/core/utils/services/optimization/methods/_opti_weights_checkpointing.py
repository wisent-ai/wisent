"""Checkpointing support for WeightsOptimizer."""

from __future__ import annotations

import json
import os
from typing import Any

import optuna
import torch

from wisent.core.utils.services.optimization.core.atoms import Direction, HPOConfig, HPORun
from wisent.core.utils.config_tools.constants import JSON_INDENT

class WeightsCheckpointingMixin:
    """Mixin providing checkpointing support for WeightsOptimizer."""

    def _apply_direct_additive(self, params: dict[str, float]) -> None:
        """
        Apply direct additive weight modification.

        This directly adds steering vectors to weight matrices:
        W' = W + strength * steering_vector
        """
        strength = params["strength"] * params["max_weight"]
        max_weight_position = params["max_weight_position"] * (self.num_layers - 1)
        min_weight = params["min_weight"]
        min_weight_distance = self.config.weight_min_distance_fraction * (self.num_layers - 1)

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
                            component.weight.data += layer_strength * vec.unsqueeze(1)
                except AttributeError:
                    continue

    def _restore_base_weights(self) -> None:
        """
        Restore model to base weights.

        Uses strict=False because bake_steering may add bias parameters
        that didn't exist in the original model.
        """
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

                    if hasattr(component, "bias") and component.bias is not None:
                        bias_key = None
                        for key in self.base_state_dict.keys():
                            if component_name in key and key.endswith(".bias"):
                                bias_key = key
                                break

                        if bias_key is None:
                            component.bias = None
                except AttributeError:
                    continue

        self.model.load_state_dict(self.base_state_dict, strict=False)

    def optimize_with_checkpointing(
        self,
        cfg: HPOConfig,
        checkpoint_path: str | None = None,
        checkpoint_interval: int = None,
        output_dir: str | None = None,
        tokenizer: Any = None,
        gcs_bucket: str | None = None,
        gcs_key_prefix: str | None = None,
    ) -> HPORun:
        """
        Run optimization with checkpointing support.

        Saves checkpoint after every checkpoint_interval trials and can resume
        from an existing checkpoint file.

        arguments:
            cfg: HPOConfig object with optimization settings.
            checkpoint_path: Path to save/load checkpoint file.
            checkpoint_interval: Save checkpoint every N trials.
            output_dir: Directory to save best model periodically.
            tokenizer: Tokenizer to save with model.

        returns:
            HPORun object with the results of the optimization.
        """
        if checkpoint_interval is None:
            raise ValueError("checkpoint_interval is required")
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

        sampler = self._make_sampler(cfg)
        pruner = self._make_pruner(cfg)
        direction: Direction = getattr(self, "direction", cfg.direction)

        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )

        for trial_data in existing_trials:
            study.enqueue_trial(trial_data["params"])

        remaining_trials = cfg.n_trials - start_trial
        if remaining_trials <= 0:
            print(f"   Optimization already complete ({start_trial} trials)")
            return HPORun(study=study, best_params=study.best_params, best_value=study.best_value)

        def checkpoint_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            trial_num = trial.number + 1

            if checkpoint_path and trial_num % checkpoint_interval == 0:
                self._save_checkpoint(study, checkpoint_path)
                print(f"   [Checkpoint saved at trial {trial_num}]")

                if gcs_bucket and gcs_key_prefix:
                    self._upload_to_gcs(checkpoint_path, gcs_bucket, f"{gcs_key_prefix}/checkpoint.json")

            if output_dir and trial_num % checkpoint_interval == 0:
                if study.best_trial is not None:
                    self._save_best_model_checkpoint(study, output_dir, tokenizer)

                    if gcs_bucket and gcs_key_prefix:
                        checkpoint_dir = os.path.join(output_dir, "checkpoint_best")
                        self._upload_to_gcs(checkpoint_dir, gcs_bucket, f"{gcs_key_prefix}/checkpoint_best/")

        study.optimize(
            self._objective,
            n_trials=remaining_trials,
            show_progress_bar=False,
            callbacks=[checkpoint_callback],
        )

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

        temp_path = checkpoint_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(checkpoint, f, indent=JSON_INDENT)
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

        best_params = study.best_params
        self._restore_base_weights()
        self._apply_weight_modification(best_params)

        checkpoint_dir = os.path.join(output_dir, "checkpoint_best")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        if tokenizer:
            tokenizer.save_pretrained(checkpoint_dir)

        metadata = {
            "best_params": best_params,
            "best_value": study.best_value,
            "trial_number": study.best_trial.number,
            "total_trials": len(study.trials),
        }
        with open(os.path.join(checkpoint_dir, "checkpoint_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=JSON_INDENT)

    def _upload_to_gcs(self, local_path: str, gcs_bucket: str, gcs_key: str) -> bool:
        """Upload a file or directory to GCS."""
        import subprocess
        try:
            if os.path.isdir(local_path):
                cmd = ["gcloud", "storage", "rsync", local_path, f"gs://{gcs_bucket}/{gcs_key}"]
            else:
                cmd = ["gcloud", "storage", "cp", local_path, f"gs://{gcs_bucket}/{gcs_key}"]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except Exception:
            return False
