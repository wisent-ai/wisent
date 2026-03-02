from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import json
import torch
import datetime as _dt

from wisent.core.activations.core.atoms import (
    LayerActivations,
    RawActivationMap,
)
from wisent.core.activations import ExtractionStrategy
from wisent.core.models.wisent_model import WisentModel

from wisent.core.trainers.core.atoms import (
    TrainingResult,
    BaseSteeringTrainer
)

from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.activations.activations_collector import ActivationCollector  
from wisent.core.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.contrastive_pairs.diagnostics import run_control_vector_diagnostics, run_vector_quality_diagnostics, VectorQualityConfig
from wisent.core.errors import ControlVectorDiagnosticsError, NoTrainingResultError, VectorQualityTooLowError
from wisent.core.constants import STEERABILITY_MIN_PAIRS, DISPLAY_TOP_N_TINY

__all__ = [
    "WisentSteeringTrainer",
]


logger = logging.getLogger(__name__)

@dataclass(slots=True)
class WisentSteeringTrainer(BaseSteeringTrainer):
    """
    Orchestrates activation collection + steering vector training for a given model and pair set.

    Minimal usage:
        trainer = WisentSteeringTrainer(model, pair_set, steering_method)
        result = trainer.run(layers_spec=..., method_kwargs=..., aggregation=..., ...)
        # result is a TrainingResult with steered vectors, enriched pair set, and metadata
        trainer.save_result(output_dir)  # optional save
    
    arguments:
        model: WisentModel to use for activation collection.
        pair_set: ContrastivePairSet with pairs to use for collection and training.
        steering_method: BaseSteeringMethod instance to use for training.
        store_device: Device to store collected activations on (default: "cpu" to avoid GPU OOM).
        dtype: Optional torch.dtype to cast collected activations to.
    """

    model: WisentModel
    pair_set: ContrastivePairSet
    steering_method: BaseSteeringMethod
    store_device: str | torch.device = "cpu"
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        self.collector = ActivationCollector(model=self.model, store_device=self.store_device, dtype=self.dtype)
        self._last_result: TrainingResult | None = None

    def run(
        self,
        layers_spec: Sequence[str] | str | int | Sequence[int] | None,
        method_kwargs: dict[str, Any] | None = None,
        strategy: ExtractionStrategy = ExtractionStrategy.CHAT_LAST,
        normalize_layers: bool = False,
        save_dir: str | Path | None = None,
        accept_low_quality_vector: bool = False,
        quality_config: VectorQualityConfig | None = None,
    ) -> TrainingResult:
        """
        Full pipeline:
          1) Decide which layers to use (from spec or all layers if None).
          2) Collect activations for each pair at these layers.
          3) Train steering vectors using the selected method.
          4) Return a TrainingResult with vectors, enriched pair set, and metadata.
          5) Optionally save artifacts to disk.

        arguments:
            layers_spec:
                - list like ["10","20","30"] or [10, 20, 30]
                - range string "10-30" / "10..30"
                - single int "12"
                - None → use all available layers on the model
            method_kwargs:
                Dict of hyperparameters for the method (e.g., {"normalize": True, "scale": 1.0}).
            strategy:
                ExtractionStrategy to use during collection.
            normalize_layers:
                If True, L2-normalize activations layer-wise during collection.
            save_dir:
                If provided, artifacts are written there. Directory is created if missing.

        returns:
            TrainingResult
        """
        method_kwargs = method_kwargs or {}

        # 1) Resolve layer names
        layers = self._resolve_layers(layers_spec)

        # 2) Collect activations for each pair
        for i, pair in enumerate(self.pair_set.pairs):
            updated = self.collector.collect(
                pair,
                strategy=strategy,
                layers=layers,
                normalize=normalize_layers,
            )
            self.pair_set.pairs[i] = updated  

        # 3) Train using selected method
        raw_vectors: RawActivationMap = self.steering_method.train(self.pair_set, **(method_kwargs or {}))

        steered = LayerActivations(raw_vectors)

        control_vector_report = run_control_vector_diagnostics(steered)
        for issue in control_vector_report.issues:
            log_method = logger.error if issue.severity == "critical" else logger.warning
            log_method(
                "[control_vector diagnostics] %s (details=%s)",
                issue.message,
                issue.details,
            )

        control_vector_summary = control_vector_report.summary.get("control_vectors", {})
        control_vector_issues = [
            {
                "metric": issue.metric,
                "severity": issue.severity,
                "message": issue.message,
                "details": issue.details,
            }
            for issue in control_vector_report.issues
        ]

        if control_vector_report.has_critical_issues:
            raise ControlVectorDiagnosticsError()

        # 3b) Run vector quality diagnostics if we have enough pairs
        quality_report = None
        quality_diagnostics_report = None
        if len(self.pair_set.pairs) >= STEERABILITY_MIN_PAIRS:
            try:
                # Extract activations for quality analysis (use first layer with data)
                positive_activations = []
                negative_activations = []
                pair_prompts = []
                
                for pair in self.pair_set.pairs:
                    if pair.positive_response and pair.positive_response.layers_activations:
                        pos_acts = pair.positive_response.layers_activations.to_dict()
                        if pos_acts:
                            first_layer = next(iter(pos_acts.keys()))
                            pos_tensor = pos_acts[first_layer]
                            if pos_tensor is not None:
                                positive_activations.append(pos_tensor)
                    
                    if pair.negative_response and pair.negative_response.layers_activations:
                        neg_acts = pair.negative_response.layers_activations.to_dict()
                        if neg_acts:
                            first_layer = next(iter(neg_acts.keys()))
                            neg_tensor = neg_acts[first_layer]
                            if neg_tensor is not None:
                                negative_activations.append(neg_tensor)
                    
                    pair_prompts.append(pair.prompt if hasattr(pair, 'prompt') else "")
                
                if len(positive_activations) >= STEERABILITY_MIN_PAIRS and len(negative_activations) >= STEERABILITY_MIN_PAIRS:
                    pos_stacked = torch.stack(positive_activations)
                    neg_stacked = torch.stack(negative_activations)
                    
                    quality_report, quality_diagnostics_report = run_vector_quality_diagnostics(
                        pos_stacked, neg_stacked, pair_prompts, quality_config
                    )
                    
                    # Log quality issues
                    for issue in quality_diagnostics_report.issues:
                        log_method = logger.error if issue.severity == "critical" else logger.warning
                        log_method(
                            "[vector_quality diagnostics] %s (details=%s)",
                            issue.message,
                            issue.details,
                        )
                    
                    # Raise error if quality is poor and not accepted
                    if quality_diagnostics_report.has_critical_issues and not accept_low_quality_vector:
                        critical_issues = [i for i in quality_diagnostics_report.issues if i.severity == "critical"]
                        reason = "; ".join(i.message for i in critical_issues[:DISPLAY_TOP_N_TINY])
                        raise VectorQualityTooLowError(
                            quality=quality_report.overall_quality,
                            reason=reason,
                            details={
                                "convergence": quality_report.convergence_score,
                                "cv_score": quality_report.cv_score_mean,
                                "snr": quality_report.snr,
                                "recommendations": quality_report.recommendations,
                            }
                        )
            except ImportError:
                # sklearn not available, skip quality diagnostics
                logger.warning("sklearn not available, skipping vector quality diagnostics")

        # 4) Metadata
        now = _dt.datetime.now().astimezone()
        metadata: dict[str, Any] = {
            "timestamp": now.isoformat(),
            "model_name": getattr(self.model, "model_name", getattr(self.model, "name", None)),
            "layers_used": layers or "all",
            "method": self.steering_method.name,
            "method_kwargs": method_kwargs,
            "extraction_strategy": strategy.value,
            "normalize_layers": bool(normalize_layers),
            "num_pairs": len(self.pair_set.pairs),
            "hidden_size": getattr(self.model, "hidden_size", None),
            "control_vector_diagnostics": control_vector_summary,
        }

        if control_vector_issues:
            metadata["control_vector_issues"] = control_vector_issues

        # Add quality diagnostics to metadata
        if quality_report is not None:
            metadata["vector_quality"] = {
                "overall_quality": quality_report.overall_quality,
                "convergence_score": quality_report.convergence_score,
                "cv_score_mean": quality_report.cv_score_mean,
                "snr": quality_report.snr,
                "pca_pc1_variance": quality_report.pca_pc1_variance,
                "silhouette_score": quality_report.silhouette_score,
                "held_out_transfer": quality_report.held_out_transfer,
                "cv_classification_accuracy": quality_report.cv_classification_accuracy,
                "cohens_d": quality_report.cohens_d,
                "alignment_mean": quality_report.alignment_mean,
                "alignment_std": quality_report.alignment_std,
                "num_outlier_pairs": len(quality_report.outlier_pairs),
                "recommendations": quality_report.recommendations,
            }
            if quality_diagnostics_report and quality_diagnostics_report.issues:
                metadata["vector_quality_issues"] = [
                    {
                        "metric": issue.metric,
                        "severity": issue.severity,
                        "message": issue.message,
                    }
                    for issue in quality_diagnostics_report.issues
                ]

        result = TrainingResult(steered_vectors=steered, pair_set_with_activations=self.pair_set, metadata=metadata)
        self._last_result = result

        # 5) Optional save
        if save_dir is not None:
            self.save_result(save_dir, result)

        return result

    def save_result(self, output_dir, result=None):
        """Persist vectors, metadata, and the pair set to disk."""
        from wisent.core.trainers._steering_trainer_helpers import save_result_impl
        return save_result_impl(result, self._last_result, output_dir)

    def _resolve_layers(self, spec):
        """Convert a user-facing spec into canonical layer names."""
        from wisent.core.trainers._steering_trainer_helpers import resolve_layers
        return resolve_layers(self.model, spec)

    @staticmethod
    def _parse_layer_token(token):
        """Parse a token like "5", "10-20", "10..20" into a list."""
        from wisent.core.trainers._steering_trainer_helpers import parse_layer_token
        return parse_layer_token(token)
