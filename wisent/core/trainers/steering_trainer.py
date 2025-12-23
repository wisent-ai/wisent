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
from wisent.core.activations.extraction_strategy import ExtractionStrategy
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
        if len(self.pair_set.pairs) >= 5:
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
                
                if len(positive_activations) >= 5 and len(negative_activations) >= 5:
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
                        reason = "; ".join(i.message for i in critical_issues[:3])
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

    def save_result(self, output_dir: str | Path, result: TrainingResult | None = None) -> Path:
        """
        Persist vectors, metadata, and the pair set (with activations) to disk.

        Files written:
            - metadata.json                (JSON)
            - steering_vectors.pt          (torch.save of dict[layer]->tensor on CPU)
            - pairs_with_activations.pt    (torch.save of the full ContrastivePairSet object)
            - steering_vectors_summary.json (shapes/dtypes only, human-readable)

        returns:
            Path to the created directory.
        """
        result = result or self._last_result
        if result is None:
            raise NoTrainingResultError()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Vectors
        raw_map: RawActivationMap = result.steered_vectors.to_dict()  # still tensors
        cpu_map = {k: (v.detach().to("cpu") if isinstance(v, torch.Tensor) else v) for k, v in raw_map.items()}
        torch.save(cpu_map, out / "steering_vectors.pt")

        # Summary (json-serializable)
        vec_summary = {
            k: None if v is None else {
                "shape": tuple(v.shape),
                "dtype": str(v.dtype),
            }
            for k, v in cpu_map.items()
        }
        (out / "steering_vectors_summary.json").write_text(json.dumps(vec_summary, indent=2))

        # Metadata
        (out / "metadata.json").write_text(json.dumps(result.metadata, indent=2))

        # Full pair set with activations (Python pickle via torch.save)
        torch.save(result.pair_set_with_activations, out / "pairs_with_activations.pt")

        return out

    def _resolve_layers(self, spec: Sequence[str] | str | int | Sequence[int] | None) -> list[str] | None:
        """
        Convert a user-facing spec into canonical layer names ("1","2",...).
        If None, return None (meaning: use all layers in the collector/model).

        arguments:
            spec: See 'layers_spec' argument in run().
        
        returns:
            Sorted list of layer names as strings, or None.

        examples:
            None -> None
            "10-12" -> ["10","11","12"]
            [5,10,15] -> ["5","10","15"]
            "3,7,10..12" -> ["3","7","10","11","12"]
            8 -> ["8"]
        """
        if spec is None:
            return None

        if isinstance(spec, (list, tuple)):
            names: list[str] = []
            for item in spec:
                if isinstance(item, int):
                    names.append(str(item))
                else:
                    names.extend(self._parse_layer_token(item))
            return sorted(set(names), key=lambda s: (len(s), s))

        if isinstance(spec, int):
            return [str(spec)]

        names: list[str] = []
        for token in str(spec).replace(" ", "").split(","):
            names.extend(self._parse_layer_token(token))
        return sorted(set(names), key=lambda s: (len(s), s))

    @staticmethod
    def _parse_layer_token(token: str) -> list[str]:
        """
        Parse a token like "5", "10-20", "10..20" into a list of names.
        """
        if not token:
            return []
        if "-" in token or ".." in token:
            a, b = token.replace("..", "-").split("-")
            a_i, b_i = int(a), int(b)
            lo, hi = (a_i, b_i) if a_i <= b_i else (b_i, a_i)
            return [str(i) for i in range(lo, hi + 1)]
        else:
            return [str(int(token))]