"""Threshold calibration from empirical metric distributions.

Instead of searching over thresholds with Optuna, collects the empirical
distribution of the underlying metric from real data, then sets thresholds
at configurable quantiles.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from wisent.core.constants import DEFAULT_LIMIT, DEFAULT_SCORE, CALIBRATION_MIN_OBSERVATIONS, JSON_INDENT

logger = logging.getLogger(__name__)


@dataclass
class ThresholdCalibrationEntry:
    """Calibration result for a single threshold constant."""

    constant_name: str
    quantile: float
    calibrated_value: float
    original_value: float
    n_observations: int
    distribution_stats: Dict[str, float]


@dataclass
class CalibrationResult:
    """Full calibration results for all calibrated thresholds."""

    model_name: str
    task_name: str
    entries: List[ThresholdCalibrationEntry] = field(default_factory=list)
    total_time: float = DEFAULT_SCORE

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "model_name": self.model_name,
            "task_name": self.task_name,
            "total_time": self.total_time,
            "entries": [asdict(e) for e in self.entries],
        }

    def save(self, path: Path) -> None:
        """Save to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=JSON_INDENT)

    @classmethod
    def load(cls, path: Path) -> CalibrationResult:
        """Load from JSON."""
        with open(path) as f:
            data = json.load(f)
        entries = [ThresholdCalibrationEntry(**e) for e in data.pop("entries", [])]
        return cls(**data, entries=entries)

    def as_overrides(self) -> Dict[str, float]:
        """Convert to a dict of constant name -> calibrated value."""
        return {e.constant_name: e.calibrated_value for e in self.entries}


class ThresholdCalibrator:
    """Calibrate Group E thresholds from empirical metric distributions."""

    def __init__(
        self,
        model: Any,
        task_name: str,
        pairs: Any,
        verbose: bool = True,
    ):
        self.model = model
        self.task_name = task_name
        self.pairs = pairs
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def calibrate(
        self,
        specs: Optional[List] = None,
        quantile_overrides: Optional[Dict[str, float]] = None,
    ) -> CalibrationResult:
        """Run calibration for all mapped thresholds."""
        from .threshold_metric_map import THRESHOLD_METRIC_MAP, get_collector
        from ..registry import get_registry, get_constant_value

        start_time = time.time()
        registry = get_registry()

        if specs is None:
            specs = THRESHOLD_METRIC_MAP

        quantile_overrides = quantile_overrides or {}
        entries: List[ThresholdCalibrationEntry] = []

        self._log(f"Threshold calibration: {len(specs)} thresholds")
        self._log(f"Task: {self.task_name}, Pairs: {len(self.pairs)}\n")

        for spec in specs:
            if spec.constant_name not in registry:
                logger.warning("Threshold %s not in registry", spec.constant_name)
                continue

            collector = get_collector(spec.collector_name)
            if collector is None:
                logger.warning(
                    "No collector %s for %s", spec.collector_name, spec.constant_name,
                )
                continue

            self._log(f"  {spec.constant_name}: collecting {spec.collector_name}...")

            observations = collector(self.model, self.pairs, self.task_name)

            if len(observations) < CALIBRATION_MIN_OBSERVATIONS:
                self._log(f"    Insufficient data ({len(observations)} obs), skipping")
                continue

            arr = np.array(observations)
            quantile = quantile_overrides.get(
                spec.constant_name, spec.default_quantile,
            )
            calibrated = float(np.quantile(arr, quantile))
            original = get_constant_value(spec.constant_name)

            stats = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
                "p10": float(np.quantile(arr, 0.1)),
                "p90": float(np.quantile(arr, 0.9)),
            }

            entry = ThresholdCalibrationEntry(
                constant_name=spec.constant_name,
                quantile=quantile,
                calibrated_value=calibrated,
                original_value=original,
                n_observations=len(observations),
                distribution_stats=stats,
            )
            entries.append(entry)

            delta = calibrated - original
            self._log(
                f"    q={quantile:.2f}: {calibrated:.4f} "
                f"(was {original:.4f}, delta={delta:+.4f}, "
                f"n={len(observations)})"
            )

        total_time = time.time() - start_time
        model_name = str(getattr(self.model, 'model_name', 'unknown'))

        result = CalibrationResult(
            model_name=model_name,
            task_name=self.task_name,
            entries=entries,
            total_time=total_time,
        )

        self._log(f"\nCalibration complete: {len(entries)} thresholds in "
                   f"{total_time:.1f}s")
        return result

    def calibrate_to_profile(
        self,
        specs: Optional[List] = None,
        quantile_overrides: Optional[Dict[str, float]] = None,
    ):
        """Run calibration and return a ConstantProfile."""
        from ..profiles import ConstantProfile

        result = self.calibrate(specs, quantile_overrides)

        return ConstantProfile(
            model_name=result.model_name,
            task_name=self.task_name,
            constants=result.as_overrides(),
            source="calibration",
            metrics={"calibrated_thresholds": float(len(result.entries))},
        )
