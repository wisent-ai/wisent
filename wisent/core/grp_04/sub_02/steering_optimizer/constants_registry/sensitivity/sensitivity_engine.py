"""Sensitivity analysis engine for named constants.

Varies each constant one-at-a-time across its registry-defined range,
patches the constant module, runs steering via MethodOptimizer at a
fixed operating point, and measures score delta to identify which
constants actually matter.

The operating point (layer, strength, extraction strategy) is found
once via a full optimization pass before sensitivity measurement begins.
This avoids embedding arbitrary layer/strength choices into the
measurement apparatus.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

from wisent.core.constants import (
    DEFAULT_LIMIT, DEFAULT_SCORE, SENSITIVITY_DEFAULT_STEPS,
    SENSITIVITY_DEFAULT_CONSTANT_LIMIT,
)

logger = logging.getLogger(__name__)


@dataclass
class ConstantSensitivity:
    """Sensitivity measurement for a single constant."""

    name: str
    group: str
    method: str
    current_value: float
    tested_values: List[float]
    scores: List[float]
    sensitivity: float
    best_value: float
    best_score: float
    evaluation_time: float


@dataclass
class SensitivityResult:
    """Full sensitivity analysis results."""

    model_name: str
    task_name: str
    method_name: str
    baseline_score: float
    operating_point: Dict[str, Any]
    constants_tested: int
    results: List[ConstantSensitivity] = field(default_factory=list)
    total_time: float = DEFAULT_SCORE

    def ranked(self) -> List[ConstantSensitivity]:
        """Return constants ranked by sensitivity (highest first)."""
        return sorted(
            self.results, key=lambda r: r.sensitivity, reverse=True,
        )

    def top_n(self, n: int) -> List[ConstantSensitivity]:
        """Return the top-N most sensitive constants."""
        return self.ranked()[:n]

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "model_name": self.model_name,
            "task_name": self.task_name,
            "method_name": self.method_name,
            "baseline_score": self.baseline_score,
            "operating_point": self.operating_point,
            "constants_tested": self.constants_tested,
            "total_time": self.total_time,
            "results": [asdict(r) for r in self.results],
        }

    def save(self, path: Path) -> None:
        """Save results to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> SensitivityResult:
        """Load results from JSON."""
        with open(path) as f:
            data = json.load(f)
        results = [
            ConstantSensitivity(**r) for r in data.pop("results", [])
        ]
        return cls(**data, results=results)


class SensitivityEngine:
    """Measure constant sensitivity at a fixed operating point.

    Phase 1: Run a full optimization to find the best
    layer, strength, extraction strategy, and method params.

    Phase 2: Fix that single config. For each constant, vary it
    across its range and measure the score at that fixed point.
    One train+eval per constant per step -- no re-searching.
    """

    def __init__(
        self,
        model,
        method_name: str,
        task_name: str,
        train_pairs,
        test_pairs,
        evaluator,
        steps: int = SENSITIVITY_DEFAULT_STEPS,
        verbose: bool = True,
    ):
        self.model = model
        self.method_name = method_name.lower()
        self.task_name = task_name
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs
        self.evaluator = evaluator
        self.steps = steps
        self.verbose = verbose
        self._fixed_config = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _find_operating_point(self) -> None:
        """Run full optimization once to find the best config."""
        from wisent.core.cli.optimization.core.method_optimizer import (
            MethodOptimizer,
        )
        self._log("Phase 1: Finding optimal operating point "
                   "(full layer/strength search)...")

        optimizer = MethodOptimizer(
            model=self.model,
            method_name=self.method_name,
            verbose=self.verbose,
        )
        summary = optimizer.optimize(
            train_pairs=self.train_pairs,
            test_pairs=self.test_pairs,
            evaluator=self.evaluator,
            task_name=self.task_name,
        )

        if summary.best_result is None:
            raise RuntimeError(
                "Full optimization found no valid configs. "
                "Cannot establish operating point."
            )

        self._fixed_config = summary.best_result.config
        self._log(
            f"Operating point found: "
            f"layers={self._fixed_config.layers}, "
            f"strength={self._fixed_config.strength}, "
            f"score={summary.best_result.score:.4f}\n"
        )

    def _evaluate_at_fixed_point(
        self, overrides: Dict[str, float],
    ) -> float:
        """Evaluate steering at fixed operating point with overrides."""
        from wisent.core.cli.optimization.core.method_optimizer import (
            MethodOptimizer,
        )
        from ..registry import ConstantPatcher

        with ConstantPatcher(overrides):
            optimizer = MethodOptimizer(
                model=self.model,
                method_name=self.method_name,
                verbose=False,
            )
            summary = optimizer.optimize(
                train_pairs=self.train_pairs,
                test_pairs=self.test_pairs,
                evaluator=self.evaluator,
                task_name=self.task_name,
                configs=[self._fixed_config],
            )
            if summary.best_result is not None:
                return summary.best_result.score
            return DEFAULT_SCORE

    def run(
        self,
        group: Optional[str] = None,
        method_filter: Optional[str] = None,
        limit: int = SENSITIVITY_DEFAULT_CONSTANT_LIMIT,
    ) -> SensitivityResult:
        """Run sensitivity analysis across registered constants."""
        from ..registry import (
            get_registry, get_constants_by_group,
            get_constants_by_method,
        )

        start_time = time.time()

        self._find_operating_point()

        if group:
            constants = get_constants_by_group(group)
        elif method_filter:
            constants = get_constants_by_method(method_filter)
        else:
            constants = get_registry()

        constant_list = list(constants.values())[:limit]

        self._log(
            f"Phase 2: Sensitivity sweep — {len(constant_list)} "
            f"constants, {self.steps} steps each"
        )

        baseline_score = self._evaluate_at_fixed_point({})
        self._log(f"Baseline at operating point: {baseline_score:.4f}\n")

        results: List[ConstantSensitivity] = []

        for idx, meta in enumerate(constant_list):
            t0 = time.time()
            test_values = meta.sample_linspace(self.steps)
            scores = []

            self._log(
                f"[{idx + 1}/{len(constant_list)}] {meta.name} "
                f"({meta.group}): {len(test_values)} values "
                f"in [{meta.low}, {meta.high}]"
            )

            for val in test_values:
                score = self._evaluate_at_fixed_point({meta.name: val})
                scores.append(score)

            sensitivity = (
                max(scores) - min(scores) if scores else DEFAULT_SCORE
            )
            best_idx = scores.index(max(scores)) if scores else 0
            best_value = (
                test_values[best_idx] if test_values
                else meta.current_value
            )
            best_score = scores[best_idx] if scores else DEFAULT_SCORE
            eval_time = time.time() - t0

            result = ConstantSensitivity(
                name=meta.name,
                group=meta.group,
                method=meta.method,
                current_value=meta.current_value,
                tested_values=test_values,
                scores=scores,
                sensitivity=sensitivity,
                best_value=best_value,
                best_score=best_score,
                evaluation_time=eval_time,
            )
            results.append(result)

            delta = best_score - baseline_score
            self._log(
                f"    sensitivity={sensitivity:.4f}, "
                f"best={best_value:.4g} (delta={delta:+.4f}), "
                f"time={eval_time:.1f}s\n"
            )

        total_time = time.time() - start_time
        op_dict = self._fixed_config.to_dict()

        return SensitivityResult(
            model_name=str(
                getattr(self.model, 'model_name', 'unknown'),
            ),
            task_name=self.task_name,
            method_name=self.method_name,
            baseline_score=baseline_score,
            operating_point=op_dict,
            constants_tested=len(results),
            results=results,
            total_time=total_time,
        )
