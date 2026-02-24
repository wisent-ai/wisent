"""Optuna-based optimizer for recommendation parameters.

CPU-only: each trial runs compute_configurable_recommendation on all
ground-truth records (microseconds per trial).
"""
from __future__ import annotations
from dataclasses import fields
from typing import Literal

import optuna

from wisent.core.opti.core.atoms import BaseOptimizer, HPOConfig, HPORun
from .config import RecommendationConfig, Thresholds, ScoreWeights
from .configurable import compute_configurable_recommendation
from .collector import GroundTruthDataset
from wisent.core.constants import RECOMMEND_TOP_K, RECOMMEND_N_TRIALS, DEFAULT_RANDOM_SEED

ObjectiveType = Literal["top1", "topk", "regret"]

_THRESHOLD_BOUNDS = {
    "linear_probe_high": (0.5, 0.99),
    "linear_probe_low": (0.3, 0.85),
    "nonlinearity_gap_significant": (0.01, 0.2),
    "icd_high": (0.3, 0.95),
    "icd_low": (0.05, 0.6),
    "stability_high": (0.5, 0.99),
    "stability_low": (0.2, 0.8),
    "multi_concept_silhouette": (0.05, 0.6),
    "alignment_high": (0.05, 0.6),
    "alignment_low": (0.01, 0.3),
    "variance_concentrated": (0.05, 0.5),
    "effective_dims_low": (5.0, 50.0),
    "multi_dir_gain_high": (0.01, 0.2),
    "coherence_high": (0.5, 0.99),
    "coherence_low": (0.2, 0.8),
    "icd_top5_threshold": (0.1, 0.8),
}


def _weight_bounds(name: str) -> tuple[float, float]:
    if name in ("not_viable_scale", "confidence_base"):
        return (0.1, 1.0)
    return (-5.0, 5.0)


class RecommendationOptimizer(BaseOptimizer):
    """Optimizes recommendation config to maximize agreement
    with empirical best-method data."""

    name = "recommendation-optimizer"
    direction = "maximize"

    def __init__(
        self, dataset: GroundTruthDataset,
        objective_type: ObjectiveType = "top1", top_k: int = RECOMMEND_TOP_K,
    ):
        self.dataset = dataset
        self.objective_type = objective_type
        self.top_k = top_k

    def _objective(self, trial: optuna.Trial) -> float:
        t_kw = {}
        for f in fields(Thresholds):
            lo, hi = _THRESHOLD_BOUNDS[f.name]
            t_kw[f.name] = trial.suggest_float(f"t_{f.name}", lo, hi)
        w_kw = {}
        for f in fields(ScoreWeights):
            lo, hi = _weight_bounds(f.name)
            w_kw[f.name] = trial.suggest_float(f"w_{f.name}", lo, hi)
        cfg = RecommendationConfig(
            thresholds=Thresholds(**t_kw),
            weights=ScoreWeights(**w_kw))
        if self.objective_type == "top1":
            return self._top1(cfg)
        if self.objective_type == "topk":
            return self._topk(cfg)
        return self._regret(cfg)

    def _top1(self, cfg: RecommendationConfig) -> float:
        hits = sum(
            1 for r in self.dataset.records
            if compute_configurable_recommendation(
                r.metrics, cfg)["recommended_method"] == r.best_method)
        return hits / max(len(self.dataset.records), 1)

    def _topk(self, cfg: RecommendationConfig) -> float:
        hits = 0
        for rec in self.dataset.records:
            res = compute_configurable_recommendation(rec.metrics, cfg)
            ranked = sorted(res["method_scores"].items(),
                            key=lambda x: x[1], reverse=True)
            if rec.best_method in [n for n, _ in ranked[:self.top_k]]:
                hits += 1
        return hits / max(len(self.dataset.records), 1)

    def _regret(self, cfg: RecommendationConfig) -> float:
        total = 0.0
        for rec in self.dataset.records:
            res = compute_configurable_recommendation(rec.metrics, cfg)
            mr = rec.method_results.get(res["recommended_method"])
            total += rec.best_accuracy - (mr.accuracy if mr else 0.0)
        return -(total / max(len(self.dataset.records), 1))

    def tune(
        self, n_trials: int = RECOMMEND_N_TRIALS,
        output_path: str | None = None, seed: int = DEFAULT_RANDOM_SEED,
    ) -> RecommendationConfig:
        """Run optimization and return best config."""
        hpo = HPOConfig(
            n_trials=n_trials, direction="maximize",
            sampler="tpe", pruner=None, seed=seed)
        run = self.optimize(hpo)
        t_kw = {f.name: run.best_params[f"t_{f.name}"]
                for f in fields(Thresholds)}
        w_kw = {f.name: run.best_params[f"w_{f.name}"]
                for f in fields(ScoreWeights)}
        learned = RecommendationConfig(
            thresholds=Thresholds(**t_kw),
            weights=ScoreWeights(**w_kw))
        if output_path:
            learned.save(output_path)
            print(f"Saved learned config to {output_path}")
        print(f"Best agreement: {run.best_value:.4f}")
        return learned
