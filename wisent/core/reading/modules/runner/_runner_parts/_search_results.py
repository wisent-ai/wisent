"""GeometrySearchResults and SteeringEvaluationResult dataclasses."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Any

from wisent.core.utils.config_tools.constants import (
    JSON_INDENT,
)
from wisent.core.reading.modules import GeometrySearchConfig
from wisent.core.reading.modules.runner._runner_parts._test_result import GeometryTestResult


@dataclass
class GeometrySearchResults:
    """Results from a full geometry search."""
    model_name: str
    config: GeometrySearchConfig
    results: List[GeometryTestResult] = field(default_factory=list)
    # Timing
    total_time_seconds: float = None
    extraction_time_seconds: float = None
    test_time_seconds: float = None
    # Counts
    benchmarks_tested: int = 0
    strategies_tested: int = 0
    layer_combos_tested: int = 0

    def add_result(self, result: GeometryTestResult) -> None:
        self.results.append(result)

    def get_best_by_linear_score(self, n: int) -> List[GeometryTestResult]:
        """Get top N configurations by linear score."""
        return sorted(self.results, key=lambda r: r.linear_score, reverse=True)[:n]

    def get_best_by_structure(self, structure: str, n: int) -> List[GeometryTestResult]:
        """Get top N configurations by a specific structure score."""
        score_attr = f"{structure}_score"
        return sorted(
            self.results,
            key=lambda r: getattr(r, score_attr),
            reverse=True
        )[:n]

    def get_structure_distribution(self) -> Dict[str, int]:
        """Count how many configurations have each structure as best."""
        counts: Dict[str, int] = {}
        for r in self.results:
            s = r.best_structure
            counts[s] = counts.get(s, 0) + 1
        return counts

    def get_summary_by_benchmark(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics grouped by benchmark."""
        by_bench: Dict[str, List[float]] = {}
        for r in self.results:
            if r.benchmark not in by_bench:
                by_bench[r.benchmark] = []
            by_bench[r.benchmark].append(r.linear_score)
        return {
            bench: {
                "mean": sum(scores) / len(scores),
                "max": max(scores),
                "min": min(scores),
                "count": len(scores),
            }
            for bench, scores in by_bench.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "config": self.config.to_dict(),
            "total_time_seconds": self.total_time_seconds,
            "extraction_time_seconds": self.extraction_time_seconds,
            "test_time_seconds": self.test_time_seconds,
            "benchmarks_tested": self.benchmarks_tested,
            "strategies_tested": self.strategies_tested,
            "layer_combos_tested": self.layer_combos_tested,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=JSON_INDENT)

    @classmethod
    def load(cls, path: str) -> "GeometrySearchResults":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        config = GeometrySearchConfig(
            pairs_per_benchmark=data["config"]["pairs_per_benchmark"],
            max_layer_combo_size=data["config"]["max_layer_combo_size"],
            random_seed=data["config"]["random_seed"],
        )
        results = cls(
            model_name=data["model_name"],
            config=config,
            total_time_seconds=data["total_time_seconds"],
            extraction_time_seconds=data["extraction_time_seconds"],
            test_time_seconds=data["test_time_seconds"],
            benchmarks_tested=data["benchmarks_tested"],
            strategies_tested=data["strategies_tested"],
            layer_combos_tested=data["layer_combos_tested"],
        )
        for r_data in data["results"]:
            try:
                nonlinear = r_data["nonlinear_signals"]
                manifold_d = r_data["manifold_details"]
                sparse_d = r_data["sparse_details"]
                cluster_d = r_data["cluster_details"]
                multi_dir = r_data["multi_direction_analysis"]
                steer_m = r_data["steerability_metrics"]
                icd_m = r_data["icd_metrics"]
                nonsense_b = r_data["nonsense_baseline"]
                result = GeometryTestResult(
                    benchmark=r_data["benchmark"],
                    strategy=r_data["strategy"],
                    layers=r_data["layers"],
                    n_samples=r_data["n_samples"],
                    signal_strength=r_data["signal_strength"],
                    linear_score=r_data["linear_score"],
                    linear_probe_accuracy=r_data["linear_probe_accuracy"],
                    best_structure=r_data["best_structure"],
                    is_linear=r_data["is_linear"],
                    cohens_d=r_data["cohens_d"],
                    knn_accuracy_k5=nonlinear["knn_accuracy_k5"],
                    knn_accuracy_k10=nonlinear["knn_accuracy_k10"],
                    knn_accuracy_k20=nonlinear["knn_accuracy_k20"],
                    knn_pca_accuracy=nonlinear["knn_pca_accuracy"],
                    knn_umap_accuracy=nonlinear["knn_umap_accuracy"],
                    knn_pacmap_accuracy=nonlinear["knn_pacmap_accuracy"],
                    mlp_probe_accuracy=nonlinear["mlp_probe_accuracy"],
                    mmd_rbf=nonlinear["mmd_rbf"],
                    local_dim_pos=nonlinear["local_dim_pos"],
                    local_dim_neg=nonlinear["local_dim_neg"],
                    local_dim_ratio=nonlinear["local_dim_ratio"],
                    fisher_max=nonlinear["fisher_max"],
                    fisher_mean=nonlinear["fisher_mean"],
                    fisher_top10_mean=nonlinear["fisher_top10_mean"],
                    density_ratio=nonlinear["density_ratio"],
                    manifold_score=r_data["manifold_score"],
                    cluster_score=r_data["cluster_score"],
                    sparse_score=r_data["sparse_score"],
                    hybrid_score=r_data["hybrid_score"],
                    all_scores={},
                    pca_top2_variance=manifold_d["pca_top2_variance"],
                    local_nonlinearity=manifold_d["local_nonlinearity"],
                    gini_coefficient=sparse_d["gini_coefficient"],
                    active_fraction=sparse_d["active_fraction"],
                    top_10_contribution=sparse_d["top_10_contribution"],
                    best_silhouette=cluster_d["best_silhouette"],
                    best_k=cluster_d["best_k"],
                    multi_dir_accuracy_k1=multi_dir["accuracy_k1"],
                    multi_dir_accuracy_k2=multi_dir["accuracy_k2"],
                    multi_dir_accuracy_k3=multi_dir["accuracy_k3"],
                    multi_dir_accuracy_k5=multi_dir["accuracy_k5"],
                    multi_dir_accuracy_k10=multi_dir["accuracy_k10"],
                    multi_dir_min_k_for_good=multi_dir["min_k_for_good"],
                    multi_dir_saturation_k=multi_dir["saturation_k"],
                    multi_dir_gain=multi_dir["gain_from_multi"],
                    diff_mean_alignment=steer_m["diff_mean_alignment"],
                    pct_positive_alignment=steer_m["pct_positive_alignment"],
                    steering_vector_norm_ratio=steer_m["steering_vector_norm_ratio"],
                    cluster_direction_angle=steer_m["cluster_direction_angle"],
                    per_cluster_alignment_k2=steer_m["per_cluster_alignment_k2"],
                    spherical_silhouette_k2=steer_m["spherical_silhouette_k2"],
                    effective_steering_dims=steer_m["effective_steering_dims"],
                    steerability_score=steer_m["steerability_score"],
                    icd=icd_m["icd"],
                    icd_top1_variance=icd_m["top1_variance"],
                    icd_top5_variance=icd_m["top5_variance"],
                    nonsense_baseline_accuracy=nonsense_b["nonsense_accuracy"],
                    signal_vs_baseline_ratio=nonsense_b["signal_ratio"],
                    signal_above_baseline=nonsense_b["signal_above_baseline"],
                    has_real_signal=nonsense_b["has_real_signal"],
                    recommended_method=r_data["recommended_method"],
                )
                results.results.append(result)
            except Exception:
                pass
        return results


@dataclass
class SteeringEvaluationResult:
    """Result from evaluating steering effectiveness."""
    steering_strength: float
    neg_to_pos_shift: float
    pos_stability: float
    separation_after: float
    neg_classified_as_pos_before: float
    neg_classified_as_pos_after: float
    flip_rate: float
    cosine_shift_neg: float
    magnitude_ratio: float
    steering_effective: bool
    optimal_strength: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steering_strength": self.steering_strength,
            "neg_to_pos_shift": self.neg_to_pos_shift,
            "pos_stability": self.pos_stability,
            "separation_after": self.separation_after,
            "neg_classified_as_pos_before": self.neg_classified_as_pos_before,
            "neg_classified_as_pos_after": self.neg_classified_as_pos_after,
            "flip_rate": self.flip_rate,
            "cosine_shift_neg": self.cosine_shift_neg,
            "magnitude_ratio": self.magnitude_ratio,
            "steering_effective": self.steering_effective,
            "optimal_strength": self.optimal_strength,
        }
