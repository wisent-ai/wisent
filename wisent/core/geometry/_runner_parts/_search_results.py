"""GeometrySearchResults and SteeringEvaluationResult dataclasses."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Any

from wisent.core.geometry import GeometrySearchConfig
from wisent.core.geometry._runner_parts._test_result import GeometryTestResult


@dataclass
class GeometrySearchResults:
    """Results from a full geometry search."""
    model_name: str
    config: GeometrySearchConfig
    results: List[GeometryTestResult] = field(default_factory=list)
    # Timing
    total_time_seconds: float = 0.0
    extraction_time_seconds: float = 0.0
    test_time_seconds: float = 0.0
    # Counts
    benchmarks_tested: int = 0
    strategies_tested: int = 0
    layer_combos_tested: int = 0

    def add_result(self, result: GeometryTestResult) -> None:
        self.results.append(result)

    def get_best_by_linear_score(self, n: int = 10) -> List[GeometryTestResult]:
        """Get top N configurations by linear score."""
        return sorted(self.results, key=lambda r: r.linear_score, reverse=True)[:n]

    def get_best_by_structure(self, structure: str, n: int = 10) -> List[GeometryTestResult]:
        """Get top N configurations by a specific structure score."""
        score_attr = f"{structure}_score"
        return sorted(
            self.results,
            key=lambda r: getattr(r, score_attr, 0.0),
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
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GeometrySearchResults":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        config = GeometrySearchConfig(
            pairs_per_benchmark=data.get("config", {}).get("pairs_per_benchmark", 50),
            max_layer_combo_size=data.get("config", {}).get("max_layer_combo_size", 3),
            random_seed=data.get("config", {}).get("random_seed", 42),
        )
        results = cls(
            model_name=data.get("model_name", "unknown"),
            config=config,
            total_time_seconds=data.get("total_time_seconds", 0.0),
            extraction_time_seconds=data.get("extraction_time_seconds", 0.0),
            test_time_seconds=data.get("test_time_seconds", 0.0),
            benchmarks_tested=data.get("benchmarks_tested", 0),
            strategies_tested=data.get("strategies_tested", 0),
            layer_combos_tested=data.get("layer_combos_tested", 0),
        )
        for r_data in data.get("results", []):
            try:
                result = GeometryTestResult(
                    benchmark=r_data.get("benchmark", "unknown"),
                    strategy=r_data.get("strategy", "unknown"),
                    layers=r_data.get("layers", []),
                    n_samples=r_data.get("n_samples", 0),
                    signal_strength=r_data.get("signal_strength", 0.5),
                    linear_score=r_data.get("linear_score", 0.0),
                    linear_probe_accuracy=r_data.get("linear_probe_accuracy", 0.5),
                    best_structure=r_data.get("best_structure", "unknown"),
                    is_linear=r_data.get("is_linear", False),
                    cohens_d=r_data.get("cohens_d", 0.0),
                    knn_accuracy_k5=r_data.get("nonlinear_signals", {}).get("knn_accuracy_k5", 0.5),
                    knn_accuracy_k10=r_data.get("nonlinear_signals", {}).get("knn_accuracy_k10", 0.5),
                    knn_accuracy_k20=r_data.get("nonlinear_signals", {}).get("knn_accuracy_k20", 0.5),
                    knn_pca_accuracy=r_data.get("nonlinear_signals", {}).get("knn_pca_accuracy", 0.5),
                    knn_umap_accuracy=r_data.get("nonlinear_signals", {}).get("knn_umap_accuracy", 0.5),
                    knn_pacmap_accuracy=r_data.get("nonlinear_signals", {}).get("knn_pacmap_accuracy", 0.5),
                    mlp_probe_accuracy=r_data.get("nonlinear_signals", {}).get("mlp_probe_accuracy", 0.5),
                    mmd_rbf=r_data.get("nonlinear_signals", {}).get("mmd_rbf", 0.0),
                    local_dim_pos=r_data.get("nonlinear_signals", {}).get("local_dim_pos", 0.0),
                    local_dim_neg=r_data.get("nonlinear_signals", {}).get("local_dim_neg", 0.0),
                    local_dim_ratio=r_data.get("nonlinear_signals", {}).get("local_dim_ratio", 1.0),
                    fisher_max=r_data.get("nonlinear_signals", {}).get("fisher_max", 0.0),
                    fisher_mean=r_data.get("nonlinear_signals", {}).get("fisher_mean", 0.0),
                    fisher_top10_mean=r_data.get("nonlinear_signals", {}).get("fisher_top10_mean", 0.0),
                    density_ratio=r_data.get("nonlinear_signals", {}).get("density_ratio", 1.0),
                    manifold_score=r_data.get("manifold_score", 0.0),
                    cluster_score=r_data.get("cluster_score", 0.0),
                    sparse_score=r_data.get("sparse_score", 0.0),
                    hybrid_score=r_data.get("hybrid_score", 0.0),
                    all_scores={},
                    pca_top2_variance=r_data.get("manifold_details", {}).get("pca_top2_variance", 0.0),
                    local_nonlinearity=r_data.get("manifold_details", {}).get("local_nonlinearity", 0.0),
                    gini_coefficient=r_data.get("sparse_details", {}).get("gini_coefficient", 0.0),
                    active_fraction=r_data.get("sparse_details", {}).get("active_fraction", 0.0),
                    top_10_contribution=r_data.get("sparse_details", {}).get("top_10_contribution", 0.0),
                    best_silhouette=r_data.get("cluster_details", {}).get("best_silhouette", 0.0),
                    best_k=r_data.get("cluster_details", {}).get("best_k", 2),
                    multi_dir_accuracy_k1=r_data.get("multi_direction_analysis", {}).get("accuracy_k1", 0.5),
                    multi_dir_accuracy_k2=r_data.get("multi_direction_analysis", {}).get("accuracy_k2", 0.5),
                    multi_dir_accuracy_k3=r_data.get("multi_direction_analysis", {}).get("accuracy_k3", 0.5),
                    multi_dir_accuracy_k5=r_data.get("multi_direction_analysis", {}).get("accuracy_k5", 0.5),
                    multi_dir_accuracy_k10=r_data.get("multi_direction_analysis", {}).get("accuracy_k10", 0.5),
                    multi_dir_min_k_for_good=r_data.get("multi_direction_analysis", {}).get("min_k_for_good", -1),
                    multi_dir_saturation_k=r_data.get("multi_direction_analysis", {}).get("saturation_k", 1),
                    multi_dir_gain=r_data.get("multi_direction_analysis", {}).get("gain_from_multi", 0.0),
                    diff_mean_alignment=r_data.get("steerability_metrics", {}).get("diff_mean_alignment", 0.0),
                    pct_positive_alignment=r_data.get("steerability_metrics", {}).get("pct_positive_alignment", 0.5),
                    steering_vector_norm_ratio=r_data.get("steerability_metrics", {}).get("steering_vector_norm_ratio", 0.0),
                    cluster_direction_angle=r_data.get("steerability_metrics", {}).get("cluster_direction_angle", 90.0),
                    per_cluster_alignment_k2=r_data.get("steerability_metrics", {}).get("per_cluster_alignment_k2", 0.0),
                    spherical_silhouette_k2=r_data.get("steerability_metrics", {}).get("spherical_silhouette_k2", 0.0),
                    effective_steering_dims=r_data.get("steerability_metrics", {}).get("effective_steering_dims", 1),
                    steerability_score=r_data.get("steerability_metrics", {}).get("steerability_score", 0.0),
                    icd=r_data.get("icd_metrics", {}).get("icd", 0.0),
                    icd_top1_variance=r_data.get("icd_metrics", {}).get("top1_variance", 0.0),
                    icd_top5_variance=r_data.get("icd_metrics", {}).get("top5_variance", 0.0),
                    nonsense_baseline_accuracy=r_data.get("nonsense_baseline", {}).get("nonsense_accuracy", 0.5),
                    signal_vs_baseline_ratio=r_data.get("nonsense_baseline", {}).get("signal_ratio", 1.0),
                    signal_above_baseline=r_data.get("nonsense_baseline", {}).get("signal_above_baseline", 0.0),
                    has_real_signal=r_data.get("nonsense_baseline", {}).get("has_real_signal", True),
                    recommended_method=r_data.get("recommended_method", "unknown"),
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
