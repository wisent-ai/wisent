"""GeometrySearchResults and SteeringEvaluationResult dataclasses."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Any

from wisent.core.utils.config_tools.constants import (
    BLEND_DEFAULT,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SCALE,
    DEFAULT_SCORE,
    GEO_DEFAULT_DIRECTION_ANGLE,
    GEO_MAX_LAYER_COMBO_SIZE,
    GEO_PAIRS_PER_BENCHMARK,
    JSON_INDENT,
    MIN_CONCEPT_DIM,
    MULTI_DIR_MIN_K_NOT_FOUND,
    MULTI_DIR_SATURATION_K_DEFAULT,
    SEARCH_RESULTS_TOP_N,
)
from wisent.core.reading.modules import GeometrySearchConfig
from wisent.core.reading.modules._runner_parts._test_result import GeometryTestResult


@dataclass
class GeometrySearchResults:
    """Results from a full geometry search."""
    model_name: str
    config: GeometrySearchConfig
    results: List[GeometryTestResult] = field(default_factory=list)
    # Timing
    total_time_seconds: float = DEFAULT_SCORE
    extraction_time_seconds: float = DEFAULT_SCORE
    test_time_seconds: float = DEFAULT_SCORE
    # Counts
    benchmarks_tested: int = 0
    strategies_tested: int = 0
    layer_combos_tested: int = 0

    def add_result(self, result: GeometryTestResult) -> None:
        self.results.append(result)

    def get_best_by_linear_score(self, n: int = SEARCH_RESULTS_TOP_N) -> List[GeometryTestResult]:
        """Get top N configurations by linear score."""
        return sorted(self.results, key=lambda r: r.linear_score, reverse=True)[:n]

    def get_best_by_structure(self, structure: str, n: int = SEARCH_RESULTS_TOP_N) -> List[GeometryTestResult]:
        """Get top N configurations by a specific structure score."""
        score_attr = f"{structure}_score"
        return sorted(
            self.results,
            key=lambda r: getattr(r, score_attr, DEFAULT_SCORE),
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
            pairs_per_benchmark=data.get("config", {}).get("pairs_per_benchmark", GEO_PAIRS_PER_BENCHMARK),
            max_layer_combo_size=data.get("config", {}).get("max_layer_combo_size", GEO_MAX_LAYER_COMBO_SIZE),
            random_seed=data.get("config", {}).get("random_seed", DEFAULT_RANDOM_SEED),
        )
        results = cls(
            model_name=data.get("model_name", "unknown"),
            config=config,
            total_time_seconds=data.get("total_time_seconds", DEFAULT_SCORE),
            extraction_time_seconds=data.get("extraction_time_seconds", DEFAULT_SCORE),
            test_time_seconds=data.get("test_time_seconds", DEFAULT_SCORE),
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
                    signal_strength=r_data.get("signal_strength", BLEND_DEFAULT),
                    linear_score=r_data.get("linear_score", DEFAULT_SCORE),
                    linear_probe_accuracy=r_data.get("linear_probe_accuracy", BLEND_DEFAULT),
                    best_structure=r_data.get("best_structure", "unknown"),
                    is_linear=r_data.get("is_linear", False),
                    cohens_d=r_data.get("cohens_d", DEFAULT_SCORE),
                    knn_accuracy_k5=r_data.get("nonlinear_signals", {}).get("knn_accuracy_k5", BLEND_DEFAULT),
                    knn_accuracy_k10=r_data.get("nonlinear_signals", {}).get("knn_accuracy_k10", BLEND_DEFAULT),
                    knn_accuracy_k20=r_data.get("nonlinear_signals", {}).get("knn_accuracy_k20", BLEND_DEFAULT),
                    knn_pca_accuracy=r_data.get("nonlinear_signals", {}).get("knn_pca_accuracy", BLEND_DEFAULT),
                    knn_umap_accuracy=r_data.get("nonlinear_signals", {}).get("knn_umap_accuracy", BLEND_DEFAULT),
                    knn_pacmap_accuracy=r_data.get("nonlinear_signals", {}).get("knn_pacmap_accuracy", BLEND_DEFAULT),
                    mlp_probe_accuracy=r_data.get("nonlinear_signals", {}).get("mlp_probe_accuracy", BLEND_DEFAULT),
                    mmd_rbf=r_data.get("nonlinear_signals", {}).get("mmd_rbf", DEFAULT_SCORE),
                    local_dim_pos=r_data.get("nonlinear_signals", {}).get("local_dim_pos", DEFAULT_SCORE),
                    local_dim_neg=r_data.get("nonlinear_signals", {}).get("local_dim_neg", DEFAULT_SCORE),
                    local_dim_ratio=r_data.get("nonlinear_signals", {}).get("local_dim_ratio", DEFAULT_SCALE),
                    fisher_max=r_data.get("nonlinear_signals", {}).get("fisher_max", DEFAULT_SCORE),
                    fisher_mean=r_data.get("nonlinear_signals", {}).get("fisher_mean", DEFAULT_SCORE),
                    fisher_top10_mean=r_data.get("nonlinear_signals", {}).get("fisher_top10_mean", DEFAULT_SCORE),
                    density_ratio=r_data.get("nonlinear_signals", {}).get("density_ratio", DEFAULT_SCALE),
                    manifold_score=r_data.get("manifold_score", DEFAULT_SCORE),
                    cluster_score=r_data.get("cluster_score", DEFAULT_SCORE),
                    sparse_score=r_data.get("sparse_score", DEFAULT_SCORE),
                    hybrid_score=r_data.get("hybrid_score", DEFAULT_SCORE),
                    all_scores={},
                    pca_top2_variance=r_data.get("manifold_details", {}).get("pca_top2_variance", DEFAULT_SCORE),
                    local_nonlinearity=r_data.get("manifold_details", {}).get("local_nonlinearity", DEFAULT_SCORE),
                    gini_coefficient=r_data.get("sparse_details", {}).get("gini_coefficient", DEFAULT_SCORE),
                    active_fraction=r_data.get("sparse_details", {}).get("active_fraction", DEFAULT_SCORE),
                    top_10_contribution=r_data.get("sparse_details", {}).get("top_10_contribution", DEFAULT_SCORE),
                    best_silhouette=r_data.get("cluster_details", {}).get("best_silhouette", DEFAULT_SCORE),
                    best_k=r_data.get("cluster_details", {}).get("best_k", MIN_CONCEPT_DIM),
                    multi_dir_accuracy_k1=r_data.get("multi_direction_analysis", {}).get("accuracy_k1", BLEND_DEFAULT),
                    multi_dir_accuracy_k2=r_data.get("multi_direction_analysis", {}).get("accuracy_k2", BLEND_DEFAULT),
                    multi_dir_accuracy_k3=r_data.get("multi_direction_analysis", {}).get("accuracy_k3", BLEND_DEFAULT),
                    multi_dir_accuracy_k5=r_data.get("multi_direction_analysis", {}).get("accuracy_k5", BLEND_DEFAULT),
                    multi_dir_accuracy_k10=r_data.get("multi_direction_analysis", {}).get("accuracy_k10", BLEND_DEFAULT),
                    multi_dir_min_k_for_good=r_data.get("multi_direction_analysis", {}).get("min_k_for_good", MULTI_DIR_MIN_K_NOT_FOUND),
                    multi_dir_saturation_k=r_data.get("multi_direction_analysis", {}).get("saturation_k", MULTI_DIR_SATURATION_K_DEFAULT),
                    multi_dir_gain=r_data.get("multi_direction_analysis", {}).get("gain_from_multi", DEFAULT_SCORE),
                    diff_mean_alignment=r_data.get("steerability_metrics", {}).get("diff_mean_alignment", DEFAULT_SCORE),
                    pct_positive_alignment=r_data.get("steerability_metrics", {}).get("pct_positive_alignment", BLEND_DEFAULT),
                    steering_vector_norm_ratio=r_data.get("steerability_metrics", {}).get("steering_vector_norm_ratio", DEFAULT_SCORE),
                    cluster_direction_angle=r_data.get("steerability_metrics", {}).get("cluster_direction_angle", GEO_DEFAULT_DIRECTION_ANGLE),
                    per_cluster_alignment_k2=r_data.get("steerability_metrics", {}).get("per_cluster_alignment_k2", DEFAULT_SCORE),
                    spherical_silhouette_k2=r_data.get("steerability_metrics", {}).get("spherical_silhouette_k2", DEFAULT_SCORE),
                    effective_steering_dims=r_data.get("steerability_metrics", {}).get("effective_steering_dims", MULTI_DIR_SATURATION_K_DEFAULT),
                    steerability_score=r_data.get("steerability_metrics", {}).get("steerability_score", DEFAULT_SCORE),
                    icd=r_data.get("icd_metrics", {}).get("icd", DEFAULT_SCORE),
                    icd_top1_variance=r_data.get("icd_metrics", {}).get("top1_variance", DEFAULT_SCORE),
                    icd_top5_variance=r_data.get("icd_metrics", {}).get("top5_variance", DEFAULT_SCORE),
                    nonsense_baseline_accuracy=r_data.get("nonsense_baseline", {}).get("nonsense_accuracy", BLEND_DEFAULT),
                    signal_vs_baseline_ratio=r_data.get("nonsense_baseline", {}).get("signal_ratio", DEFAULT_SCALE),
                    signal_above_baseline=r_data.get("nonsense_baseline", {}).get("signal_above_baseline", DEFAULT_SCORE),
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
