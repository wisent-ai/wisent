"""Recommendation tuning sub-package."""
from .config import RecommendationConfig, Thresholds, ScoreWeights, METHODS
from .configurable import compute_configurable_recommendation
from .collector import (
    MethodResult, BenchmarkGroundTruth, GroundTruthDataset,
    collect_ground_truth, collect_benchmark_ground_truth,
)
from .optimizer import RecommendationOptimizer
from wisent.core.geometry.zwiad.geometry_types import (
    GeometryType, GeometryTypeFine, GeometryProfile,
    SHAPE_MAP, METHOD_MAP, classify_geometry,
    select_representative_benchmarks,
)

__all__ = [
    "RecommendationConfig", "Thresholds", "ScoreWeights", "METHODS",
    "compute_configurable_recommendation",
    "MethodResult", "BenchmarkGroundTruth", "GroundTruthDataset",
    "collect_ground_truth", "collect_benchmark_ground_truth",
    "RecommendationOptimizer",
    "GeometryType", "GeometryTypeFine", "GeometryProfile",
    "SHAPE_MAP", "METHOD_MAP", "classify_geometry",
    "select_representative_benchmarks",
]
