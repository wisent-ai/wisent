"""
Geometry search runner.

Runs geometry tests across the search space using cached activations.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import torch

from wisent.core.geometry_search_space import GeometrySearchSpace, GeometrySearchConfig
from wisent.core.activations.extraction_strategy import ExtractionStrategy
from wisent.core.activations.activation_cache import (
    ActivationCache,
    CachedActivations,
    collect_and_cache_activations,
)
from wisent.core.utils.layer_combinations import get_layer_combinations


@dataclass
class GeometryTestResult:
    """Result of a single geometry test."""
    benchmark: str
    strategy: str
    layers: List[int]
    
    # Best structure detected
    best_structure: str  # 'linear', 'cone', 'cluster', 'manifold', 'sparse', 'bimodal', 'orthogonal'
    best_score: float
    
    # All structure scores
    linear_score: float
    cone_score: float
    orthogonal_score: float
    manifold_score: float
    sparse_score: float
    cluster_score: float
    bimodal_score: float
    
    # Detailed metrics per structure
    # Linear
    cohens_d: float  # separation quality
    variance_explained: float  # by primary direction
    within_class_consistency: float
    
    # Cone
    raw_mean_cosine_similarity: float  # between diff vectors
    positive_correlation_fraction: float  # fraction in same half-space
    
    # Orthogonal
    near_zero_fraction: float  # fraction of near-zero correlations
    
    # Manifold
    pca_top2_variance: float  # variance by top 2 PCs
    local_nonlinearity: float  # curvature measure
    
    # Sparse
    gini_coefficient: float  # inequality of activations
    active_fraction: float  # fraction of active neurons
    top_10_contribution: float  # contribution of top 10 neurons
    
    # Cluster
    best_silhouette: float  # clustering quality
    best_k: int  # optimal number of clusters
    
    # Recommendation
    recommended_method: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "strategy": self.strategy,
            "layers": self.layers,
            "best_structure": self.best_structure,
            "best_score": self.best_score,
            "structure_scores": {
                "linear": self.linear_score,
                "cone": self.cone_score,
                "orthogonal": self.orthogonal_score,
                "manifold": self.manifold_score,
                "sparse": self.sparse_score,
                "cluster": self.cluster_score,
                "bimodal": self.bimodal_score,
            },
            "linear_details": {
                "cohens_d": self.cohens_d,
                "variance_explained": self.variance_explained,
                "within_class_consistency": self.within_class_consistency,
            },
            "cone_details": {
                "raw_mean_cosine_similarity": self.raw_mean_cosine_similarity,
                "positive_correlation_fraction": self.positive_correlation_fraction,
            },
            "orthogonal_details": {
                "near_zero_fraction": self.near_zero_fraction,
            },
            "manifold_details": {
                "pca_top2_variance": self.pca_top2_variance,
                "local_nonlinearity": self.local_nonlinearity,
            },
            "sparse_details": {
                "gini_coefficient": self.gini_coefficient,
                "active_fraction": self.active_fraction,
                "top_10_contribution": self.top_10_contribution,
            },
            "cluster_details": {
                "best_silhouette": self.best_silhouette,
                "best_k": self.best_k,
            },
            "recommended_method": self.recommended_method,
        }


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


def compute_geometry_metrics(
    cached: CachedActivations,
    layers: List[int],
) -> GeometryTestResult:
    """
    Compute geometry metrics for a layer combination from cached activations.
    
    Uses the comprehensive detect_geometry_structure() to get scores for:
    - linear, cone, cluster, manifold, sparse, bimodal, orthogonal
    
    Args:
        cached: Cached activations with all layers
        layers: Layer indices (0-based) to analyze
        
    Returns:
        GeometryTestResult with all structure scores
    """
    from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
        detect_geometry_structure,
        GeometryAnalysisConfig,
    )
    
    # Stack positive and negative activations for specified layers
    # Convert 0-based indices to 1-based layer names used in cache
    pos_acts_list = []
    neg_acts_list = []
    
    for layer_idx in layers:
        layer_name = str(layer_idx + 1)  # Convert 0-based to 1-based
        try:
            pos = cached.get_positive_activations(layer_name)  # [num_pairs, hidden_size]
            neg = cached.get_negative_activations(layer_name)  # [num_pairs, hidden_size]
            pos_acts_list.append(pos)
            neg_acts_list.append(neg)
        except (KeyError, IndexError):
            continue
    
    if not pos_acts_list:
        return GeometryTestResult(
            benchmark=cached.benchmark,
            strategy=cached.strategy.value,
            layers=layers,
            best_structure="error",
            best_score=0.0,
            linear_score=0.0,
            cone_score=0.0,
            orthogonal_score=0.0,
            manifold_score=0.0,
            sparse_score=0.0,
            cluster_score=0.0,
            bimodal_score=0.0,
            cohens_d=0.0,
            variance_explained=0.0,
            within_class_consistency=0.0,
            raw_mean_cosine_similarity=0.0,
            positive_correlation_fraction=0.0,
            near_zero_fraction=0.0,
            pca_top2_variance=0.0,
            local_nonlinearity=0.0,
            gini_coefficient=0.0,
            active_fraction=0.0,
            top_10_contribution=0.0,
            best_silhouette=0.0,
            best_k=0,
            recommended_method="error: no activations",
        )
    
    # Concatenate across layers: [num_pairs, hidden_size * num_layers]
    pos_activations = torch.cat(pos_acts_list, dim=-1)
    neg_activations = torch.cat(neg_acts_list, dim=-1)
    
    # Convert to float32 for geometry analysis (bf16/float16 can cause dtype mismatches)
    pos_activations = pos_activations.float()
    neg_activations = neg_activations.float()
    
    # Run comprehensive geometry detection
    config = GeometryAnalysisConfig(
        num_components=5,
        optimization_steps=50,  # Reduced for speed since we're testing many combos
    )
    
    try:
        result = detect_geometry_structure(pos_activations, neg_activations, config)
        
        # Helper to safely get detail
        def get_detail(struct_name: str, key: str, default=0.0):
            if struct_name in result.all_scores:
                return result.all_scores[struct_name].details.get(key, default)
            return default
        
        return GeometryTestResult(
            benchmark=cached.benchmark,
            strategy=cached.strategy.value,
            layers=layers,
            best_structure=result.best_structure.value,
            best_score=result.best_score,
            # Structure scores
            linear_score=result.all_scores.get("linear", type('', (), {'score': 0.0})()).score,
            cone_score=result.all_scores.get("cone", type('', (), {'score': 0.0})()).score,
            orthogonal_score=result.all_scores.get("orthogonal", type('', (), {'score': 0.0})()).score,
            manifold_score=result.all_scores.get("manifold", type('', (), {'score': 0.0})()).score,
            sparse_score=result.all_scores.get("sparse", type('', (), {'score': 0.0})()).score,
            cluster_score=result.all_scores.get("cluster", type('', (), {'score': 0.0})()).score,
            bimodal_score=result.all_scores.get("bimodal", type('', (), {'score': 0.0})()).score,
            # Linear details
            cohens_d=get_detail("linear", "cohens_d", 0.0),
            variance_explained=get_detail("linear", "variance_explained", 0.0),
            within_class_consistency=get_detail("linear", "within_class_consistency", 0.0),
            # Cone details
            raw_mean_cosine_similarity=get_detail("cone", "raw_mean_cosine_similarity", 0.0),
            positive_correlation_fraction=get_detail("cone", "positive_correlation_fraction", 0.0),
            # Orthogonal details
            near_zero_fraction=get_detail("orthogonal", "near_zero_fraction", 0.0),
            # Manifold details
            pca_top2_variance=get_detail("manifold", "pca_top2_variance", 0.0),
            local_nonlinearity=get_detail("manifold", "local_nonlinearity", 0.0),
            # Sparse details
            gini_coefficient=get_detail("sparse", "gini_coefficient", 0.0),
            active_fraction=get_detail("sparse", "active_fraction", 0.0),
            top_10_contribution=get_detail("sparse", "top_10_contribution", 0.0),
            # Cluster details
            best_silhouette=get_detail("cluster", "best_silhouette", 0.0),
            best_k=int(get_detail("cluster", "best_k", 2)),
            # Recommendation
            recommended_method=result.recommendation,
        )
    except Exception as e:
        return GeometryTestResult(
            benchmark=cached.benchmark,
            strategy=cached.strategy.value,
            layers=layers,
            best_structure="error",
            best_score=0.0,
            linear_score=0.0,
            cone_score=0.0,
            orthogonal_score=0.0,
            manifold_score=0.0,
            sparse_score=0.0,
            cluster_score=0.0,
            bimodal_score=0.0,
            cohens_d=0.0,
            variance_explained=0.0,
            within_class_consistency=0.0,
            raw_mean_cosine_similarity=0.0,
            positive_correlation_fraction=0.0,
            near_zero_fraction=0.0,
            pca_top2_variance=0.0,
            local_nonlinearity=0.0,
            gini_coefficient=0.0,
            active_fraction=0.0,
            top_10_contribution=0.0,
            best_silhouette=0.0,
            best_k=0,
            recommended_method=f"error: {str(e)}",
        )


class GeometryRunner:
    """
    Runs geometry search across the search space.
    
    Uses activation caching for efficiency:
    1. Extract ALL layers once per (benchmark, strategy)
    2. Test all layer combinations from cache
    """
    
    def __init__(
        self,
        search_space: GeometrySearchSpace,
        model: "WisentModel",
        cache_dir: Optional[str] = None,
    ):
        self.search_space = search_space
        self.model = model
        self.cache_dir = cache_dir or f"/tmp/wisent_geometry_cache_{model.model_name.replace('/', '_')}"
        self.cache = ActivationCache(self.cache_dir)
    
    def run(
        self,
        benchmarks: Optional[List[str]] = None,
        strategies: Optional[List[ExtractionStrategy]] = None,
        max_layer_combo_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> GeometrySearchResults:
        """
        Run the geometry search.
        
        Args:
            benchmarks: Benchmarks to test (default: all from search space)
            strategies: Strategies to test (default: all from search space)
            max_layer_combo_size: Override max layer combo size
            show_progress: Print progress
            
        Returns:
            GeometrySearchResults with all test results
        """
        benchmarks = benchmarks or self.search_space.benchmarks
        strategies = strategies or self.search_space.strategies
        max_combo = max_layer_combo_size or self.search_space.config.max_layer_combo_size
        
        # Get layer combinations
        num_layers = self.model.num_layers
        layer_combos = get_layer_combinations(num_layers, max_combo)
        
        results = GeometrySearchResults(
            model_name=self.model.model_name,
            config=self.search_space.config,
        )
        
        start_time = time.time()
        extraction_time = 0.0
        test_time = 0.0
        
        total_extractions = len(benchmarks) * len(strategies)
        extraction_count = 0
        
        for benchmark in benchmarks:
            for strategy in strategies:
                extraction_count += 1
                
                if show_progress:
                    print(f"\n[{extraction_count}/{total_extractions}] {benchmark} / {strategy.value}")
                
                # Get or create cached activations
                extract_start = time.time()
                try:
                    cached = self._get_cached_activations(benchmark, strategy, show_progress)
                except Exception as e:
                    if show_progress:
                        print(f"  SKIP: {e}")
                    continue
                extraction_time += time.time() - extract_start
                
                # Test all layer combinations
                test_start = time.time()
                for combo in layer_combos:
                    result = compute_geometry_metrics(cached, combo)
                    results.add_result(result)
                test_time += time.time() - test_start
                
                results.benchmarks_tested = len(set(r.benchmark for r in results.results))
                results.strategies_tested = len(set(r.strategy for r in results.results))
                results.layer_combos_tested = len(results.results)
                
                if show_progress:
                    print(f"  Tested {len(layer_combos)} layer combos")
        
        results.total_time_seconds = time.time() - start_time
        results.extraction_time_seconds = extraction_time
        results.test_time_seconds = test_time
        
        return results
    
    def _get_cached_activations(
        self,
        benchmark: str,
        strategy: ExtractionStrategy,
        show_progress: bool = True,
    ) -> CachedActivations:
        """Get cached activations, extracting if necessary."""
        # Check cache
        if self.cache.has(self.model.model_name, benchmark, strategy):
            if show_progress:
                print(f"  Loading from cache...")
            return self.cache.get(self.model.model_name, benchmark, strategy)
        
        # Need to extract - load pairs first
        if show_progress:
            print(f"  Loading pairs...")
        
        pairs = self._load_pairs(benchmark)
        
        if show_progress:
            print(f"  Extracting activations for {len(pairs)} pairs...")
        
        return collect_and_cache_activations(
            model=self.model,
            pairs=pairs,
            benchmark=benchmark,
            strategy=strategy,
            cache=self.cache,
            show_progress=show_progress,
        )
    
    def _load_pairs(self, benchmark: str) -> List:
        """Load contrastive pairs for a benchmark."""
        from lm_eval.tasks import TaskManager
        from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
        
        tm = TaskManager()
        try:
            task_dict = tm.load_task_or_group([benchmark])
            task = list(task_dict.values())[0]
        except Exception:
            task = None
        
        pairs = lm_build_contrastive_pairs(
            benchmark, 
            task, 
            limit=self.search_space.config.pairs_per_benchmark
        )
        
        # Random sample if we have more pairs than needed
        if len(pairs) > self.search_space.config.pairs_per_benchmark:
            random.seed(self.search_space.config.random_seed)
            pairs = random.sample(pairs, self.search_space.config.pairs_per_benchmark)
        
        return pairs


# Type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wisent.core.models.wisent_model import WisentModel
