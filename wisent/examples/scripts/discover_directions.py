"""
Discover unified directions for skill categories (coding, math, hallucination, etc.)

Uses GeometrySearchSpace to test all models, strategies, and layer combinations.
For each category, determines if a unified direction exists.

Usage:
    # Run for all models (sequentially)
    python -m wisent.examples.scripts.discover_directions
    
    # Run for a specific model (for parallel execution)
    python -m wisent.examples.scripts.discover_directions --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

S3_BUCKET = "wisent-bucket"
S3_PREFIX = "direction_discovery"


def s3_sync_download(model_name: str, output_dir: Path) -> None:
    """Download existing results from S3."""
    model_prefix = model_name.replace('/', '_')
    s3_path = f"s3://{S3_BUCKET}/{S3_PREFIX}/{model_prefix}/"
    try:
        subprocess.run(
            ["aws", "s3", "sync", s3_path, str(output_dir), "--quiet"],
            check=False,
            capture_output=True,
        )
        print(f"Synced existing results from S3: {s3_path}")
    except Exception as e:
        print(f"S3 download skipped: {e}")


def s3_upload_file(local_path: Path, model_name: str) -> None:
    """Upload a single file to S3."""
    model_prefix = model_name.replace('/', '_')
    s3_path = f"s3://{S3_BUCKET}/{S3_PREFIX}/{model_prefix}/{local_path.name}"
    try:
        subprocess.run(
            ["aws", "s3", "cp", str(local_path), s3_path, "--quiet"],
            check=True,
            capture_output=True,
        )
        print(f"  Uploaded to S3: {s3_path}")
    except Exception as e:
        print(f"  S3 upload failed: {e}")

from wisent.core.geometry_search_space import (
    GeometrySearchSpace,
    GeometrySearchConfig,
)
from wisent.core.geometry_runner import (
    GeometryRunner,
    GeometrySearchResults,
    GeometryTestResult,
    compute_icd,
    compute_nonsense_baseline,
    generate_nonsense_activations,
    analyze_with_nonsense_baseline,
    compute_recommendation,
    should_increase_pairs,
    compute_adaptive_recommendation,
)
from wisent.core.activations.extraction_strategy import ExtractionStrategy
from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
    GeometryAnalysisConfig,
    StructureType,
)
from wisent.core.models.wisent_model import WisentModel


def load_categorized_benchmarks() -> Dict[str, List[str]]:
    """Load benchmarks grouped by category."""
    params_dir = Path(__file__).parent.parent.parent / "parameters" / "lm_eval"
    with open(params_dir / "working_benchmarks_categorized.json") as f:
        return json.load(f)


def load_category_directions() -> Dict[str, Dict]:
    """Load hypothesized directions for each category."""
    params_dir = Path(__file__).parent.parent.parent / "parameters" / "lm_eval"
    with open(params_dir / "category_directions.json") as f:
        return json.load(f)


@dataclass
class OptimalConfig:
    """Optimal configuration for extracting a concept."""
    # Best layer configuration
    optimal_layer: int  # Single best layer
    optimal_layer_range: List[int]  # Top 3 layers within 5% of best
    layer_accuracy: float  # Accuracy at optimal layer
    
    # Best extraction strategy
    optimal_strategy: str  # e.g., "last_token", "mean_diff"
    strategy_accuracy: float
    
    # Minimum pairs needed
    min_pairs_for_stable_signal: int  # Minimum n for accuracy within 5% of max
    pairs_saturation_curve: Dict[int, float]  # n_pairs -> accuracy
    
    # Steering configuration
    num_directions_needed: int  # k for multi-direction steering
    single_direction_accuracy: float  # accuracy with k=1
    multi_direction_accuracy: float  # accuracy with optimal k
    steering_gain: float  # multi - single direction accuracy
    
    # Signal quality
    icd: float  # Intrinsic Concept Dimensionality
    signal_above_noise: float  # accuracy - nonsense_accuracy
    is_linear: bool
    verdict: str  # STRONG_CONCEPT, DIFFUSE_CONCEPT, WEAK_SIGNAL, NO_SIGNAL


@dataclass
class CategoryResult:
    """Result for a single category."""
    category: str
    description: str
    benchmarks_tested: List[str]
    total_tests: int
    
    # Step 1: Signal detection
    avg_signal_strength: float  # MLP CV accuracy
    signal_exists: bool  # avg_signal_strength > 0.6
    
    # Step 2: Linearity check  
    avg_linear_probe_accuracy: float  # Linear probe CV accuracy
    is_linear: bool  # signal is linear (CAA will work)
    
    # NEW: Nonlinear signal metrics
    avg_knn_accuracy_k10: float  # k-NN CV accuracy
    avg_knn_pca_accuracy: float  # k-NN on PCA features (addresses curse of dimensionality)
    avg_knn_umap_accuracy: float  # k-NN on UMAP features (preserves nonlinear structure)
    avg_knn_pacmap_accuracy: float  # k-NN on PaCMAP features (preserves local+global structure)
    avg_mlp_probe_accuracy: float  # MLP probe (regularized nonlinear)
    avg_best_nonlinear: float  # max(knn, knn_pca, knn_umap, knn_pacmap, mlp) - best nonlinear signal detector
    avg_mmd_rbf: float  # Maximum Mean Discrepancy
    avg_local_dim_pos: float  # Local intrinsic dim of positive class
    avg_local_dim_neg: float  # Local intrinsic dim of negative class
    avg_fisher_max: float  # Max Fisher ratio
    avg_density_ratio: float  # Density ratio
    
    # Step 3: Geometry details (only meaningful if signal_exists)
    structure_distribution: Dict[str, int]
    structure_percentages: Dict[str, float]
    dominant_structure: str
    avg_linear_score: float
    avg_cohens_d: float
    
    # Multi-direction analysis: how many directions needed?
    avg_multi_dir_accuracy_k1: float  # accuracy with 1 direction
    avg_multi_dir_accuracy_k3: float  # accuracy with 3 directions
    avg_multi_dir_accuracy_k5: float  # accuracy with 5 directions
    avg_multi_dir_min_k: float  # average min k needed for good accuracy
    avg_multi_dir_gain: float  # average gain from using multiple directions
    
    # NEW: ICD (Intrinsic Concept Dimensionality) and nonsense baseline
    avg_icd: float  # average Intrinsic Concept Dimensionality
    avg_icd_top1_variance: float  # average variance explained by top-1 direction
    avg_nonsense_icd: float  # ICD of nonsense baseline (random tokens)
    avg_icd_ratio: float  # nonsense_icd / real_icd (higher = more concentrated)
    avg_nonsense_accuracy: float  # accuracy on random token baseline
    avg_signal_above_baseline: float  # real accuracy - nonsense accuracy
    signal_verdict: str  # STRONG_CONCEPT, DIFFUSE_CONCEPT, WEAK_SIGNAL, NO_SIGNAL
    
    # Final recommendation
    recommendation: str  # NO_SIGNAL, CAA, or NONLINEAR
    has_unified_direction: bool
    best_config: Optional[Dict[str, Any]] = None
    
    # NEW: Optimal configuration (full diagnosis)
    optimal_config: Optional[OptimalConfig] = None


@dataclass 
class DiscoveryResults:
    """Results from full discovery run."""
    model: str
    categories: Dict[str, CategoryResult] = field(default_factory=dict)
    
    def summary(self) -> str:
        lines = [
            f"Model: {self.model}",
            f"Categories analyzed: {len(self.categories)}",
            "",
        ]
        
        # Group by recommendation
        caa_ready = []  # Has signal AND linear
        nonlinear = []  # Has signal but NOT linear
        no_signal = []  # No signal
        
        for name, cat in self.categories.items():
            if not cat.signal_exists:
                no_signal.append(name)
            elif cat.is_linear:
                caa_ready.append(name)
            else:
                nonlinear.append(name)
        
        if caa_ready:
            lines.append(f"CAA READY - Linear signal ({len(caa_ready)}):")
            for name in sorted(caa_ready, key=lambda n: self.categories[n].avg_best_nonlinear, reverse=True):
                cat = self.categories[name]
                lines.append(f"  {name}: best_nonlinear={cat.avg_best_nonlinear:.2f}, linear={cat.avg_linear_probe_accuracy:.2f}, gap={cat.avg_linear_probe_accuracy - cat.avg_best_nonlinear:.2f}")
        
        if nonlinear:
            lines.append(f"\nNONLINEAR - Need different method ({len(nonlinear)}):")
            for name in nonlinear:
                cat = self.categories[name]
                lines.append(f"  {name}: best_nonlinear={cat.avg_best_nonlinear:.2f}, linear={cat.avg_linear_probe_accuracy:.2f}, gap={cat.avg_linear_probe_accuracy - cat.avg_best_nonlinear:.2f}")
        
        if no_signal:
            lines.append(f"\nNO SIGNAL ({len(no_signal)}):")
            for name in no_signal:
                cat = self.categories[name]
                lines.append(f"  {name}: best_nonlinear={cat.avg_best_nonlinear:.2f} (kNN={cat.avg_knn_accuracy_k10:.2f}, kNN_pca={cat.avg_knn_pca_accuracy:.2f}, kNN_umap={cat.avg_knn_umap_accuracy:.2f}, kNN_pacmap={cat.avg_knn_pacmap_accuracy:.2f}, MLP={cat.avg_mlp_probe_accuracy:.2f})")
        
        return "\n".join(lines)


def run_pairs_ablation(
    runner: "GeometryRunner",
    benchmark: str,
    layer: int,
    strategy: "ExtractionStrategy",
    pair_counts: List[int] = None,
) -> Dict[int, float]:
    """
    Run ablation study on number of contrastive pairs.
    
    Tests different numbers of pairs to find the minimum needed for stable signal.
    
    Args:
        runner: GeometryRunner with loaded model
        benchmark: Benchmark name
        layer: Layer to test
        strategy: Extraction strategy
        pair_counts: List of pair counts to test (default: [10, 25, 50, 100, 200])
        
    Returns:
        Dict mapping n_pairs -> accuracy
    """
    from wisent.core.geometry_runner import compute_geometry_metrics
    from wisent.core.activations.activation_cache import CachedActivations
    
    pair_counts = pair_counts or [10, 25, 50, 100, 200]
    results = {}
    
    # Get full cached activations
    try:
        cached = runner._get_cached_activations(benchmark, strategy, show_progress=False)
    except Exception:
        return {}
    
    layer_name = str(layer)
    if layer_name not in cached.get_available_layers():
        return {}
    
    max_pairs = cached.num_pairs
    
    for n in pair_counts:
        if n > max_pairs:
            continue
        
        # Create subsampled CachedActivations
        sub_cached = CachedActivations(
            benchmark=cached.benchmark,
            strategy=cached.strategy,
            model_name=cached.model_name,
            num_layers=cached.num_layers,
            hidden_size=cached.hidden_size,
        )
        sub_cached.pair_activations = cached.pair_activations[:n]
        sub_cached.num_pairs = n
        
        # Compute metrics
        try:
            result = compute_geometry_metrics(sub_cached, (layer,))
            results[n] = result.linear_probe_accuracy
        except Exception:
            pass
    
    return results


def find_optimal_config(
    results: "GeometrySearchResults",
    nonsense_analysis: Optional[Dict[str, Dict]] = None,
    pairs_ablation: Optional[Dict[str, Dict[int, float]]] = None,
) -> Optional[OptimalConfig]:
    """
    Find optimal configuration from geometry search results.
    
    Args:
        results: GeometrySearchResults from runner
        nonsense_analysis: Per-benchmark nonsense baseline results
        pairs_ablation: Per-benchmark pairs ablation results
        
    Returns:
        OptimalConfig with optimal settings, or None if no signal
    """
    if not results.results:
        return None
    
    # Find best result by linear_probe_accuracy
    best_result = max(results.results, key=lambda r: r.linear_probe_accuracy)
    
    # Find optimal layer (single layer with best accuracy)
    layer_accuracies: Dict[int, List[float]] = {}
    for r in results.results:
        if len(r.layers) == 1:
            layer = r.layers[0]
            if layer not in layer_accuracies:
                layer_accuracies[layer] = []
            layer_accuracies[layer].append(r.linear_probe_accuracy)
    
    if layer_accuracies:
        layer_avg = {l: sum(accs)/len(accs) for l, accs in layer_accuracies.items()}
        optimal_layer = max(layer_avg, key=layer_avg.get)
        layer_accuracy = layer_avg[optimal_layer]
        
        # Find layers within 5% of best
        threshold = layer_accuracy * 0.95
        optimal_layer_range = sorted([l for l, acc in layer_avg.items() if acc >= threshold])[:3]
    else:
        optimal_layer = best_result.layers[0] if best_result.layers else 0
        layer_accuracy = best_result.linear_probe_accuracy
        optimal_layer_range = [optimal_layer]
    
    # Find optimal strategy
    strategy_accuracies: Dict[str, List[float]] = {}
    for r in results.results:
        if r.strategy not in strategy_accuracies:
            strategy_accuracies[r.strategy] = []
        strategy_accuracies[r.strategy].append(r.linear_probe_accuracy)
    
    if strategy_accuracies:
        strategy_avg = {s: sum(accs)/len(accs) for s, accs in strategy_accuracies.items()}
        optimal_strategy = max(strategy_avg, key=strategy_avg.get)
        strategy_accuracy = strategy_avg[optimal_strategy]
    else:
        optimal_strategy = best_result.strategy
        strategy_accuracy = best_result.linear_probe_accuracy
    
    # Aggregate pairs ablation
    pairs_saturation_curve: Dict[int, float] = {}
    if pairs_ablation:
        # Average across benchmarks
        all_n_pairs = set()
        for ablation in pairs_ablation.values():
            all_n_pairs.update(ablation.keys())
        
        for n in sorted(all_n_pairs):
            accs = [ablation[n] for ablation in pairs_ablation.values() if n in ablation]
            if accs:
                pairs_saturation_curve[n] = sum(accs) / len(accs)
    
    # Find minimum pairs for stable signal (within 5% of max)
    if pairs_saturation_curve:
        max_acc = max(pairs_saturation_curve.values())
        threshold = max_acc * 0.95
        min_pairs = min([n for n, acc in pairs_saturation_curve.items() if acc >= threshold], default=50)
    else:
        min_pairs = 50  # default
    
    # Multi-direction analysis
    num_directions = int(best_result.multi_dir_min_k_for_good) if best_result.multi_dir_min_k_for_good > 0 else 1
    single_dir_acc = best_result.multi_dir_accuracy_k1
    multi_dir_acc = best_result.multi_dir_accuracy_k5
    steering_gain = multi_dir_acc - single_dir_acc
    
    # ICD and nonsense baseline
    if nonsense_analysis:
        valid_analyses = [a for a in nonsense_analysis.values() if a]
        if valid_analyses:
            avg_icd = sum(a["real_icd"]["icd"] for a in valid_analyses) / len(valid_analyses)
            avg_signal_above = sum(a["baseline_comparison"]["signal_above_baseline"] for a in valid_analyses) / len(valid_analyses)
            # Majority verdict
            from collections import Counter
            verdicts = [a["verdict"] for a in valid_analyses]
            verdict = Counter(verdicts).most_common(1)[0][0]
        else:
            avg_icd = 0.0
            avg_signal_above = 0.0
            verdict = "NO_DATA"
    else:
        avg_icd = 0.0
        avg_signal_above = 0.0
        verdict = "NOT_COMPUTED"
    
    # Determine if linear
    avg_linear = sum(r.linear_probe_accuracy for r in results.results) / len(results.results)
    avg_best_nonlinear = sum(
        max(r.knn_accuracy_k10, r.knn_pca_accuracy, r.knn_umap_accuracy, r.mlp_probe_accuracy) 
        for r in results.results
    ) / len(results.results)
    is_linear = avg_linear >= avg_best_nonlinear - 0.15
    
    return OptimalConfig(
        optimal_layer=optimal_layer,
        optimal_layer_range=optimal_layer_range,
        layer_accuracy=layer_accuracy,
        optimal_strategy=optimal_strategy,
        strategy_accuracy=strategy_accuracy,
        min_pairs_for_stable_signal=min_pairs,
        pairs_saturation_curve=pairs_saturation_curve,
        num_directions_needed=num_directions,
        single_direction_accuracy=single_dir_acc,
        multi_direction_accuracy=multi_dir_acc,
        steering_gain=steering_gain,
        icd=avg_icd,
        signal_above_noise=avg_signal_above,
        is_linear=is_linear,
        verdict=verdict,
    )


def analyze_category_results(
    results: GeometrySearchResults, 
    category: str, 
    description: str, 
    benchmarks: List[str],
    nonsense_analysis: Optional[Dict[str, Dict]] = None,
    pairs_ablation: Optional[Dict[str, Dict[int, float]]] = None,
) -> CategoryResult:
    """Analyze geometry results for a category."""
    if not results.results:
        return CategoryResult(
            category=category,
            description=description,
            benchmarks_tested=benchmarks,
            total_tests=0,
            avg_signal_strength=0.5,
            signal_exists=False,
            avg_linear_probe_accuracy=0.5,
            is_linear=False,
            avg_knn_accuracy_k10=0.5,
            avg_knn_pca_accuracy=0.5,
            avg_knn_umap_accuracy=0.5,
            avg_knn_pacmap_accuracy=0.5,
            avg_mlp_probe_accuracy=0.5,
            avg_best_nonlinear=0.5,
            avg_mmd_rbf=0.0,
            avg_local_dim_pos=0.0,
            avg_local_dim_neg=0.0,
            avg_fisher_max=0.0,
            avg_density_ratio=1.0,
            structure_distribution={},
            structure_percentages={},
            dominant_structure="error",
            avg_linear_score=0.0,
            avg_cohens_d=0.0,
            avg_multi_dir_accuracy_k1=0.5,
            avg_multi_dir_accuracy_k3=0.5,
            avg_multi_dir_accuracy_k5=0.5,
            avg_multi_dir_min_k=-1.0,
            avg_multi_dir_gain=0.0,
            avg_icd=0.0,
            avg_icd_top1_variance=0.0,
            avg_nonsense_icd=0.0,
            avg_icd_ratio=1.0,
            avg_nonsense_accuracy=0.5,
            avg_signal_above_baseline=0.0,
            signal_verdict="NO_RESULTS",
            recommendation="NO_RESULTS",
            has_unified_direction=False,
        )
    
    dist = results.get_structure_distribution()
    total = sum(dist.values())
    
    percentages = {k: 100 * v / total for k, v in dist.items()} if total > 0 else {}
    
    # Determine dominant structure
    dominant = max(dist.items(), key=lambda x: x[1])[0] if dist else "unknown"
    
    # Step 1: Compute ALL probe accuracies
    avg_signal_strength = sum(r.signal_strength for r in results.results) / len(results.results)
    avg_linear_probe_accuracy = sum(r.linear_probe_accuracy for r in results.results) / len(results.results)
    avg_knn_accuracy_k10 = sum(r.knn_accuracy_k10 for r in results.results) / len(results.results)
    avg_knn_pca_accuracy = sum(r.knn_pca_accuracy for r in results.results) / len(results.results)
    avg_knn_umap_accuracy = sum(r.knn_umap_accuracy for r in results.results) / len(results.results)
    avg_knn_pacmap_accuracy = sum(r.knn_pacmap_accuracy for r in results.results) / len(results.results)
    avg_mlp_probe_accuracy = sum(r.mlp_probe_accuracy for r in results.results) / len(results.results)
    avg_mmd_rbf = sum(r.mmd_rbf for r in results.results) / len(results.results)
    avg_local_dim_pos = sum(r.local_dim_pos for r in results.results) / len(results.results)
    avg_local_dim_neg = sum(r.local_dim_neg for r in results.results) / len(results.results)
    avg_fisher_max = sum(r.fisher_max for r in results.results) / len(results.results)
    avg_density_ratio = sum(r.density_ratio for r in results.results) / len(results.results)
    
    # Compute best nonlinear for each result, then average
    # best_nonlinear = max(knn_k10, knn_pca, knn_umap, mlp) for each result
    avg_best_nonlinear = sum(
        max(r.knn_accuracy_k10, r.knn_pca_accuracy, r.knn_umap_accuracy, r.mlp_probe_accuracy) 
        for r in results.results
    ) / len(results.results)
    
    # Step 2: Signal detection and classification (matching paper methodology)
    # Use MAXIMUM of nonlinear probes as signal detector (addresses curse of dimensionality)
    # Thresholds from paper: tau_exist = 0.6, tau_gap = 0.15
    tau_exist = 0.6
    tau_gap = 0.15
    
    # Signal exists if ANY nonlinear method can separate classes above chance
    signal_exists = avg_best_nonlinear >= tau_exist
    
    # Step 3: Determine if signal is linear or nonlinear
    # NO_SIGNAL: best_nonlinear < 0.6 (no separable signal by any method)
    # LINEAR: best_nonlinear >= 0.6 AND linear >= best_nonlinear - 0.15 (linear methods work)
    # NONLINEAR: best_nonlinear >= 0.6 AND linear < best_nonlinear - 0.15 (linear methods fail but nonlinear works)
    if not signal_exists:
        is_linear = False
        recommendation = "NO_SIGNAL"
    elif avg_linear_probe_accuracy >= avg_best_nonlinear - tau_gap:
        is_linear = True
        recommendation = "CAA"
    else:
        is_linear = False
        recommendation = "NONLINEAR"
    
    # Step 4: Geometry details
    avg_linear_score = sum(r.linear_score for r in results.results) / len(results.results)
    avg_cohens_d = sum(r.cohens_d for r in results.results) / len(results.results)
    
    # Step 5: Multi-direction analysis
    avg_multi_dir_accuracy_k1 = sum(r.multi_dir_accuracy_k1 for r in results.results) / len(results.results)
    avg_multi_dir_accuracy_k3 = sum(r.multi_dir_accuracy_k3 for r in results.results) / len(results.results)
    avg_multi_dir_accuracy_k5 = sum(r.multi_dir_accuracy_k5 for r in results.results) / len(results.results)
    # For min_k, only average valid values (> 0)
    valid_min_k = [r.multi_dir_min_k_for_good for r in results.results if r.multi_dir_min_k_for_good > 0]
    avg_multi_dir_min_k = sum(valid_min_k) / len(valid_min_k) if valid_min_k else -1.0
    avg_multi_dir_gain = sum(r.multi_dir_gain for r in results.results) / len(results.results)
    
    # Unified direction exists if we have linear signal
    has_unified = is_linear
    
    # Step 6: ICD and nonsense baseline analysis
    if nonsense_analysis:
        # Aggregate nonsense analysis results across benchmarks
        icd_values = []
        icd_top1_values = []
        nonsense_icd_values = []
        icd_ratio_values = []
        nonsense_acc_values = []
        signal_above_values = []
        verdicts = []
        
        for benchmark, analysis in nonsense_analysis.items():
            if analysis:
                icd_values.append(analysis["real_icd"]["icd"])
                icd_top1_values.append(analysis["real_icd"]["top1_variance"])
                nonsense_icd_values.append(analysis["nonsense_icd"]["icd"])
                icd_ratio_values.append(analysis["icd_ratio"])
                nonsense_acc_values.append(analysis["baseline_comparison"]["nonsense_accuracy"])
                signal_above_values.append(analysis["baseline_comparison"]["signal_above_baseline"])
                verdicts.append(analysis["verdict"])
        
        if icd_values:
            avg_icd = sum(icd_values) / len(icd_values)
            avg_icd_top1_variance = sum(icd_top1_values) / len(icd_top1_values)
            avg_nonsense_icd = sum(nonsense_icd_values) / len(nonsense_icd_values)
            avg_icd_ratio = sum(icd_ratio_values) / len(icd_ratio_values)
            avg_nonsense_accuracy = sum(nonsense_acc_values) / len(nonsense_acc_values)
            avg_signal_above_baseline = sum(signal_above_values) / len(signal_above_values)
            # Determine overall verdict by majority
            from collections import Counter
            verdict_counts = Counter(verdicts)
            signal_verdict = verdict_counts.most_common(1)[0][0]
        else:
            avg_icd = 0.0
            avg_icd_top1_variance = 0.0
            avg_nonsense_icd = 0.0
            avg_icd_ratio = 1.0
            avg_nonsense_accuracy = 0.5
            avg_signal_above_baseline = 0.0
            signal_verdict = "NO_DATA"
    else:
        # No nonsense analysis provided
        avg_icd = 0.0
        avg_icd_top1_variance = 0.0
        avg_nonsense_icd = 0.0
        avg_icd_ratio = 1.0
        avg_nonsense_accuracy = 0.5
        avg_signal_above_baseline = 0.0
        signal_verdict = "NOT_COMPUTED"
    
    # Best config - prefer high signal_strength
    best = sorted(results.results, key=lambda r: r.signal_strength, reverse=True)[:1]
    best_config = None
    if best:
        b = best[0]
        best_config = {
            "benchmark": b.benchmark,
            "strategy": b.strategy,
            "layers": b.layers,
            "signal_strength": b.signal_strength,
            "linear_probe_accuracy": b.linear_probe_accuracy,
            "is_linear": b.is_linear,
        }
    
    return CategoryResult(
        category=category,
        description=description,
        benchmarks_tested=benchmarks,
        total_tests=total,
        avg_signal_strength=avg_signal_strength,
        signal_exists=signal_exists,
        avg_linear_probe_accuracy=avg_linear_probe_accuracy,
        is_linear=is_linear,
        avg_knn_accuracy_k10=avg_knn_accuracy_k10,
        avg_knn_pca_accuracy=avg_knn_pca_accuracy,
        avg_knn_umap_accuracy=avg_knn_umap_accuracy,
        avg_knn_pacmap_accuracy=avg_knn_pacmap_accuracy,
        avg_mlp_probe_accuracy=avg_mlp_probe_accuracy,
        avg_best_nonlinear=avg_best_nonlinear,
        avg_mmd_rbf=avg_mmd_rbf,
        avg_local_dim_pos=avg_local_dim_pos,
        avg_local_dim_neg=avg_local_dim_neg,
        avg_fisher_max=avg_fisher_max,
        avg_density_ratio=avg_density_ratio,
        structure_distribution=dist,
        structure_percentages=percentages,
        dominant_structure=dominant,
        avg_linear_score=avg_linear_score,
        avg_cohens_d=avg_cohens_d,
        avg_multi_dir_accuracy_k1=avg_multi_dir_accuracy_k1,
        avg_multi_dir_accuracy_k3=avg_multi_dir_accuracy_k3,
        avg_multi_dir_accuracy_k5=avg_multi_dir_accuracy_k5,
        avg_multi_dir_min_k=avg_multi_dir_min_k,
        avg_multi_dir_gain=avg_multi_dir_gain,
        avg_icd=avg_icd,
        avg_icd_top1_variance=avg_icd_top1_variance,
        avg_nonsense_icd=avg_nonsense_icd,
        avg_icd_ratio=avg_icd_ratio,
        avg_nonsense_accuracy=avg_nonsense_accuracy,
        avg_signal_above_baseline=avg_signal_above_baseline,
        signal_verdict=signal_verdict,
        recommendation=recommendation,
        has_unified_direction=has_unified,
        best_config=best_config,
        optimal_config=find_optimal_config(results, nonsense_analysis, pairs_ablation),
    )


def run_discovery_for_model(model_name: str, output_dir: Path, with_nonsense_baseline: bool = False, with_pairs_ablation: bool = False):
    """Run discovery for a single model with resume support."""
    categories = load_categorized_benchmarks()
    category_info = load_category_directions()
    search_space = GeometrySearchSpace()
    
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name}")
    print("=" * 70)
    
    # Download existing results from S3 for resume
    s3_sync_download(model_name, output_dir)
    
    # Check which categories need work
    model_prefix = model_name.replace('/', '_')
    completed_categories = set()
    needs_diagnosis = set()  # Has results but missing optimal_config
    
    for cat_name in categories.keys():
        cat_file = output_dir / f"{model_prefix}_{cat_name}.json"
        if cat_file.exists() and cat_file.stat().st_size > 100:
            # Check if it has optimal_config (full diagnosis)
            if with_nonsense_baseline or with_pairs_ablation:
                try:
                    with open(cat_file) as f:
                        existing = json.load(f)
                    # Check if optimal_config exists and has real data
                    has_diagnosis = existing.get("optimal_config") is not None
                    if not has_diagnosis:
                        needs_diagnosis.add(cat_name)
                        print(f"  [UPGRADE] {cat_name} needs full diagnosis")
                        continue
                except Exception:
                    pass
            completed_categories.add(cat_name)
            print(f"  [SKIP] {cat_name} already completed")
    
    remaining = [c for c in categories.keys() if c not in completed_categories]
    if not remaining:
        print("All categories already completed!")
        return None
    
    print(f"\nCompleted: {len(completed_categories)}/{len(categories)}, Remaining: {len(remaining)}")
    print(f"Categories to run: {remaining}")
    
    try:
        model = WisentModel(model_name, device="cuda")
        print(f"Loaded: {model.num_layers} layers, hidden={model.hidden_size}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    
    cache_dir = f"/tmp/wisent_direction_cache_{model_prefix}"
    
    model_results = DiscoveryResults(model=model_name)
    
    # Run for each remaining category
    for cat_name in remaining:
        benchmarks = categories[cat_name]
        print(f"\n{'-' * 50}")
        print(f"Category: {cat_name.upper()} ({len(benchmarks)} benchmarks)")
        print("-" * 50)
        
        info = category_info.get(cat_name, {})
        description = info.get("description", "")
        print(f"Description: {description}")
        
        # Create search space for this category
        cat_config = GeometrySearchConfig(
            pairs_per_benchmark=search_space.config.pairs_per_benchmark,
            max_layer_combo_size=search_space.config.max_layer_combo_size,
            cache_dir=cache_dir,
        )
        
        cat_space = GeometrySearchSpace(
            models=[model_name],
            strategies=search_space.strategies,
            benchmarks=benchmarks,
            config=cat_config,
        )
        
        # Run geometry search
        runner = GeometryRunner(cat_space, model, cache_dir=cache_dir)
        
        try:
            # Check if we can load existing results (upgrade mode)
            cat_file = output_dir / f"{model_prefix}_{cat_name}.json"
            if cat_name in needs_diagnosis and cat_file.exists():
                print(f"  [UPGRADE MODE] Loading existing results, adding diagnosis...")
                results = GeometrySearchResults.load(str(cat_file))
            else:
                results = runner.run(show_progress=True)
            
            # Optionally compute nonsense baseline for each benchmark
            nonsense_analysis = {}
            if with_nonsense_baseline:
                print(f"  Computing nonsense baseline...")
                for benchmark in benchmarks:
                    # Get cached activations for this benchmark
                    try:
                        cached = runner._get_cached_activations(
                            benchmark, 
                            ExtractionStrategy.CHAT_LAST,
                            show_progress=False
                        )
                        # Use middle layer for analysis
                        mid_layer = model.num_layers // 2
                        layer_name = str(mid_layer)
                        if layer_name in cached.get_available_layers():
                            pos_acts = cached.get_positive_activations(mid_layer)
                            neg_acts = cached.get_negative_activations(mid_layer)
                            n_pairs = min(len(pos_acts), len(neg_acts))
                            
                            # Generate nonsense baseline with SAME number of pairs
                            nonsense_pos, nonsense_neg = runner.get_nonsense_baseline(
                                n_pairs=n_pairs,
                                layer=mid_layer,
                            )
                            
                            # Analyze
                            analysis = analyze_with_nonsense_baseline(
                                pos_acts[:n_pairs], neg_acts[:n_pairs],
                                nonsense_pos, nonsense_neg,
                                benchmark
                            )
                            nonsense_analysis[benchmark] = analysis
                            print(f"    {benchmark}: {analysis['verdict']} (ICD={analysis['real_icd']['icd']:.1f}, acc={analysis['baseline_comparison']['real_accuracy']:.0%})")
                    except Exception as e:
                        print(f"    {benchmark}: SKIP ({e})")
                        nonsense_analysis[benchmark] = None
            
            # Optionally run pairs ablation
            pairs_ablation = {}
            if with_pairs_ablation:
                print(f"  Running pairs ablation...")
                mid_layer = model.num_layers // 2
                for benchmark in benchmarks:
                    try:
                        ablation = run_pairs_ablation(
                            runner, benchmark, mid_layer, 
                            ExtractionStrategy.CHAT_LAST,
                            pair_counts=[10, 25, 50, 100, 200]
                        )
                        if ablation:
                            pairs_ablation[benchmark] = ablation
                            curve = ", ".join(f"{n}:{acc:.0%}" for n, acc in sorted(ablation.items()))
                            print(f"    {benchmark}: {curve}")
                    except Exception as e:
                        print(f"    {benchmark}: SKIP ({e})")
            
            cat_result = analyze_category_results(
                results, cat_name, description, benchmarks,
                nonsense_analysis=nonsense_analysis if with_nonsense_baseline else None,
                pairs_ablation=pairs_ablation if with_pairs_ablation else None,
            )
            model_results.categories[cat_name] = cat_result
            
            print(f"\n  Step 1 - Signal: {cat_result.avg_signal_strength:.3f} ({'EXISTS' if cat_result.signal_exists else 'NONE'})")
            print(f"  Step 2 - Linear: {cat_result.avg_linear_probe_accuracy:.3f} ({'YES' if cat_result.is_linear else 'NO'})")
            if with_nonsense_baseline:
                print(f"  Step 3 - Nonsense baseline: ICD={cat_result.avg_icd:.1f}, verdict={cat_result.signal_verdict}")
            if cat_result.optimal_config:
                oc = cat_result.optimal_config
                print(f"  Optimal config: layer={oc.optimal_layer}, strategy={oc.optimal_strategy}, min_pairs={oc.min_pairs_for_stable_signal}, k_directions={oc.num_directions_needed}")
            print(f"  Recommendation: {cat_result.recommendation}")
            
            # Save per-category results immediately (cat_file defined earlier)
            results.save(str(cat_file))
            print(f"  Saved: {cat_file}")
            
            # Upload to S3 immediately for durability
            s3_upload_file(cat_file, model_name)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Save/update model summary (merge with existing if any)
    summary_file = output_dir / f"{model_prefix}_summary.json"
    
    # Load existing summary if present
    existing_categories = {}
    if summary_file.exists():
        with open(summary_file) as f:
            existing = json.load(f)
            existing_categories = existing.get("categories", {})
    
    # Merge new results
    all_categories = {**existing_categories, **{k: asdict(v) for k, v in model_results.categories.items()}}
    
    with open(summary_file, "w") as f:
        json.dump({
            "model": model_name,
            "categories": all_categories
        }, f, indent=2)
    
    # Upload summary to S3
    s3_upload_file(summary_file, model_name)
    
    print(f"\n{model_results.summary()}")
    
    # Cleanup model
    del model
    
    return model_results


def generate_cross_model_comparison(all_model_results: Dict[str, "DiscoveryResults"]) -> Dict[str, Any]:
    """
    Generate cross-model comparison for each category.
    
    Identifies which model best represents each concept.
    """
    if not all_model_results:
        return {}
    
    comparison = {}
    
    # Get all categories across models
    all_categories = set()
    for results in all_model_results.values():
        all_categories.update(results.categories.keys())
    
    for category in all_categories:
        cat_comparison = {
            "best_model": None,
            "best_accuracy": 0.0,
            "models": {},
            "consensus_verdict": None,
            "consensus_is_linear": None,
        }
        
        verdicts = []
        linearities = []
        
        for model_name, results in all_model_results.items():
            if category not in results.categories:
                continue
                
            cat_result = results.categories[category]
            
            # Store per-model results
            cat_comparison["models"][model_name] = {
                "signal_exists": cat_result.signal_exists,
                "is_linear": cat_result.is_linear,
                "avg_linear_accuracy": cat_result.avg_linear_probe_accuracy,
                "avg_best_nonlinear": cat_result.avg_best_nonlinear,
                "recommendation": cat_result.recommendation,
                "optimal_layer": cat_result.optimal_config.optimal_layer if cat_result.optimal_config else None,
                "optimal_strategy": cat_result.optimal_config.optimal_strategy if cat_result.optimal_config else None,
                "min_pairs": cat_result.optimal_config.min_pairs_for_stable_signal if cat_result.optimal_config else None,
                "k_directions": cat_result.optimal_config.num_directions_needed if cat_result.optimal_config else None,
                "verdict": cat_result.optimal_config.verdict if cat_result.optimal_config else cat_result.signal_verdict,
            }
            
            # Track best model (by signal strength)
            accuracy = cat_result.avg_best_nonlinear
            if accuracy > cat_comparison["best_accuracy"]:
                cat_comparison["best_accuracy"] = accuracy
                cat_comparison["best_model"] = model_name
            
            # Collect for consensus
            if cat_result.optimal_config:
                verdicts.append(cat_result.optimal_config.verdict)
                linearities.append(cat_result.optimal_config.is_linear)
            elif cat_result.signal_verdict:
                verdicts.append(cat_result.signal_verdict)
                linearities.append(cat_result.is_linear)
        
        # Determine consensus
        if verdicts:
            from collections import Counter
            cat_comparison["consensus_verdict"] = Counter(verdicts).most_common(1)[0][0]
        if linearities:
            cat_comparison["consensus_is_linear"] = sum(linearities) / len(linearities) >= 0.5
        
        comparison[category] = cat_comparison
    
    return comparison


def run_discovery(model_filter: Optional[str] = None, samples_per_benchmark: int = 50, with_nonsense_baseline: bool = False, with_pairs_ablation: bool = False):
    """Run full category direction discovery."""
    print("=" * 70)
    print("CATEGORY DIRECTION DISCOVERY")
    print("=" * 70)
    
    # Load categories
    categories = load_categorized_benchmarks()
    category_info = load_category_directions()
    
    print(f"Categories: {list(categories.keys())}")
    print(f"Total benchmarks: {sum(len(b) for b in categories.values())}")
    
    # Get search space config
    search_space = GeometrySearchSpace()
    search_space.config.pairs_per_benchmark = samples_per_benchmark
    
    # Filter models if specified
    if model_filter:
        models_to_test = [model_filter]
    else:
        models_to_test = search_space.models
    
    print(f"\nModels to test: {models_to_test}")
    print(f"Strategies: {[s.value for s in search_space.strategies]}")
    print(f"Pairs per benchmark: {search_space.config.pairs_per_benchmark}")
    
    # Output directory
    output_dir = Path("/tmp/direction_discovery")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_model_results = {}
    
    # Run for each model
    for model_name in models_to_test:
        model_results = run_discovery_for_model(model_name, output_dir, with_nonsense_baseline, with_pairs_ablation)
        if model_results:
            all_model_results[model_name] = model_results
    
    # Save overall summary (only if running all models)
    if not model_filter and all_model_results:
        overall_file = output_dir / "discovery_summary.json"
        overall = {
            "models": list(all_model_results.keys()),
            "categories": list(categories.keys()),
            "results": {}
        }
        for model_name, results in all_model_results.items():
            overall["results"][model_name] = {
                cat: {
                    "has_unified_direction": r.has_unified_direction,
                    "dominant_structure": r.dominant_structure,
                    "recommendation": r.recommendation,
                    "avg_linear_score": r.avg_linear_score,
                    # NEW: Add optimal config summary
                    "optimal_config": {
                        "layer": r.optimal_config.optimal_layer if r.optimal_config else None,
                        "strategy": r.optimal_config.optimal_strategy if r.optimal_config else None,
                        "min_pairs": r.optimal_config.min_pairs_for_stable_signal if r.optimal_config else None,
                        "k_directions": r.optimal_config.num_directions_needed if r.optimal_config else None,
                        "is_linear": r.optimal_config.is_linear if r.optimal_config else None,
                        "verdict": r.optimal_config.verdict if r.optimal_config else None,
                    } if r.optimal_config else None,
                }
                for cat, r in results.categories.items()
            }
        
        # Add cross-model comparison
        overall["cross_model_comparison"] = generate_cross_model_comparison(all_model_results)
        
        with open(overall_file, "w") as f:
            json.dump(overall, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print("DISCOVERY COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover unified directions for skill categories")
    parser.add_argument("--model", type=str, default=None, help="Specific model to test (for parallel execution)")
    parser.add_argument("--samples-per-benchmark", type=int, default=0, help="Number of samples per benchmark (default: 0 = use all available)")
    parser.add_argument("--with-nonsense-baseline", action="store_true", 
                        help="Compare against random token baseline (requires generating activations through the model)")
    parser.add_argument("--with-pairs-ablation", action="store_true",
                        help="Run ablation on number of pairs to find minimum needed for stable signal")
    parser.add_argument("--full-diagnosis", action="store_true",
                        help="Run full diagnosis (enables both --with-nonsense-baseline and --with-pairs-ablation)")
    args = parser.parse_args()
    
    # --full-diagnosis enables both
    with_nonsense = args.with_nonsense_baseline or args.full_diagnosis
    with_pairs = args.with_pairs_ablation or args.full_diagnosis
    
    run_discovery(
        model_filter=args.model, 
        samples_per_benchmark=args.samples_per_benchmark,
        with_nonsense_baseline=with_nonsense,
        with_pairs_ablation=with_pairs,
    )
