"""
Category result analysis for direction discovery.
"""

from typing import Dict, List, Optional

from wisent.core.utils.config_tools.constants import (
    BLEND_DEFAULT, DEFAULT_SCORE, DENSITY_RATIO_DEFAULT,
    MULTI_DIR_MIN_K_NOT_FOUND, SIGNAL_EXIST_THRESHOLD, SIGNAL_LINEAR_GAP,
)
from wisent.core.reading.modules.runner.geometry_runner import GeometrySearchResults
from wisent.examples.scripts._discovery_utils import CategoryResult
from wisent.examples.scripts._pairs_ablation import find_optimal_config


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
            avg_signal_strength=BLEND_DEFAULT,
            signal_exists=False,
            avg_linear_probe_accuracy=BLEND_DEFAULT,
            is_linear=False,
            avg_knn_accuracy_k10=BLEND_DEFAULT,
            avg_knn_pca_accuracy=BLEND_DEFAULT,
            avg_knn_umap_accuracy=BLEND_DEFAULT,
            avg_knn_pacmap_accuracy=BLEND_DEFAULT,
            avg_mlp_probe_accuracy=BLEND_DEFAULT,
            avg_best_nonlinear=BLEND_DEFAULT,
            avg_mmd_rbf=DEFAULT_SCORE,
            avg_local_dim_pos=DEFAULT_SCORE,
            avg_local_dim_neg=DEFAULT_SCORE,
            avg_fisher_max=DEFAULT_SCORE,
            avg_density_ratio=DENSITY_RATIO_DEFAULT,
            structure_distribution={},
            structure_percentages={},
            dominant_structure="error",
            avg_linear_score=DEFAULT_SCORE,
            avg_cohens_d=DEFAULT_SCORE,
            avg_multi_dir_accuracy_k1=BLEND_DEFAULT,
            avg_multi_dir_accuracy_k3=BLEND_DEFAULT,
            avg_multi_dir_accuracy_k5=BLEND_DEFAULT,
            avg_multi_dir_min_k=float(MULTI_DIR_MIN_K_NOT_FOUND),
            avg_multi_dir_gain=DEFAULT_SCORE,
            avg_icd=DEFAULT_SCORE,
            avg_icd_top1_variance=DEFAULT_SCORE,
            avg_nonsense_icd=DEFAULT_SCORE,
            avg_icd_ratio=DENSITY_RATIO_DEFAULT,
            avg_nonsense_accuracy=BLEND_DEFAULT,
            avg_signal_above_baseline=DEFAULT_SCORE,
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
    tau_exist = SIGNAL_EXIST_THRESHOLD
    tau_gap = SIGNAL_LINEAR_GAP
    
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
    avg_multi_dir_min_k = sum(valid_min_k) / len(valid_min_k) if valid_min_k else float(MULTI_DIR_MIN_K_NOT_FOUND)
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
            avg_icd = DEFAULT_SCORE
            avg_icd_top1_variance = DEFAULT_SCORE
            avg_nonsense_icd = DEFAULT_SCORE
            avg_icd_ratio = DENSITY_RATIO_DEFAULT
            avg_nonsense_accuracy = BLEND_DEFAULT
            avg_signal_above_baseline = DEFAULT_SCORE
            signal_verdict = "NO_DATA"
    else:
        # No nonsense analysis provided
        avg_icd = DEFAULT_SCORE
        avg_icd_top1_variance = DEFAULT_SCORE
        avg_nonsense_icd = DEFAULT_SCORE
        avg_icd_ratio = DENSITY_RATIO_DEFAULT
        avg_nonsense_accuracy = BLEND_DEFAULT
        avg_signal_above_baseline = DEFAULT_SCORE
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

