"""Threshold Analysis for Zwiad."""

import argparse
from pathlib import Path
from dataclasses import asdict

import numpy as np

from wisent.core.constants import (
    NULL_DISTRIBUTION_SAMPLES, THRESHOLD_HIDDEN_DIM_LARGE,
    JSON_ARRAY_LIMIT, GAP_THRESHOLD_CANDIDATES,
    SEPARATOR_LINE_WIDTH, SEPARATOR_WIDTH_STANDARD, JSON_INDENT,
)
from wisent.examples.scripts.threshold_analysis_helpers import (
    gcs_upload_file,
    ThresholdAnalysisResult,
    generate_null_distribution,
    generate_synthetic_data,
    compute_roc_for_existence,
    run_sensitivity_analysis,
    load_diagnosis_results,
)


def run_threshold_analysis(model_name: str):
    """
    Run full threshold analysis.
    
    Args:
        model_name: Model to analyze
    """
    print("=" * SEPARATOR_LINE_WIDTH)
    print("THRESHOLD ANALYSIS")
    print("=" * SEPARATOR_LINE_WIDTH)
    print(f"Model: {model_name}")
    
    output_dir = Path("/tmp/threshold_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load diagnosis results
    results = load_diagnosis_results(model_name, output_dir)
    if not results:
        print("ERROR: No diagnosis results found.")
        return
    
    print(f"Loaded {len(results)} results")
    
    # 1. Generate null distribution
    print("\n1. Generating null distribution...")
    null_knn, null_linear = generate_null_distribution(None, n_samples=NULL_DISTRIBUTION_SAMPLES, hidden_dim=THRESHOLD_HIDDEN_DIM_LARGE)
    
    print(f"   Null kNN: mean={np.mean(null_knn):.3f}, std={np.std(null_knn):.3f}")
    print(f"   Null linear: mean={np.mean(null_linear):.3f}, std={np.std(null_linear):.3f}")
    
    # 2. ROC for existence threshold
    print("\n2. Computing ROC for existence threshold...")
    thresholds, tpr, fpr, roc_auc = compute_roc_for_existence(results, null_knn)
    
    # Find optimal threshold (Youden's J)
    j_scores = [t - f for t, f in zip(tpr, fpr)]
    optimal_idx = np.argmax(j_scores)
    optimal_exist = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.6
    
    print(f"   AUC: {roc_auc:.3f}")
    print(f"   Optimal existence threshold: {optimal_exist:.3f}")
    
    # 3. Synthetic validation
    print("\n3. Synthetic validation...")
    from wisent.core.geometry_runner import compute_knn_accuracy, compute_linear_probe_accuracy
    
    synthetic_results = {}
    for structure in ["linear", "xor", "spirals", "random"]:
        pos, neg = generate_synthetic_data(structure)
        knn = compute_knn_accuracy(pos, neg, k=10)
        linear = compute_linear_probe_accuracy(pos, neg)
        gap = knn - linear
        
        synthetic_results[structure] = {
            "knn": knn,
            "linear": linear,
            "gap": gap,
        }
        print(f"   {structure}: kNN={knn:.3f}, linear={linear:.3f}, gap={gap:.3f}")
    
    # Validate that gap threshold separates linear from nonlinear
    linear_gap = synthetic_results["linear"]["gap"]
    xor_gap = synthetic_results["xor"]["gap"]
    spirals_gap = synthetic_results["spirals"]["gap"]
    
    # Good gap threshold should be > linear_gap and < min(xor_gap, spirals_gap)
    optimal_gap = (linear_gap + min(xor_gap, spirals_gap)) / 2
    print(f"\n   Suggested gap threshold: {optimal_gap:.3f}")
    
    # 4. Sensitivity analysis
    print("\n4. Running sensitivity analysis...")
    sensitivity = run_sensitivity_analysis(results)
    
    print("\n   Diagnosis distribution (% of benchmarks):")
    print("   " + "-" * SEPARATOR_WIDTH_STANDARD)
    print(f"   {'Exist':>6} | {'Gap':>6} | {'LINEAR':>8} | {'NONLINEAR':>10} | {'NO_SIGNAL':>10}")
    print("   " + "-" * SEPARATOR_WIDTH_STANDARD)
    
    for exist_t, gap_data in sensitivity.items():
        for gap_t, diagnoses in gap_data.items():
            print(f"   {exist_t:>6} | {gap_t:>6} | {diagnoses['LINEAR']:>7.1f}% | "
                  f"{diagnoses['NONLINEAR']:>9.1f}% | {diagnoses['NO_SIGNAL']:>9.1f}%")
    
    # 5. Save results
    analysis_result = ThresholdAnalysisResult(
        existence_thresholds=thresholds[:JSON_ARRAY_LIMIT],  # Limit for JSON
        existence_tpr=tpr[:JSON_ARRAY_LIMIT],
        existence_fpr=fpr[:JSON_ARRAY_LIMIT],
        existence_auc=roc_auc,
        optimal_existence_threshold=float(optimal_exist),
        gap_thresholds=GAP_THRESHOLD_CANDIDATES,
        gap_precision=[],  # Would need ground truth
        gap_recall=[],
        gap_f1=[],
        optimal_gap_threshold=float(optimal_gap),
        null_mean_knn=float(np.mean(null_knn)),
        null_std_knn=float(np.std(null_knn)),
        null_mean_linear=float(np.mean(null_linear)),
        null_std_linear=float(np.std(null_linear)),
        sensitivity_matrix=sensitivity,
    )
    
    model_prefix = model_name.replace('/', '_')
    results_file = output_dir / f"{model_prefix}_threshold_analysis.json"
    
    with open(results_file, "w") as f:
        json.dump(asdict(analysis_result), f, indent=JSON_INDENT)
    
    print(f"\nResults saved to: {results_file}")
    gcs_upload_file(results_file, model_name)
    
    # Summary
    print("\n" + "=" * SEPARATOR_LINE_WIDTH)
    print("RECOMMENDATIONS")
    print("=" * SEPARATOR_LINE_WIDTH)
    print(f"\n1. Existence threshold: {optimal_exist:.2f}")
    print(f"   - Based on ROC analysis (AUC={roc_auc:.3f})")
    print(f"   - Null distribution: kNN={np.mean(null_knn):.3f} ± {np.std(null_knn):.3f}")
    
    print(f"\n2. Gap threshold: {optimal_gap:.2f}")
    print(f"   - Based on synthetic validation")
    print(f"   - Linear structure gap: {linear_gap:.3f}")
    print(f"   - XOR structure gap: {xor_gap:.3f}")
    print(f"   - Spirals structure gap: {spirals_gap:.3f}")
    
    return analysis_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold analysis for Zwiad")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model to analyze")
    args = parser.parse_args()
    
    run_threshold_analysis(args.model)
