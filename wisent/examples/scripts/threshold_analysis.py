"""
Threshold Analysis for RepScan.

Analyzes sensitivity of diagnosis to threshold choices:
- ROC curves for existence threshold
- Precision/recall tradeoff
- Null distribution analysis
- Synthetic validation

Usage:
    python -m wisent.examples.scripts.threshold_analysis --model Qwen/Qwen3-8B
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import random

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

S3_BUCKET = "wisent-bucket"
S3_PREFIX = "threshold_analysis"


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


@dataclass
class ThresholdAnalysisResult:
    """Result of threshold analysis."""
    # Existence threshold analysis
    existence_thresholds: List[float]
    existence_tpr: List[float]  # True positive rate
    existence_fpr: List[float]  # False positive rate
    existence_auc: float
    optimal_existence_threshold: float
    
    # Gap threshold analysis
    gap_thresholds: List[float]
    gap_precision: List[float]
    gap_recall: List[float]
    gap_f1: List[float]
    optimal_gap_threshold: float
    
    # Null distribution stats
    null_mean_knn: float
    null_std_knn: float
    null_mean_linear: float
    null_std_linear: float
    
    # Sensitivity analysis
    sensitivity_matrix: Dict[str, Dict[str, float]]  # threshold -> diagnosis distribution


def generate_null_distribution(
    model: "WisentModel",
    n_samples: int = 100,
    hidden_dim: int = 4096,
) -> Tuple[List[float], List[float]]:
    """
    Generate null distribution by testing random/nonsense data.
    
    Args:
        model: WisentModel instance
        n_samples: Number of random samples
        hidden_dim: Hidden dimension
        
    Returns:
        (knn_scores, linear_scores) for random data
    """
    from wisent.core.geometry_runner import compute_knn_accuracy, compute_linear_probe_accuracy
    
    knn_scores = []
    linear_scores = []
    
    for _ in range(n_samples):
        # Generate random activations (no real signal)
        pos = torch.randn(50, hidden_dim)
        neg = torch.randn(50, hidden_dim)
        
        knn = compute_knn_accuracy(pos, neg, k=10)
        linear = compute_linear_probe_accuracy(pos, neg)
        
        knn_scores.append(knn)
        linear_scores.append(linear)
    
    return knn_scores, linear_scores


def generate_synthetic_data(
    structure: str,
    n_samples: int = 50,
    hidden_dim: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data with known structure for validation.
    
    Args:
        structure: 'linear', 'xor', 'spirals', 'random'
        n_samples: Samples per class
        hidden_dim: Dimension
        
    Returns:
        (pos_activations, neg_activations)
    """
    if structure == "linear":
        # Linear separable: positive class shifted in one direction
        direction = torch.randn(hidden_dim)
        direction = direction / direction.norm()
        
        pos = torch.randn(n_samples, hidden_dim) + 2 * direction
        neg = torch.randn(n_samples, hidden_dim) - 2 * direction
        
    elif structure == "xor":
        # XOR pattern: nonlinear but separable
        base = torch.randn(n_samples, hidden_dim)
        
        # Positive: (high dim1 AND high dim2) OR (low dim1 AND low dim2)
        pos_mask1 = (base[:n_samples//2, 0] > 0) & (base[:n_samples//2, 1] > 0)
        pos_mask2 = (base[n_samples//2:, 0] < 0) & (base[n_samples//2:, 1] < 0)
        
        pos = torch.randn(n_samples, hidden_dim)
        pos[:n_samples//2, 0] = torch.abs(pos[:n_samples//2, 0]) + 1
        pos[:n_samples//2, 1] = torch.abs(pos[:n_samples//2, 1]) + 1
        pos[n_samples//2:, 0] = -torch.abs(pos[n_samples//2:, 0]) - 1
        pos[n_samples//2:, 1] = -torch.abs(pos[n_samples//2:, 1]) - 1
        
        neg = torch.randn(n_samples, hidden_dim)
        neg[:n_samples//2, 0] = torch.abs(neg[:n_samples//2, 0]) + 1
        neg[:n_samples//2, 1] = -torch.abs(neg[:n_samples//2, 1]) - 1
        neg[n_samples//2:, 0] = -torch.abs(neg[n_samples//2:, 0]) - 1
        neg[n_samples//2:, 1] = torch.abs(neg[n_samples//2:, 1]) + 1
        
    elif structure == "spirals":
        # Interleaved spirals: nonlinear separable
        t_pos = torch.linspace(0, 4*np.pi, n_samples)
        t_neg = torch.linspace(0, 4*np.pi, n_samples) + np.pi
        
        pos = torch.zeros(n_samples, hidden_dim)
        pos[:, 0] = t_pos * torch.cos(t_pos) + 0.5 * torch.randn(n_samples)
        pos[:, 1] = t_pos * torch.sin(t_pos) + 0.5 * torch.randn(n_samples)
        pos[:, 2:] = torch.randn(n_samples, hidden_dim - 2) * 0.1
        
        neg = torch.zeros(n_samples, hidden_dim)
        neg[:, 0] = t_neg * torch.cos(t_neg) + 0.5 * torch.randn(n_samples)
        neg[:, 1] = t_neg * torch.sin(t_neg) + 0.5 * torch.randn(n_samples)
        neg[:, 2:] = torch.randn(n_samples, hidden_dim - 2) * 0.1
        
    else:  # random
        pos = torch.randn(n_samples, hidden_dim)
        neg = torch.randn(n_samples, hidden_dim)
    
    return pos, neg


def compute_roc_for_existence(
    real_results: List[Dict],
    null_scores: List[float],
) -> Tuple[List[float], List[float], List[float], float]:
    """
    Compute ROC curve for existence threshold.
    
    Args:
        real_results: Results from real benchmarks
        null_scores: kNN scores from null distribution
        
    Returns:
        (thresholds, tpr, fpr, auc)
    """
    # Labels: 1 for real data (should be detected), 0 for null (should not)
    real_knn = [r["nonlinear_metrics"]["knn_accuracy_k10"] for r in real_results]
    
    scores = real_knn + null_scores
    labels = [1] * len(real_knn) + [0] * len(null_scores)
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    return thresholds.tolist(), tpr.tolist(), fpr.tolist(), roc_auc


def compute_precision_recall_for_gap(
    results: List[Dict],
    ground_truth_linear: List[bool],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Compute precision-recall for gap threshold (linear vs nonlinear).
    
    Args:
        results: Results from benchmarks
        ground_truth_linear: Ground truth labels (True = linear, False = nonlinear)
        
    Returns:
        (thresholds, precision, recall, f1)
    """
    # Gap = signal_strength - linear_probe_accuracy
    gaps = [r["signal_strength"] - r["linear_probe_accuracy"] for r in results]
    
    # Labels: 1 for nonlinear (gap > threshold), 0 for linear
    labels = [0 if gt else 1 for gt in ground_truth_linear]
    
    precision, recall, thresholds = precision_recall_curve(labels, gaps)
    
    # Compute F1
    f1 = [2 * p * r / (p + r + 1e-10) for p, r in zip(precision, recall)]
    
    return thresholds.tolist(), precision.tolist(), recall.tolist(), f1


def run_sensitivity_analysis(
    results: List[Dict],
    existence_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7],
    gap_thresholds: List[float] = [0.05, 0.10, 0.15, 0.20, 0.25],
) -> Dict[str, Dict[str, float]]:
    """
    Run sensitivity analysis across threshold combinations.
    
    Args:
        results: Results from benchmarks
        existence_thresholds: Thresholds to test for existence
        gap_thresholds: Thresholds to test for gap
        
    Returns:
        Nested dict: {exist_thresh: {gap_thresh: {diagnosis: percentage}}}
    """
    sensitivity = {}
    
    for exist_t in existence_thresholds:
        sensitivity[str(exist_t)] = {}
        
        for gap_t in gap_thresholds:
            diagnoses = {"LINEAR": 0, "NONLINEAR": 0, "NO_SIGNAL": 0}
            
            for r in results:
                signal = r["signal_strength"]
                gap = signal - r["linear_probe_accuracy"]
                
                if signal < exist_t:
                    diagnoses["NO_SIGNAL"] += 1
                elif gap < gap_t:
                    diagnoses["LINEAR"] += 1
                else:
                    diagnoses["NONLINEAR"] += 1
            
            total = len(results)
            sensitivity[str(exist_t)][str(gap_t)] = {
                k: v / total * 100 for k, v in diagnoses.items()
            }
    
    return sensitivity


def load_diagnosis_results(model_name: str, output_dir: Path) -> List[Dict]:
    """Load all diagnosis results."""
    model_prefix = model_name.replace('/', '_')
    
    # Try to download from S3
    try:
        subprocess.run(
            ["aws", "s3", "sync", 
             f"s3://{S3_BUCKET}/direction_discovery/{model_prefix}/",
             str(output_dir / "diagnosis"),
             "--quiet"],
            check=False,
            capture_output=True,
        )
    except Exception:
        pass
    
    # Load all results
    all_results = []
    diagnosis_dir = output_dir / "diagnosis"
    
    if diagnosis_dir.exists():
        for f in diagnosis_dir.glob(f"{model_prefix}_*.json"):
            if "summary" not in f.name:
                with open(f) as fp:
                    data = json.load(fp)
                    all_results.extend(data.get("results", []))
    
    return all_results


def run_threshold_analysis(model_name: str):
    """
    Run full threshold analysis.
    
    Args:
        model_name: Model to analyze
    """
    print("=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)
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
    null_knn, null_linear = generate_null_distribution(None, n_samples=100, hidden_dim=4096)
    
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
    print("   " + "-" * 60)
    print(f"   {'Exist':>6} | {'Gap':>6} | {'LINEAR':>8} | {'NONLINEAR':>10} | {'NO_SIGNAL':>10}")
    print("   " + "-" * 60)
    
    for exist_t, gap_data in sensitivity.items():
        for gap_t, diagnoses in gap_data.items():
            print(f"   {exist_t:>6} | {gap_t:>6} | {diagnoses['LINEAR']:>7.1f}% | "
                  f"{diagnoses['NONLINEAR']:>9.1f}% | {diagnoses['NO_SIGNAL']:>9.1f}%")
    
    # 5. Save results
    analysis_result = ThresholdAnalysisResult(
        existence_thresholds=thresholds[:100],  # Limit for JSON
        existence_tpr=tpr[:100],
        existence_fpr=fpr[:100],
        existence_auc=roc_auc,
        optimal_existence_threshold=float(optimal_exist),
        gap_thresholds=[0.05, 0.10, 0.15, 0.20, 0.25],
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
        json.dump(asdict(analysis_result), f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    s3_upload_file(results_file, model_name)
    
    # Summary
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
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
    parser = argparse.ArgumentParser(description="Threshold analysis for RepScan")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model to analyze")
    args = parser.parse_args()
    
    run_threshold_analysis(args.model)
