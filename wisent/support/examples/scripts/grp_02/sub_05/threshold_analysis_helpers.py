"""Helper functions and data classes for threshold_analysis."""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from wisent.core.constants import (
    N_BOOTSTRAP_DEFAULT, PAIR_GENERATORS_DEFAULT_N,
    THRESHOLD_HIDDEN_DIM_LARGE, THRESHOLD_HIDDEN_DIM_DEFAULT,
    ZERO_THRESHOLD, NULL_DISTRIBUTION_SAMPLES_PER_CLASS,
    EXISTENCE_THRESHOLD_GRID, GAP_THRESHOLD_CANDIDATES,
)

GCS_BUCKET = "wisent-images-bucket"
GCS_PREFIX = "threshold_analysis"


def gcs_upload_file(local_path: Path, model_name: str) -> None:
    """Upload a single file to GCS."""
    model_prefix = model_name.replace('/', '_')
    gcs_path = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/{model_prefix}/{local_path.name}"
    try:
        subprocess.run(
            ["gcloud", "storage", "cp", str(local_path), gcs_path, "--quiet"],
            check=True,
            capture_output=True,
        )
        print(f"  Uploaded to GCS: {gcs_path}")
    except Exception as e:
        print(f"  GCS upload failed: {e}")


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
    n_samples: int = N_BOOTSTRAP_DEFAULT,
    hidden_dim: int = THRESHOLD_HIDDEN_DIM_LARGE,
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
        pos = torch.randn(NULL_DISTRIBUTION_SAMPLES_PER_CLASS, hidden_dim)
        neg = torch.randn(NULL_DISTRIBUTION_SAMPLES_PER_CLASS, hidden_dim)
        
        knn = compute_knn_accuracy(pos, neg, k=10)
        linear = compute_linear_probe_accuracy(pos, neg)
        
        knn_scores.append(knn)
        linear_scores.append(linear)
    
    return knn_scores, linear_scores


def generate_synthetic_data(
    structure: str,
    n_samples: int = PAIR_GENERATORS_DEFAULT_N,
    hidden_dim: int = THRESHOLD_HIDDEN_DIM_DEFAULT,
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
    f1 = [2 * p * r / (p + r + ZERO_THRESHOLD) for p, r in zip(precision, recall)]
    
    return thresholds.tolist(), precision.tolist(), recall.tolist(), f1


def run_sensitivity_analysis(
    results: List[Dict],
    existence_thresholds: List[float] = EXISTENCE_THRESHOLD_GRID,
    gap_thresholds: List[float] = GAP_THRESHOLD_CANDIDATES,
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
    
    # Try to download from GCS
    try:
        subprocess.run(
            ["gcloud", "storage", "rsync",
             f"gs://{GCS_BUCKET}/direction_discovery/{model_prefix}/",
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

