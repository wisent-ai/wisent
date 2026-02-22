"""Helper utilities for visualization_gallery."""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

S3_BUCKET = "wisent-bucket"
S3_PREFIX = "visualizations"


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


def load_diagnosis_results(model_name: str, output_dir: Path) -> Dict[str, Any]:
    """Load diagnosis results from S3/local."""
    model_prefix = model_name.replace('/', '_')
    
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
    
    results = {}
    diagnosis_dir = output_dir / "diagnosis"
    if diagnosis_dir.exists():
        for f in diagnosis_dir.glob(f"{model_prefix}_*.json"):
            if "summary" not in f.name:
                category = f.stem.replace(f"{model_prefix}_", "")
                with open(f) as fp:
                    results[category] = json.load(fp)
    
    return results


def select_representative_benchmarks(
    diagnosis_results: Dict[str, Any],
    n_per_type: int = 2,
) -> Dict[str, List[str]]:
    """
    Select representative benchmarks for each diagnosis type.
    
    Args:
        diagnosis_results: Loaded diagnosis results
        n_per_type: Number of benchmarks per type
        
    Returns:
        Dict with keys 'LINEAR', 'NONLINEAR', 'NO_SIGNAL'
    """
    by_diagnosis = {"LINEAR": [], "NONLINEAR": [], "NO_SIGNAL": []}
    
    for category, data in diagnosis_results.items():
        results = data.get("results", [])
        seen = set()
        
        for r in results:
            bench = r["benchmark"]
            if bench in seen:
                continue
            seen.add(bench)
            
            signal = r["signal_strength"]
            linear = r["linear_probe_accuracy"]
            knn = r["nonlinear_metrics"]["knn_accuracy_k10"]
            
            if signal < 0.6:
                by_diagnosis["NO_SIGNAL"].append((bench, signal, linear, knn))
            elif linear > 0.6 and (signal - linear) < 0.15:
                by_diagnosis["LINEAR"].append((bench, signal, linear, knn))
            else:
                by_diagnosis["NONLINEAR"].append((bench, signal, linear, knn))
    
    # Select best examples (highest separation for LINEAR/NONLINEAR, lowest for NO_SIGNAL)
    selected = {}
    
    # LINEAR: highest linear probe accuracy
    by_diagnosis["LINEAR"].sort(key=lambda x: x[2], reverse=True)
    selected["LINEAR"] = [b[0] for b in by_diagnosis["LINEAR"][:n_per_type]]
    
    # NONLINEAR: highest gap between kNN and linear
    by_diagnosis["NONLINEAR"].sort(key=lambda x: x[3] - x[2], reverse=True)
    selected["NONLINEAR"] = [b[0] for b in by_diagnosis["NONLINEAR"][:n_per_type]]
    
    # NO_SIGNAL: lowest signal
    by_diagnosis["NO_SIGNAL"].sort(key=lambda x: x[1])
    selected["NO_SIGNAL"] = [b[0] for b in by_diagnosis["NO_SIGNAL"][:n_per_type]]
    
    return selected


def create_tsne_plot(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    title: str,
    ax: plt.Axes,
    diagnosis: str,
) -> None:
    """
    Create t-SNE visualization on given axes.
    
    Args:
        pos_activations: [N, D] positive class
        neg_activations: [N, D] negative class
        title: Plot title
        ax: Matplotlib axes
        diagnosis: 'LINEAR', 'NONLINEAR', or 'NO_SIGNAL'
    """
    if not HAS_SKLEARN or not HAS_MATPLOTLIB:
        return
    
    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()
    
    X = np.vstack([pos, neg])
    labels = np.array([1] * len(pos) + [0] * len(neg))
    
    # Reduce dimensionality with PCA first for speed
    if X.shape[1] > 50:
        pca = PCA(n_components=50)
        X = pca.fit_transform(X)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(X) // 4), random_state=42)
    X_2d = tsne.fit_transform(X)
    
    # Color scheme based on diagnosis
    colors = {
        "LINEAR": ("#2ecc71", "#e74c3c"),  # Green/Red
        "NONLINEAR": ("#3498db", "#e67e22"),  # Blue/Orange
        "NO_SIGNAL": ("#95a5a6", "#7f8c8d"),  # Gray shades
    }
    pos_color, neg_color = colors.get(diagnosis, ("#2ecc71", "#e74c3c"))
    
    # Plot
    ax.scatter(X_2d[labels == 1, 0], X_2d[labels == 1, 1], 
               c=pos_color, label='Positive', alpha=0.7, s=30)
    ax.scatter(X_2d[labels == 0, 0], X_2d[labels == 0, 1], 
               c=neg_color, label='Negative', alpha=0.7, s=30)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add diagnosis label
    ax.text(0.02, 0.98, diagnosis, transform=ax.transAxes, 
            fontsize=10, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

