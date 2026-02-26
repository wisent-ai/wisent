"""Helper utilities for visualization_gallery."""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np
from wisent.core.constants import DEFAULT_RANDOM_SEED, VIZ_MARKER_SIZE_SMALL, PCA_GALLERY_COMPONENTS, VIZ_GALLERY_N_PER_TYPE, TSNE_PERPLEXITY_MAX, SIGNAL_EXIST_THRESHOLD, SIGNAL_LINEAR_GAP, VIZ_FONTSIZE_BODY, VIZ_FONTSIZE_SUBTITLE, VIZ_ALPHA_HIGH, VIZ_BBOX_ALPHA

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

GCS_BUCKET = "wisent-images-bucket"
GCS_PREFIX = "visualizations"


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


def load_diagnosis_results(model_name: str, output_dir: Path) -> Dict[str, Any]:
    """Load diagnosis results from GCS/local."""
    model_prefix = model_name.replace('/', '_')

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
    n_per_type: int = VIZ_GALLERY_N_PER_TYPE,
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
            
            if signal < SIGNAL_EXIST_THRESHOLD:
                by_diagnosis["NO_SIGNAL"].append((bench, signal, linear, knn))
            elif linear > SIGNAL_EXIST_THRESHOLD and (signal - linear) < SIGNAL_LINEAR_GAP:
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
    if X.shape[1] > PCA_GALLERY_COMPONENTS:
        pca = PCA(n_components=PCA_GALLERY_COMPONENTS)
        X = pca.fit_transform(X)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(TSNE_PERPLEXITY_MAX, len(X) // 4), random_state=DEFAULT_RANDOM_SEED)
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
               c=pos_color, label='Positive', alpha=VIZ_ALPHA_HIGH, s=VIZ_MARKER_SIZE_SMALL)
    ax.scatter(X_2d[labels == 0, 0], X_2d[labels == 0, 1],
               c=neg_color, label='Negative', alpha=VIZ_ALPHA_HIGH, s=VIZ_MARKER_SIZE_SMALL)
    
    ax.set_title(title, fontsize=VIZ_FONTSIZE_SUBTITLE, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add diagnosis label
    ax.text(0.02, 0.98, diagnosis, transform=ax.transAxes, 
            fontsize=VIZ_FONTSIZE_BODY, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=VIZ_BBOX_ALPHA))

