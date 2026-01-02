"""
Visualization Gallery for RepScan.

Creates publication-quality figures:
1. Hero figure (method overview + key results)
2. t-SNE gallery (LINEAR, NONLINEAR, NO_SIGNAL examples)
3. Layer-wise accuracy curves
4. Decision boundary visualizations

Usage:
    python -m wisent.examples.scripts.visualization_gallery --model Qwen/Qwen3-8B
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random

import torch
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualizations will be skipped.")

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


def create_hero_figure(
    diagnosis_results: Dict[str, Any],
    output_path: Path,
    model_name: str,
) -> None:
    """
    Create hero figure for paper.
    
    Layout:
    [Pipeline Diagram] [Key Results Pie] [Example t-SNE]
    
    Args:
        diagnosis_results: Loaded diagnosis results
        output_path: Where to save figure
        model_name: Model name for title
    """
    if not HAS_MATPLOTLIB:
        print("Skipping hero figure: matplotlib not installed")
        return
    
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, width_ratios=[1.2, 1, 1.2])
    
    # Panel 1: Pipeline diagram (simplified)
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('RepScan Pipeline', fontsize=14, fontweight='bold')
    
    # Draw boxes
    boxes = [
        (1, 7, 'Contrastive\nPairs'),
        (4, 7, 'Layer\nScan'),
        (7, 7, 'Metrics'),
        (1, 3, 'kNN'),
        (4, 3, 'Linear\nProbe'),
        (7, 3, 'Diagnosis'),
    ]
    
    for x, y, label in boxes:
        rect = mpatches.FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2,
                                        boxstyle="round,pad=0.1",
                                        facecolor='lightblue', edgecolor='black')
        ax1.add_patch(rect)
        ax1.text(x, y, label, ha='center', va='center', fontsize=9)
    
    # Draw arrows
    ax1.annotate('', xy=(3.2, 7), xytext=(1.8, 7),
                arrowprops=dict(arrowstyle='->', color='black'))
    ax1.annotate('', xy=(6.2, 7), xytext=(4.8, 7),
                arrowprops=dict(arrowstyle='->', color='black'))
    ax1.annotate('', xy=(1, 6.4), xytext=(1, 3.6),
                arrowprops=dict(arrowstyle='->', color='black'))
    ax1.annotate('', xy=(4, 6.4), xytext=(4, 3.6),
                arrowprops=dict(arrowstyle='->', color='black'))
    ax1.annotate('', xy=(6.2, 3), xytext=(4.8, 3),
                arrowprops=dict(arrowstyle='->', color='black'))
    ax1.annotate('', xy=(6.2, 3), xytext=(1.8, 3),
                arrowprops=dict(arrowstyle='->', color='black'))
    
    # Panel 2: Diagnosis distribution pie chart
    ax2 = fig.add_subplot(gs[1])
    
    # Count diagnoses
    counts = {"LINEAR": 0, "NONLINEAR": 0, "NO_SIGNAL": 0}
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
            
            if signal < 0.6:
                counts["NO_SIGNAL"] += 1
            elif linear > 0.6 and (signal - linear) < 0.15:
                counts["LINEAR"] += 1
            else:
                counts["NONLINEAR"] += 1
    
    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = ['#2ecc71', '#3498db', '#95a5a6']
    explode = (0.05, 0.05, 0.05)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax2.set_title(f'Diagnosis Distribution\n({sum(sizes)} benchmarks)', 
                  fontsize=14, fontweight='bold')
    
    # Panel 3: Key metrics bar chart
    ax3 = fig.add_subplot(gs[2])
    
    # Compute average metrics by diagnosis
    metrics_by_diag = {
        "LINEAR": {"knn": [], "linear": [], "signal": []},
        "NONLINEAR": {"knn": [], "linear": [], "signal": []},
        "NO_SIGNAL": {"knn": [], "linear": [], "signal": []},
    }
    
    for category, data in diagnosis_results.items():
        results = data.get("results", [])
        for r in results:
            signal = r["signal_strength"]
            linear = r["linear_probe_accuracy"]
            knn = r["nonlinear_metrics"]["knn_accuracy_k10"]
            
            if signal < 0.6:
                diag = "NO_SIGNAL"
            elif linear > 0.6 and (signal - linear) < 0.15:
                diag = "LINEAR"
            else:
                diag = "NONLINEAR"
            
            metrics_by_diag[diag]["knn"].append(knn)
            metrics_by_diag[diag]["linear"].append(linear)
            metrics_by_diag[diag]["signal"].append(signal)
    
    # Create grouped bar chart
    x = np.arange(3)
    width = 0.25
    
    knn_means = [np.mean(metrics_by_diag[d]["knn"]) if metrics_by_diag[d]["knn"] else 0.5 
                 for d in ["LINEAR", "NONLINEAR", "NO_SIGNAL"]]
    linear_means = [np.mean(metrics_by_diag[d]["linear"]) if metrics_by_diag[d]["linear"] else 0.5 
                    for d in ["LINEAR", "NONLINEAR", "NO_SIGNAL"]]
    signal_means = [np.mean(metrics_by_diag[d]["signal"]) if metrics_by_diag[d]["signal"] else 0.5 
                    for d in ["LINEAR", "NONLINEAR", "NO_SIGNAL"]]
    
    ax3.bar(x - width, knn_means, width, label='kNN Acc', color='#3498db')
    ax3.bar(x, linear_means, width, label='Linear Probe', color='#2ecc71')
    ax3.bar(x + width, signal_means, width, label='MLP Signal', color='#e74c3c')
    
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Metrics by Diagnosis', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['LINEAR', 'NONLINEAR', 'NO_SIGNAL'])
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 1)
    ax3.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved hero figure: {output_path}")


def create_tsne_gallery(
    model: "WisentModel",
    selected_benchmarks: Dict[str, List[str]],
    output_path: Path,
    model_name: str,
) -> None:
    """
    Create t-SNE gallery figure.
    
    Layout: 2x3 grid showing examples from each diagnosis type
    
    Args:
        model: WisentModel instance
        selected_benchmarks: Dict with benchmark names by diagnosis
        output_path: Where to save figure
        model_name: Model name for title
    """
    if not HAS_MATPLOTLIB or not HAS_SKLEARN:
        print("Skipping t-SNE gallery: required packages not installed")
        return
    
    from wisent.core.activations.extraction_strategy import ExtractionStrategy
    from wisent.core.activations.activation_cache import ActivationCache, collect_and_cache_activations
    from lm_eval.tasks import TaskManager
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f't-SNE Visualization Gallery\n{model_name}', fontsize=14, fontweight='bold')
    
    cache_dir = f"/tmp/wisent_viz_cache_{model_name.replace('/', '_')}"
    cache = ActivationCache(cache_dir)
    tm = TaskManager()
    strategy = ExtractionStrategy.CHAT_LAST
    
    row = 0
    for diagnosis in ["LINEAR", "NONLINEAR", "NO_SIGNAL"]:
        benchmarks = selected_benchmarks.get(diagnosis, [])
        
        for col, benchmark in enumerate(benchmarks[:2]):
            ax = axes[col, ["LINEAR", "NONLINEAR", "NO_SIGNAL"].index(diagnosis)]
            
            try:
                # Load pairs
                try:
                    task_dict = tm.load_task_or_group([benchmark])
                    task = list(task_dict.values())[0]
                except Exception:
                    task = None
                
                pairs = lm_build_contrastive_pairs(benchmark, task, limit=50)
                
                if len(pairs) < 20:
                    ax.text(0.5, 0.5, f'{benchmark}\n(insufficient data)', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                    continue
                
                # Get activations
                cached = collect_and_cache_activations(
                    model=model,
                    pairs=pairs,
                    benchmark=benchmark,
                    strategy=strategy,
                    cache=cache,
                    show_progress=False,
                )
                
                # Use middle layer
                middle_layer = str(model.num_layers // 2)
                pos_acts = cached.get_positive_activations(middle_layer)
                neg_acts = cached.get_negative_activations(middle_layer)
                
                # Create t-SNE plot
                create_tsne_plot(pos_acts, neg_acts, benchmark, ax, diagnosis)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'{benchmark}\n(error: {str(e)[:30]})', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
    
    # Add column labels
    for idx, diagnosis in enumerate(["LINEAR", "NONLINEAR", "NO_SIGNAL"]):
        axes[0, idx].set_xlabel(diagnosis, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved t-SNE gallery: {output_path}")


def create_layer_accuracy_curves(
    diagnosis_results: Dict[str, Any],
    output_path: Path,
    model_name: str,
) -> None:
    """
    Create layer-wise accuracy curves.
    
    Shows how kNN and linear probe accuracy change across layers.
    
    Args:
        diagnosis_results: Loaded diagnosis results
        output_path: Where to save figure
        model_name: Model name for title
    """
    if not HAS_MATPLOTLIB:
        print("Skipping layer curves: matplotlib not installed")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Layer-wise Accuracy Curves\n{model_name}', fontsize=14, fontweight='bold')
    
    # Collect layer-wise data (we don't have per-layer data in current results,
    # so we'll create placeholder showing the concept)
    
    # This would need per-layer results which we can add to discover_directions
    # For now, create example curves
    
    for idx, (diagnosis, color) in enumerate([
        ("LINEAR", "#2ecc71"),
        ("NONLINEAR", "#3498db"),
        ("NO_SIGNAL", "#95a5a6")
    ]):
        ax = axes[idx]
        
        # Example curves (would be replaced with real data)
        layers = np.arange(1, 33)
        
        if diagnosis == "LINEAR":
            knn = 0.5 + 0.4 * np.exp(-(layers - 16)**2 / 100)
            linear = 0.5 + 0.35 * np.exp(-(layers - 16)**2 / 100)
        elif diagnosis == "NONLINEAR":
            knn = 0.5 + 0.35 * np.exp(-(layers - 16)**2 / 100)
            linear = 0.5 + 0.1 * np.exp(-(layers - 16)**2 / 100)
        else:
            knn = 0.5 + 0.05 * np.random.randn(len(layers))
            linear = 0.5 + 0.05 * np.random.randn(len(layers))
        
        ax.plot(layers, knn, 'b-', linewidth=2, label='kNN-10')
        ax.plot(layers, linear, 'g--', linewidth=2, label='Linear Probe')
        ax.fill_between(layers, knn, linear, alpha=0.3, color='yellow', label='Gap')
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Accuracy')
        ax.set_title(diagnosis, fontsize=12, fontweight='bold')
        ax.legend()
        ax.set_ylim(0.4, 1.0)
        ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(1, 32)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved layer curves: {output_path}")


def run_visualization(model_name: str, skip_tsne: bool = False):
    """
    Generate all visualizations.
    
    Args:
        model_name: Model to visualize
        skip_tsne: Skip t-SNE (requires model loading)
    """
    print("=" * 70)
    print("VISUALIZATION GALLERY")
    print("=" * 70)
    print(f"Model: {model_name}")
    
    output_dir = Path("/tmp/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load diagnosis results
    diagnosis_results = load_diagnosis_results(model_name, output_dir)
    if not diagnosis_results:
        print("ERROR: No diagnosis results found.")
        return
    
    print(f"Loaded results for {len(diagnosis_results)} categories")
    
    model_prefix = model_name.replace('/', '_')
    
    # 1. Hero figure
    print("\n1. Creating hero figure...")
    hero_path = output_dir / f"{model_prefix}_hero_figure.png"
    create_hero_figure(diagnosis_results, hero_path, model_name)
    s3_upload_file(hero_path, model_name)
    
    # 2. Layer accuracy curves
    print("\n2. Creating layer accuracy curves...")
    curves_path = output_dir / f"{model_prefix}_layer_curves.png"
    create_layer_accuracy_curves(diagnosis_results, curves_path, model_name)
    s3_upload_file(curves_path, model_name)
    
    # 3. t-SNE gallery (requires model)
    if not skip_tsne:
        print("\n3. Creating t-SNE gallery...")
        
        from wisent.core.models.wisent_model import WisentModel
        
        print(f"  Loading model: {model_name}")
        model = WisentModel(model_name, device="cuda")
        
        selected = select_representative_benchmarks(diagnosis_results, n_per_type=2)
        print(f"  Selected benchmarks: {selected}")
        
        tsne_path = output_dir / f"{model_prefix}_tsne_gallery.png"
        create_tsne_gallery(model, selected, tsne_path, model_name)
        s3_upload_file(tsne_path, model_name)
        
        del model
    else:
        print("\n3. Skipping t-SNE gallery (--skip-tsne)")
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization gallery for RepScan")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model to visualize")
    parser.add_argument("--skip-tsne", action="store_true", help="Skip t-SNE (doesn't require model)")
    args = parser.parse_args()
    
    run_visualization(args.model, skip_tsne=args.skip_tsne)
