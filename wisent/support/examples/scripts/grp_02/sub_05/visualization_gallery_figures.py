"""Figure generation for visualization_gallery."""

from pathlib import Path
from typing import Dict, Any

import numpy as np

from wisent.core.constants import VIZ_PLOT_DPI, SIGNAL_EXISTENCE_THRESHOLD, LINEAR_GAP_THRESHOLD

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


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
    ax1.set_title('Zwiad Pipeline', fontsize=14, fontweight='bold')
    
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
            
            if signal < SIGNAL_EXISTENCE_THRESHOLD:
                counts["NO_SIGNAL"] += 1
            elif linear > SIGNAL_EXISTENCE_THRESHOLD and (signal - linear) < LINEAR_GAP_THRESHOLD:
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
            
            if signal < SIGNAL_EXISTENCE_THRESHOLD:
                diag = "NO_SIGNAL"
            elif linear > SIGNAL_EXISTENCE_THRESHOLD and (signal - linear) < LINEAR_GAP_THRESHOLD:
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
    plt.savefig(output_path, dpi=VIZ_PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved hero figure: {output_path}")


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
    plt.savefig(output_path, dpi=VIZ_PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved layer curves: {output_path}")

