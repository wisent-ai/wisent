"""Visualize zwiad geometric profiles across benchmarks."""
import json
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from wisent.core.constants import VIZ_DPI, VIZ_MARKER_SIZE, VIZ_ALPHA_HIGH, VIZ_LINEWIDTH_THIN

ZWIAD_DIR = Path("zwiad_results")
MODEL_SLUG = "meta-llama_Llama-3.2-1B-Instruct"
OUTPUT_DIR = Path("zwiad_results/figures")
KEYS = [
    "signal_strength", "linear_probe_accuracy",
    "icd_top1_variance", "direction_stability_score",
    "consistency_consistency_score", "cloud_separation_ratio",
    "intrinsic_dim_ratio", "manifold_curvature",
    "fisher_fisher_gini", "concept_coherence",
    "signal_to_noise", "knn_accuracy",
]
METHOD_COLORS = {
    "CAA": "#2196F3", "Ostrze": "#9C27B0", "MLP": "#4CAF50",
    "TECZA": "#FF9800", "TETNO": "#00BCD4", "GROM": "#E91E63",
    "Concept Flow": "#795548",
}
PROFILE_COLORS = {
    "A-Pristine": "#1B5E20", "B-Strong": "#4CAF50",
    "C-Solid": "#81C784", "D-MultiConcept": "#FF9800",
    "E-Marginal": "#FFC107", "F-Nonlinear": "#E91E63",
    "G-Weak": "#9E9E9E", "H-Degenerate": "#424242",
}
CAT_MARKERS = {
    "coding": "s", "commonsense": "o", "math": "^",
    "safety": "D", "reasoning": "v", "knowledge": "p",
    "science": "*", "multilingual": "h", "hallucination": "X",
}


def get_cat(b):
    for c in CAT_MARKERS:
        if b.startswith(c):
            return c
    return "other"


def load_all():
    recs = []
    for f in sorted(ZWIAD_DIR.glob(f"{MODEL_SLUG}__*.json")):
        bench = f.stem.replace(f"{MODEL_SLUG}__", "")
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        m = data.get("metrics", {})
        if not m or m.get("signal_strength") is None:
            continue
        r = {"benchmark": bench, "category": get_cat(bench)}
        r["method"] = m.get("recommended_method") or data.get("recommended_method", "CAA")
        for k in KEYS:
            r[k] = m.get(k, 0.0)
        recs.append(r)
    return recs


def classify(r):
    s, c = r["signal_strength"], r["consistency_consistency_score"]
    cc, meth = r["concept_coherence"], r["method"]
    if s > 0.93 and c > 0.8:
        return "A-Pristine"
    if s > 0.93 and c > 0.6:
        return "B-Strong"
    if s > 0.80:
        return "C-Solid"
    if meth == "TECZA" or (cc > 0.4 and s > 0.65):
        return "D-MultiConcept"
    if s > 0.65:
        return "E-Marginal"
    if meth == "GROM" and s > 0.50:
        return "F-Nonlinear"
    if s > 0.50:
        return "G-Weak"
    return "H-Degenerate"


def plot_signal_curvature(recs):
    fig, ax = plt.subplots(figsize=(14, 10))
    for r in recs:
        p = r["_profile"]
        ax.scatter(r["signal_strength"], r["manifold_curvature"],
                   c=PROFILE_COLORS[p], marker=CAT_MARKERS.get(r["category"], "."),
                   s=60, alpha=VIZ_ALPHA_HIGH, edgecolors="white", linewidth=0.3)
    ax.set_xlabel("Signal Strength", fontsize=13)
    ax.set_ylabel("Manifold Curvature", fontsize=13)
    ax.set_title(f"Zwiad: Signal vs Curvature ({len(recs)} benchmarks)", fontsize=14)
    patches = [mpatches.Patch(color=c, label=k) for k, c in PROFILE_COLORS.items()]
    cat_handles = [plt.Line2D([0], [0], marker=m, color="gray", linestyle="None",
                              markersize=8, label=c) for c, m in CAT_MARKERS.items()]
    leg1 = ax.legend(handles=patches, loc="upper left", fontsize=9, title="Profile")
    ax.add_artist(leg1)
    ax.legend(handles=cat_handles, loc="lower right", fontsize=8, title="Category")
    for x in [0.50, 0.65, 0.80, 0.93]:
        ax.axvline(x=x, color="gray", ls="--", alpha=0.4, lw=VIZ_LINEWIDTH_THIN)
    ax.set_xlim(0.3, 1.05)
    ax.set_ylim(-0.05, 1.0)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "signal_vs_curvature.png", dpi=VIZ_DPI)
    print(f"Saved signal_vs_curvature.png")
    plt.close(fig)


def plot_heatmap(recs):
    profiles = {}
    for r in recs:
        profiles.setdefault(r["_profile"], []).append(r)
    order = sorted(profiles.keys())
    dk = ["signal_strength", "consistency_consistency_score",
          "cloud_separation_ratio", "manifold_curvature",
          "concept_coherence", "icd_top1_variance",
          "fisher_fisher_gini", "knn_accuracy"]
    matrix = np.array([[np.nanmean([r[k] for r in profiles[p] if r[k] is not None] or [0])
                        for k in dk] for p in order])
    labels = [f"{p} (n={len(profiles[p])})" for p in order]
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(dk)))
    ax.set_xticklabels([k.replace("consistency_", "").replace("_", "\n") for k in dk],
                       fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(dk)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v < 0.3 or v > 0.7 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Mean Metrics per Profile ({len(recs)} benchmarks)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "profile_heatmap.png", dpi=VIZ_DPI)
    print(f"Saved profile_heatmap.png")
    plt.close(fig)


def plot_method_dist(recs):
    profiles = {}
    for r in recs:
        profiles.setdefault(r["_profile"], []).append(r)
    order = sorted(profiles.keys())
    methods = ["CAA", "Ostrze", "MLP", "TECZA", "TETNO", "GROM", "Concept Flow"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(order))
    w = 0.11
    for i, m in enumerate(methods):
        counts = [sum(1 for r in profiles[p] if r["method"] == m) for p in order]
        bars = ax.bar(x + i * w, counts, w, label=m, color=METHOD_COLORS[m], alpha=0.85)
        for bar, c in zip(bars, counts):
            if c > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        str(c), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + w * 3)
    ax.set_xticklabels(order, fontsize=10)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Heuristic Method Recommendation per Profile ({len(recs)} benchmarks)\n"
                 "(Ostrze/TETNO/Nurt never recommended by current heuristic)",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "method_by_profile.png", dpi=VIZ_DPI)
    print(f"Saved method_by_profile.png")
    plt.close(fig)


def plot_consistency_sep(recs):
    fig, ax = plt.subplots(figsize=(12, 9))
    for r in recs:
        ax.scatter(r["consistency_consistency_score"],
                   min(r["cloud_separation_ratio"], 4.0),
                   c=PROFILE_COLORS[r["_profile"]], s=VIZ_MARKER_SIZE, alpha=0.65,
                   edgecolors="white", linewidth=0.3)
    ax.set_xlabel("Consistency Score", fontsize=13)
    ax.set_ylabel("Cloud Separation Ratio (capped 4)", fontsize=13)
    ax.set_title(f"Consistency vs Separation ({len(recs)} benchmarks)", fontsize=14)
    patches = [mpatches.Patch(color=c, label=k) for k, c in PROFILE_COLORS.items()]
    ax.legend(handles=patches, loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "consistency_vs_separation.png", dpi=VIZ_DPI)
    print(f"Saved consistency_vs_separation.png")
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recs = load_all()
    print(f"Loaded {len(recs)} benchmarks")
    for r in recs:
        r["_profile"] = classify(r)
    counts = Counter(r["_profile"] for r in recs)
    print("\nProfile distribution:")
    for p in sorted(counts):
        print(f"  {p}: {counts[p]}")
    plot_signal_curvature(recs)
    plot_heatmap(recs)
    plot_method_dist(recs)
    plot_consistency_sep(recs)
    print("\nAll figures in zwiad_results/figures/")
