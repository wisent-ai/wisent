"""Cluster 211 benchmarks by their zwiad geometry metrics."""
import json
from pathlib import Path
from wisent.core.utils.config_tools.constants import (NEAR_ZERO_TOL, DEFAULT_RANDOM_SEED, KMEANS_N_INIT_DEFAULT,
    ZWIAD_PCA_COMPONENTS, ZWIAD_KMEANS_PRIMARY_CLUSTERS, ZWIAD_KMEANS_SECONDARY_CLUSTERS,
    VIZ_DPI)

ZWIAD_DIR = Path("/tmp/zwiad_all")
PREFIX = "meta-llama_Llama-3.2-1B-Instruct__"

METRIC_KEYS = [
    ("linear_probe_accuracy", "linear_probe"),
    ("mlp_probe_accuracy", "mlp_probe"),
    ("icd_icd", "icd"),
    ("direction_stability_score", "direction_stability"),
    ("n_concepts",),
    ("concept_coherence",),
    ("best_silhouette",),
    ("manifold_variance_pc1",),
    ("steer_effective_steering_dims",),
    ("multi_dir_gain",),
    ("steer_diff_mean_alignment", "diff_mean_alignment"),
    ("icd_top5_variance",),
    ("manifold_local_linearity", "manifold_local_linearity_mean"),
    ("manifold_curvature", "manifold_curvature_proxy"),
    ("consistency_consistency_score", "consistency_score"),
    ("sparsity_diff_gini",),
]

METRIC_NAMES = [
    "linear_probe", "mlp_probe", "icd", "stability",
    "n_concepts", "coherence", "silhouette", "var_pc1",
    "eff_dims", "multi_dir_gain", "alignment", "icd_top5",
    "local_linearity", "curvature", "consistency",
    "sparsity_gini",
]


def extract(metrics, keys):
    for k in keys:
        if k in metrics and metrics[k] is not None:
            return float(metrics[k])
    return None


def load_all():
    import numpy as np
    benchmarks = []
    vectors = []
    for f in sorted(ZWIAD_DIR.glob(f"{PREFIX}*.json")):
        bench = f.stem.replace(PREFIX, "")
        data = json.loads(f.read_text())
        m = data.get("metrics", data)
        row = []
        skip = False
        for keys in METRIC_KEYS:
            val = extract(m, keys)
            if val is None:
                skip = True
                break
            row.append(val)
        if skip:
            continue
        benchmarks.append(bench)
        vectors.append(row)
    import numpy as np
    return benchmarks, np.array(vectors)


def main():
    benchmarks, X = load_all()
    print(f"Loaded {len(benchmarks)} benchmarks, "
          f"{X.shape[1]} metrics each")

    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    Z = (X - mu) / std

    from sklearn.decomposition import PCA
    pca = PCA(n_components=ZWIAD_PCA_COMPONENTS)
    P = pca.fit_transform(Z)
    print(f"\nPCA variance: "
          f"{pca.explained_variance_ratio_[:3].round(3)}")
    for i in range(3):
        loads = list(zip(METRIC_NAMES, pca.components_[i]))
        loads.sort(key=lambda x: abs(x[1]), reverse=True)
        top = [(n, f"{v:.3f}") for n, v in loads[:5]]
        print(f"  PC{i+1}: {top}")

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    print("\n--- K-means silhouette by k ---")
    for k in range(4, 12):
        km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT_DEFAULT, random_state=DEFAULT_RANDOM_SEED)
        labels = km.fit_predict(Z)
        sil = silhouette_score(Z, labels)
        sizes = sorted(
            [int((labels == i).sum()) for i in range(k)],
            reverse=True)
        print(f"k={k:2d}  sil={sil:.3f}  sizes={sizes}")

    km8 = KMeans(n_clusters=ZWIAD_KMEANS_PRIMARY_CLUSTERS, n_init=KMEANS_N_INIT_DEFAULT, random_state=DEFAULT_RANDOM_SEED)
    labels8 = km8.fit_predict(Z)

    print("\n--- Cluster profiles (k=8) ---")
    for c in range(ZWIAD_KMEANS_PRIMARY_CLUSTERS):
        mask = labels8 == c
        n = mask.sum()
        centroid = X[mask].mean(axis=0)
        ex = [benchmarks[i] for i in np.where(mask)[0][:5]]
        print(f"\nCluster {c} ({n} benchmarks): {ex}")
        for name, val in zip(METRIC_NAMES, centroid):
            print(f"  {name:18s} = {val:.3f}")
        lp, mp = centroid[0], centroid[1]
        gap = mp - lp
        stab, nc = centroid[3], centroid[4]
        vpc1, ed = centroid[7], centroid[8]
        curv, align = centroid[13], centroid[10]
        traits = []
        if lp >= 0.85:
            traits.append("HIGH_LINEAR")
        elif lp < 0.6:
            traits.append("LOW_LINEAR")
        if gap > 0.05:
            traits.append("NONLINEAR_GAP")
        if nc > 10:
            traits.append("MULTI_CONCEPT")
        elif nc <= 3:
            traits.append("FEW_CONCEPTS")
        if stab > 0.98:
            traits.append("VERY_STABLE")
        elif stab < 0.9:
            traits.append("UNSTABLE")
        if vpc1 > 0.4:
            traits.append("CONCENTRATED")
        elif vpc1 < 0.2:
            traits.append("DIFFUSE")
        if ed > 30:
            traits.append("HIGH_DIM")
        elif ed < 15:
            traits.append("LOW_DIM")
        if curv > 0.3:
            traits.append("CURVED")
        if align > 0.3:
            traits.append("HIGH_ALIGN")
        elif align < 0.1:
            traits.append("LOW_ALIGN")
        print(f"  Traits: {traits}")

    plot_clusters(Z, labels8, benchmarks, pca)

    # Also run k=5 to see which k=8 clusters merge
    km5 = KMeans(n_clusters=ZWIAD_KMEANS_SECONDARY_CLUSTERS, n_init=KMEANS_N_INIT_DEFAULT, random_state=DEFAULT_RANDOM_SEED)
    labels5 = km5.fit_predict(Z)
    print("\n--- Which k=8 clusters merge at k=5 ---")
    for c5 in range(ZWIAD_KMEANS_SECONDARY_CLUSTERS):
        mask5 = labels5 == c5
        # Which k=8 labels fall in this k=5 cluster
        sub_labels = labels8[mask5]
        from collections import Counter
        counts = Counter(sub_labels)
        merged = [(f"C{l}({n})" ) for l, n in
                  counts.most_common()]
        n = mask5.sum()
        centroid = X[mask5].mean(axis=0)
        lp = centroid[0]
        stab = centroid[3]
        nc = centroid[4]
        vpc1 = centroid[7]
        curv = centroid[13]
        print(f"\nk5-Cluster {c5} ({n} benchmarks): "
              f"merges {merged}")
        print(f"  linear={lp:.3f} stability={stab:.3f} "
              f"n_concepts={nc:.0f} var_pc1={vpc1:.3f} "
              f"curvature={curv:.3f}")


def plot_clusters(Z, labels, benchmarks, pca):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = pca.transform(Z)
    k = labels.max() + 1
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    cluster_names = {
        0: "Linear stable\n(n=47)",
        1: "Diffuse curved\n(n=41)",
        2: "Concentrated\nfew-concept (n=11)",
        3: "Linear conc.\nlow-dim (n=26)",
        4: "Mid-range\nmulti-concept (n=39)",
        5: "Linear high\nmulti-dir (n=11)",
        6: "Low-linear\nunstable (n=15)",
        7: "Many-concept\ncurved (n=18)",
    }
    colors = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
    ]

    # PC1 vs PC2
    ax = axes[0]
    for c in range(k):
        mask = labels == c
        ax.scatter(
            P[mask, 0], P[mask, 1],
            c=colors[c], label=cluster_names.get(c, f"C{c}"),
            alpha=0.7, s=40, edgecolors="k", linewidth=0.3)
    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
        " — coherence, linearity, alignment")
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
        " — var_pc1, eff_dims, silhouette")
    ax.set_title("Activation Space Geometry Clusters (PC1 vs PC2)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # PC1 vs PC3
    ax = axes[1]
    for c in range(k):
        mask = labels == c
        ax.scatter(
            P[mask, 0], P[mask, 2],
            c=colors[c], label=cluster_names.get(c, f"C{c}"),
            alpha=0.7, s=40, edgecolors="k", linewidth=0.3)
    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
        " — coherence, linearity, alignment")
    ax.set_ylabel(
        f"PC3 ({pca.explained_variance_ratio_[2]:.1%})"
        " — multi_dir_gain, sparsity, n_concepts")
    ax.set_title("Activation Space Geometry Clusters (PC1 vs PC3)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = ("/Users/lukaszbartoszcze/Documents/CodingProjects/"
           "Wisent/backends/wisent-open-source/"
           "zwiad_clusters.png")
    plt.savefig(out, dpi=VIZ_DPI)
    print(f"\nSaved plot to {out}")


def classify():
    """Run the GeometryType classifier on all benchmarks."""
    import importlib.util, sys
    mod_path = (Path(__file__).resolve().parents[3]
                / "wisent/core/geometry/zwiad/geometry_types.py")
    spec = importlib.util.spec_from_file_location(
        "geometry_types", mod_path)
    gt = importlib.util.module_from_spec(spec)
    sys.modules["geometry_types"] = gt
    spec.loader.exec_module(gt)
    from collections import Counter
    shape_map = getattr(gt, "SHAPE_MAP", {})
    for fine in (False, True):
        enum_cls = gt.GeometryTypeFine if fine else gt.GeometryType
        counts, typed = Counter(), {t: [] for t in enum_cls}
        for f in sorted(ZWIAD_DIR.glob(f"{PREFIX}*.json")):
            bench = f.stem.replace(PREFIX, "")
            m = json.loads(f.read_text())
            m = m.get("metrics", m)
            gtype, conf = gt.classify_geometry(m, fine=fine)
            counts[gtype] += 1
            typed[gtype].append((bench, conf))
        label = "Fine (k=8)" if fine else "Coarse (k=5)"
        print(f"\n=== {label} Distribution ===")
        for gtype in enum_cls:
            top = [b for b, _ in sorted(typed[gtype], key=lambda x: x[1], reverse=True)[:3]]
            sh = f" [{shape_map[gtype]}]" if gtype in shape_map else ""
            print(f"{gtype.value:30s} {counts[gtype]:3d}{sh}  top: {top}")
        sel = gt.select_representative_benchmarks(str(ZWIAD_DIR), PREFIX.rstrip("_"), 2, fine)
        print(f"\n=== Selected (2/type, {sum(len(v) for v in sel.values())} total) ===")
        for gtype in enum_cls:
            print(f"  {gtype.value:30s} {sel[gtype]}")


def centroids_k8():
    """Print k=8 centroids for the 6 key metrics."""
    import numpy as np
    _, X = load_all()
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=ZWIAD_KMEANS_PRIMARY_CLUSTERS, n_init=KMEANS_N_INIT_DEFAULT, random_state=DEFAULT_RANDOM_SEED)
    labels = km.fit_predict((X - X.mean(0)) / X.std(0).clip(NEAR_ZERO_TOL))
    keys = ["linear", "stability", "n_concepts", "var_pc1", "curvature", "coherence"]
    idx = [0, 3, 4, 7, 13, 5]
    for c in range(ZWIAD_KMEANS_PRIMARY_CLUSTERS):
        mask = labels == c
        cent = X[mask].mean(axis=0)
        vals = {k: round(float(cent[i]), 3) for k, i in zip(keys, idx)}
        print(f"C{c} (n={int(mask.sum())}): {vals}")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "main"
    {"classify": classify, "centroids": centroids_k8, "main": main}.get(cmd, main)()
