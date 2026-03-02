#!/usr/bin/env python3
"""Extract key metrics from zwiad sample JSON files."""

import json
import os
import glob
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else (
    "/Users/lukaszbartoszcze/Documents/CodingProjects"
    "/Wisent/backends/zwiad_sample_copy"
)

CANONICAL = {
    "linear_probe": ["linear_probe_accuracy", "linear_probe"],
    "mlp_probe": ["mlp_probe_accuracy", "mlp_probe"],
    "icd": ["icd_icd", "icd"],
    "dir_stability": [
        "direction_stability_score", "direction_stability",
    ],
    "n_concepts": ["n_concepts"],
    "concept_coh": ["concept_coherence"],
    "best_silh": ["best_silhouette"],
    "mani_var_pc1": ["manifold_variance_pc1"],
    "steer_eff_d": ["steer_effective_steering_dims"],
    "multi_d_gain": ["multi_dir_gain"],
}


def find_metric(data, metrics, aliases):
    """Search metrics then top-level for first alias."""
    for alias in aliases:
        if alias in metrics:
            return metrics[alias]
    for alias in aliases:
        if alias in data:
            return data[alias]
    return None


def main():
    files = sorted(glob.glob(os.path.join(BASE, "*.json")))
    results = {}
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        metrics = data.get("metrics", {})
        fname = os.path.basename(f)
        bench = fname.split("__", 1)[1].replace(".json", "")
        row = {}
        for canon, aliases in CANONICAL.items():
            val = find_metric(data, metrics, aliases)
            if val is None:
                row[canon] = "N/A"
            elif isinstance(val, float):
                row[canon] = f"{val:.3f}"
            else:
                row[canon] = str(val)
        results[bench] = row
    cols = list(CANONICAL.keys())
    hdr = f"{'Benchmark':<35}" + "".join(
        f"{c:>14}" for c in cols
    )
    print(hdr)
    print("-" * len(hdr))
    for bench, row in results.items():
        line = f"{bench:<35}"
        for c in cols:
            line += f"{row[c]:>14}"
        print(line)


if __name__ == "__main__":
    main()
