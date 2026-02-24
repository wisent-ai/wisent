"""Generate data for Zwiad paper."""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any

from wisent.examples.scripts.generate_paper_data_helpers import (
    GCS_BUCKET,
    download_all_results,
    load_model_results,
    compute_diagnosis,
    generate_main_results_table,
    generate_benchmark_table,
)


def generate_figure_data(all_models: Dict[str, Dict]) -> Dict[str, Any]:
    """Generate JSON data for figures."""
    figure_data = {
        "diagnosis_distribution": {"LINEAR": 0, "NONLINEAR": 0, "NO_SIGNAL": 0},
        "per_category": {},
        "top_benchmarks": {"linear": [], "nonlinear": [], "no_signal": []},
        "metrics_by_diagnosis": {
            "LINEAR": {"signal": [], "linear": [], "knn": [], "mmd": []},
            "NONLINEAR": {"signal": [], "linear": [], "knn": [], "mmd": []},
            "NO_SIGNAL": {"signal": [], "linear": [], "knn": [], "mmd": []},
        },
    }
    
    all_results = []
    
    for model_name, categories in all_models.items():
        for cat_name, data in categories.items():
            results = data.get("results", [])
            all_results.extend(results)
            
            if cat_name not in figure_data["per_category"]:
                figure_data["per_category"][cat_name] = {
                    "signal": [], "linear": [], "knn": []
                }
            
            for r in results:
                signal = r["signal_strength"]
                linear = r["linear_probe_accuracy"]
                knn = r["nonlinear_metrics"]["knn_accuracy_k10"]
                mmd = r["nonlinear_metrics"]["mmd_rbf"]
                diagnosis = compute_diagnosis(signal, linear)
                
                figure_data["diagnosis_distribution"][diagnosis] += 1
                figure_data["metrics_by_diagnosis"][diagnosis]["signal"].append(signal)
                figure_data["metrics_by_diagnosis"][diagnosis]["linear"].append(linear)
                figure_data["metrics_by_diagnosis"][diagnosis]["knn"].append(knn)
                figure_data["metrics_by_diagnosis"][diagnosis]["mmd"].append(mmd)
                
                figure_data["per_category"][cat_name]["signal"].append(signal)
                figure_data["per_category"][cat_name]["linear"].append(linear)
                figure_data["per_category"][cat_name]["knn"].append(knn)
    
    # Compute averages
    for diag in figure_data["metrics_by_diagnosis"]:
        for metric in list(figure_data["metrics_by_diagnosis"][diag].keys()):
            values = figure_data["metrics_by_diagnosis"][diag][metric]
            if values and isinstance(values, list):
                figure_data["metrics_by_diagnosis"][diag][f"{metric}_mean"] = sum(values) / len(values)
                figure_data["metrics_by_diagnosis"][diag][f"{metric}_std"] = (
                    sum((v - sum(values)/len(values))**2 for v in values) / len(values)
                ) ** 0.5
    
    # Top benchmarks per diagnosis
    benchmarks_by_diag = {"LINEAR": [], "NONLINEAR": [], "NO_SIGNAL": []}
    seen = set()
    
    for r in all_results:
        bench = r["benchmark"]
        if bench in seen:
            continue
        seen.add(bench)
        
        signal = r["signal_strength"]
        linear = r["linear_probe_accuracy"]
        knn = r["nonlinear_metrics"]["knn_accuracy_k10"]
        diagnosis = compute_diagnosis(signal, linear)
        
        benchmarks_by_diag[diagnosis].append({
            "benchmark": bench,
            "signal": signal,
            "linear": linear,
            "knn": knn,
            "gap": knn - linear,
        })
    
    # Sort and take top 5
    benchmarks_by_diag["LINEAR"].sort(key=lambda x: x["linear"], reverse=True)
    benchmarks_by_diag["NONLINEAR"].sort(key=lambda x: x["gap"], reverse=True)
    benchmarks_by_diag["NO_SIGNAL"].sort(key=lambda x: x["signal"])
    
    figure_data["top_benchmarks"] = {
        diag: benches[:5] for diag, benches in benchmarks_by_diag.items()
    }
    
    return figure_data


def generate_summary_statistics(all_models: Dict[str, Dict]) -> str:
    """Generate summary statistics for paper text."""
    total_results = 0
    total_linear = 0
    total_nonlinear = 0
    total_no_signal = 0
    
    categories = set()
    benchmarks = set()
    
    for model_name, model_categories in all_models.items():
        for cat_name, data in model_categories.items():
            categories.add(cat_name)
            results = data.get("results", [])
            total_results += len(results)
            
            for r in results:
                benchmarks.add(r["benchmark"])
                signal = r["signal_strength"]
                linear = r["linear_probe_accuracy"]
                diagnosis = compute_diagnosis(signal, linear)
                
                if diagnosis == "LINEAR":
                    total_linear += 1
                elif diagnosis == "NONLINEAR":
                    total_nonlinear += 1
                else:
                    total_no_signal += 1
    
    text = f"""
## Summary Statistics for Paper

- **Total tests**: {total_results:,}
- **Models tested**: {len(all_models)}
- **Categories**: {len(categories)} ({', '.join(sorted(categories))})
- **Unique benchmarks**: {len(benchmarks)}

### Diagnosis Distribution:
- **LINEAR (CAA-viable)**: {total_linear:,} ({100*total_linear/total_results:.1f}%)
- **NONLINEAR (manifold)**: {total_nonlinear:,} ({100*total_nonlinear/total_results:.1f}%)
- **NO_SIGNAL**: {total_no_signal:,} ({100*total_no_signal/total_results:.1f}%)

### Key Findings:
1. {100*total_linear/total_results:.0f}% of benchmarks have LINEAR representations suitable for CAA
2. {100*total_nonlinear/total_results:.0f}% have NONLINEAR representations requiring different methods
3. {100*total_no_signal/total_results:.0f}% show no detectable signal
"""
    return text


def main():
    """Generate all paper data."""
    print("=" * 70)
    print("GENERATING PAPER DATA")
    print("=" * 70)
    
    output_dir = Path("/tmp/paper_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download results
    print("\n1. Downloading results from GCS...")
    results_dir = output_dir / "results"
    models = download_all_results(results_dir)
    print(f"   Found {len(models)} models: {list(models.keys())}")
    
    # Load all results
    print("\n2. Loading results...")
    all_models = {}
    for model_name, model_dir in models.items():
        all_models[model_name] = load_model_results(model_dir)
        print(f"   {model_name}: {len(all_models[model_name])} categories")
    
    # Generate main table
    print("\n3. Generating main results table...")
    main_table = generate_main_results_table(all_models)
    with open(output_dir / "main_results_table.tex", "w") as f:
        f.write(main_table)
    print(f"   Saved: {output_dir / 'main_results_table.tex'}")
    
    # Generate benchmark table
    print("\n4. Generating benchmark table...")
    bench_table = generate_benchmark_table(all_models)
    with open(output_dir / "benchmark_table.tex", "w") as f:
        f.write(bench_table)
    print(f"   Saved: {output_dir / 'benchmark_table.tex'}")
    
    # Generate figure data
    print("\n5. Generating figure data...")
    figure_data = generate_figure_data(all_models)
    with open(output_dir / "figure_data.json", "w") as f:
        json.dump(figure_data, f, indent=2)
    print(f"   Saved: {output_dir / 'figure_data.json'}")
    
    # Generate summary statistics
    print("\n6. Generating summary statistics...")
    summary = generate_summary_statistics(all_models)
    with open(output_dir / "summary_statistics.md", "w") as f:
        f.write(summary)
    print(f"   Saved: {output_dir / 'summary_statistics.md'}")
    print(summary)
    
    # Upload to GCS
    print("\n7. Uploading to GCS...")
    for f in output_dir.glob("*"):
        if f.is_file():
            subprocess.run(
                ["gcloud", "storage", "cp", str(f), f"gs://{GCS_BUCKET}/paper_data/{f.name}", "--quiet"],
                check=False,
            )
    
    print("\n" + "=" * 70)
    print("PAPER DATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
