"""Helper functions for generate_paper_data."""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict
from wisent.core.utils.config_tools.constants import DISPLAY_TOP_N_MEDIUM

GCS_BUCKET = "wisent-images-bucket"


def download_all_results(output_dir: Path) -> Dict[str, Path]:
    """Download all results from GCS."""
    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["gcloud", "storage", "rsync",
         f"gs://{GCS_BUCKET}/direction_discovery/",
         str(output_dir),
         "--quiet"],
        check=False,
        capture_output=True,
    )
    
    models = {}
    for d in output_dir.iterdir():
        if d.is_dir():
            models[d.name] = d
    
    return models


def load_model_results(model_dir: Path) -> Dict[str, Any]:
    """Load all category results for a model."""
    results = {}
    for f in model_dir.glob("*.json"):
        if "summary" in f.name:
            continue
        category = f.stem.split("_")[-1]
        with open(f) as fp:
            results[category] = json.load(fp)
    return results


def compute_diagnosis(signal: float, linear: float) -> str:
    """Compute diagnosis from signal and linear probe accuracy."""
    if signal < 0.6:
        return "NO_SIGNAL"
    elif linear > 0.6 and (signal - linear) < 0.15:
        return "LINEAR"
    else:
        return "NONLINEAR"


def generate_main_results_table(all_models: Dict[str, Dict]) -> str:
    """Generate main results table in LaTeX."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Zwiad diagnosis results across models and categories. Signal = MLP CV accuracy, Linear = Linear probe CV accuracy, kNN = k-NN CV accuracy (k=10). Diagnosis: LINEAR indicates CAA-viable representation, NONLINEAR indicates manifold structure, NO\_SIGNAL indicates no detectable representation.}",
        r"\label{tab:main_results}",
        r"\small",
        r"\begin{tabular}{llccccl}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Category} & \textbf{Signal} & \textbf{Linear} & \textbf{kNN} & \textbf{Gap} & \textbf{Diagnosis} \\",
        r"\midrule",
    ]
    
    for model_name, categories in sorted(all_models.items()):
        model_short = model_name.replace("meta-llama_", "").replace("Qwen_", "").replace("openai_", "")
        first_row = True
        
        for cat_name in sorted(categories.keys()):
            data = categories[cat_name]
            results = data.get("results", [])
            if not results:
                continue
            
            n = len(results)
            avg_signal = sum(r["signal_strength"] for r in results) / n
            avg_linear = sum(r["linear_probe_accuracy"] for r in results) / n
            avg_knn = sum(r["nonlinear_metrics"]["knn_accuracy_k10"] for r in results) / n
            gap = avg_signal - avg_linear
            diagnosis = compute_diagnosis(avg_signal, avg_linear)
            
            # Color coding for diagnosis
            if diagnosis == "LINEAR":
                diag_str = r"\textcolor{green!60!black}{LINEAR}"
            elif diagnosis == "NONLINEAR":
                diag_str = r"\textcolor{blue}{NONLINEAR}"
            else:
                diag_str = r"\textcolor{gray}{NO\_SIGNAL}"
            
            model_col = model_short if first_row else ""
            first_row = False
            
            lines.append(f"{model_col} & {cat_name} & {avg_signal:.2f} & {avg_linear:.2f} & {avg_knn:.2f} & {gap:+.2f} & {diag_str} \\\\")
        
        lines.append(r"\midrule")
    
    lines[-1] = r"\bottomrule"  # Replace last midrule
    lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def generate_benchmark_table(all_models: Dict[str, Dict]) -> str:
    """Generate benchmark list with contrastive definitions."""
    # Collect unique benchmarks
    benchmarks = defaultdict(lambda: {"categories": set(), "signal": [], "linear": [], "knn": []})
    
    for model_name, categories in all_models.items():
        for cat_name, data in categories.items():
            results = data.get("results", [])
            seen = set()
            for r in results:
                bench = r["benchmark"]
                if bench in seen:
                    continue
                seen.add(bench)
                
                benchmarks[bench]["categories"].add(cat_name)
                benchmarks[bench]["signal"].append(r["signal_strength"])
                benchmarks[bench]["linear"].append(r["linear_probe_accuracy"])
                benchmarks[bench]["knn"].append(r["nonlinear_metrics"]["knn_accuracy_k10"])
    
    lines = [
        r"\begin{longtable}{p{3cm}p{2.5cm}cccl}",
        r"\caption{Per-benchmark Zwiad results (averaged across models and strategies).} \label{tab:benchmarks} \\",
        r"\toprule",
        r"\textbf{Benchmark} & \textbf{Category} & \textbf{Signal} & \textbf{Linear} & \textbf{kNN} & \textbf{Diagnosis} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{6}{c}{\tablename\ \thetable{} -- continued} \\",
        r"\toprule",
        r"\textbf{Benchmark} & \textbf{Category} & \textbf{Signal} & \textbf{Linear} & \textbf{kNN} & \textbf{Diagnosis} \\",
        r"\midrule",
        r"\endhead",
    ]
    
    for bench, data in sorted(benchmarks.items(), key=lambda x: -max(x[1]["signal"]) if x[1]["signal"] else 0):
        cats = ", ".join(sorted(data["categories"]))[:DISPLAY_TOP_N_MEDIUM]
        avg_signal = sum(data["signal"]) / len(data["signal"]) if data["signal"] else 0
        avg_linear = sum(data["linear"]) / len(data["linear"]) if data["linear"] else 0
        avg_knn = sum(data["knn"]) / len(data["knn"]) if data["knn"] else 0
        diagnosis = compute_diagnosis(avg_signal, avg_linear)
        
        if diagnosis == "LINEAR":
            diag_str = r"\textcolor{green!60!black}{LINEAR}"
        elif diagnosis == "NONLINEAR":
            diag_str = r"\textcolor{blue}{NONLINEAR}"
        else:
            diag_str = r"\textcolor{gray}{NO\_SIG}"
        
        bench_escaped = bench.replace("_", r"\_")
        lines.append(f"{bench_escaped} & {cats} & {avg_signal:.2f} & {avg_linear:.2f} & {avg_knn:.2f} & {diag_str} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{longtable}",
    ])
    
    return "\n".join(lines)

