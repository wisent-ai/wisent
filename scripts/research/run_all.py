#!/usr/bin/env python3
"""
Run All Research Analyses

Execute all four research questions across all models and all layers.

Usage:
    python -m scripts.research.run_all --output results/
    python -m scripts.research.run_all --model "Qwen/Qwen3-8B" --output results/  # Single model
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from .common import (
    load_activations_from_db,
    get_model_info,
    RESEARCH_MODELS,
)
from .q1_strategy_comparison import analyze_strategy_performance, summarize_strategy_results
from .q2_repscan_correlation import analyze_repscan_correlation, interpret_correlations
from .q3_benchmark_maximum import analyze_per_benchmark_maximum
from .q4_unified_direction import analyze_unified_direction


def run_layer_analysis(activations: Dict, questions: List[int] = None) -> Dict[str, Any]:
    """
    Run research questions for a single layer's activations.

    Args:
        activations: Dict mapping benchmark names to activation data
        questions: List of question numbers to run (default: all)

    Returns:
        Results dict for this layer
    """
    if questions is None:
        questions = [1, 2, 3, 4]

    total_activations = sum(len(v) for v in activations.values())
    num_benchmarks = len(activations)

    if total_activations == 0:
        return {"error": "No activations found"}

    results = {
        "data_summary": {
            "total_activations": total_activations,
            "num_benchmarks": num_benchmarks,
        },
    }

    strategy_results = None

    # Q1: Strategy comparison
    if 1 in questions:
        strategy_results = analyze_strategy_performance(activations)
        q1_summary = summarize_strategy_results(strategy_results)
        results["q1_strategy_comparison"] = {
            "summary": q1_summary,
            "per_benchmark": {
                b: {
                    "best_strategy": r.best_strategy,
                    "best_accuracy": r.best_accuracy,
                    "all_strategies": r.strategies,
                }
                for b, r in strategy_results.items()
            },
        }

    # Q2: RepScan correlation
    if 2 in questions:
        q2_results = analyze_repscan_correlation(activations)
        results["q2_repscan_correlation"] = {
            "results": q2_results,
            "interpretations": interpret_correlations(q2_results.get("correlations", {})) if "correlations" in q2_results else {},
        }

    # Q3: Per-benchmark maximum (needs Q1 results)
    if 3 in questions:
        if strategy_results is None:
            strategy_results = analyze_strategy_performance(activations)
        q3_results = analyze_per_benchmark_maximum(strategy_results)
        results["q3_per_benchmark_maximum"] = {"results": q3_results}

    # Q4: Unified direction
    if 4 in questions:
        q4_results = analyze_unified_direction(activations)
        results["q4_unified_direction"] = {"results": q4_results}

    return results


def run_model_analysis(model_name: str, output_dir: Optional[str] = None, questions: List[int] = None) -> Dict[str, Any]:
    """
    Run analysis for all layers of a single model.

    Args:
        model_name: Model to analyze
        output_dir: Directory to save results
        questions: List of question numbers to run

    Returns:
        Results dict for this model across all layers
    """
    if questions is None:
        questions = [1, 2, 3, 4]

    print(f"\n{'='*60}")
    print(f"Analyzing model: {model_name}")
    print(f"Questions: {questions}")
    print(f"{'='*60}")

    # Get model info
    model_info = get_model_info(model_name)
    num_layers = model_info["num_layers"]
    print(f"Model has {num_layers} layers")

    results_by_layer = {}

    for layer in range(num_layers):
        print(f"\n  Layer {layer}/{num_layers-1}...", end=" ", flush=True)

        try:
            activations = load_activations_from_db(model_name, layer)
            if not activations:
                print("no data")
                continue

            layer_results = run_layer_analysis(activations, questions)
            results_by_layer[layer] = layer_results

            # Print brief summary
            if "error" not in layer_results:
                summary_parts = []
                if "q1_strategy_comparison" in layer_results:
                    q1_best = layer_results["q1_strategy_comparison"]["summary"].get("overall_best_strategy", "N/A")
                    summary_parts.append(f"best_strategy={q1_best}")
                if "q3_per_benchmark_maximum" in layer_results:
                    q3_mean = layer_results["q3_per_benchmark_maximum"]["results"].get("summary", {}).get("mean_best_accuracy", 0)
                    summary_parts.append(f"mean_acc={q3_mean:.3f}")
                print(f"done ({', '.join(summary_parts) if summary_parts else 'complete'})")
            else:
                print(f"error: {layer_results['error']}")

        except Exception as e:
            print(f"failed: {e}")
            results_by_layer[layer] = {"error": str(e)}

    # Find best layer
    best_layer = None
    best_accuracy = 0
    for layer, results in results_by_layer.items():
        if "error" in results:
            continue
        acc = results.get("q3_per_benchmark_maximum", {}).get("results", {}).get("summary", {}).get("mean_best_accuracy", 0)
        if acc > best_accuracy:
            best_accuracy = acc
            best_layer = layer

    model_results = {
        "model": model_name,
        "num_layers": num_layers,
        "best_layer": best_layer,
        "best_layer_accuracy": best_accuracy,
        "results_by_layer": results_by_layer,
    }

    # Save model results
    if output_dir:
        model_dir = os.path.join(output_dir, model_name.replace("/", "_"))
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "results.json")
        with open(model_path, 'w') as f:
            json.dump(model_results, f, indent=2, default=str)
        print(f"\n  Saved to {model_path}")

    return model_results


def run_all_models(models: List[str], output_dir: Optional[str] = None, questions: List[int] = None) -> Dict[str, Any]:
    """
    Run analysis for all models.

    Args:
        models: List of model names to analyze
        output_dir: Directory to save results
        questions: List of question numbers to run

    Returns:
        Combined results across all models
    """
    if questions is None:
        questions = [1, 2, 3, 4]

    print(f"\n{'='*60}")
    print(f"RESEARCH ANALYSIS: {len(models)} models, Questions: {questions}")
    print(f"{'='*60}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    for model_name in models:
        try:
            model_results = run_model_analysis(model_name, output_dir, questions)
            all_results[model_name] = model_results
        except Exception as e:
            print(f"\nFailed to analyze {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}

    # Aggregate summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "models_analyzed": len(models),
        "models": models,
        "best_per_model": {
            model: {
                "best_layer": results.get("best_layer"),
                "best_accuracy": results.get("best_layer_accuracy", 0),
            }
            for model, results in all_results.items()
            if "error" not in results
        },
    }

    combined = {
        "summary": summary,
        "models": all_results,
    }

    if output_dir:
        combined_path = os.path.join(output_dir, "all_models_results.json")
        with open(combined_path, 'w') as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"\n\nCombined results saved to {combined_path}")

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for model, results in all_results.items():
        if "error" in results:
            print(f"  {model}: ERROR - {results['error']}")
        else:
            print(f"  {model}: best_layer={results['best_layer']}, accuracy={results['best_layer_accuracy']:.3f}")

    return combined


def main():
    parser = argparse.ArgumentParser(description="Run all research analyses")
    parser.add_argument("--output", type=str, default="research_results", help="Output directory")
    parser.add_argument("--questions", type=str, default="1,2,3,4", help="Comma-separated question numbers to run (e.g., '1,2')")
    args = parser.parse_args()

    questions = [int(q.strip()) for q in args.questions.split(",")]
    run_all_models(RESEARCH_MODELS, args.output, questions=questions)


if __name__ == "__main__":
    main()
