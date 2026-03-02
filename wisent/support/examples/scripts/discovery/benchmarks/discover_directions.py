"""
Discover unified directions for skill categories (coding, math, hallucination, etc.)

Uses GeometrySearchSpace to test all models, strategies, and layer combinations.
For each category, determines if a unified direction exists.

Usage:
    # Run for all models (sequentially)
    python -m wisent.examples.scripts.discover_directions
    
    # Run for a specific model (for parallel execution)
    python -m wisent.examples.scripts.discover_directions --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

from wisent.core.constants import PAIR_GENERATORS_DEFAULT_N, SEPARATOR_WIDTH_WIDE, JSON_INDENT
from wisent.core.reading import GeometrySearchSpace
from wisent.examples.scripts._discovery_utils import (
    load_categorized_benchmarks,
    load_category_directions,
    DiscoveryResults,
)
from wisent.examples.scripts._model_discovery import (
    run_discovery_for_model,
)

# Re-export all public names for backward compatibility
from wisent.examples.scripts._discovery_utils import (
    GCS_BUCKET,
    GCS_PREFIX,
    gcs_sync_download,
    gcs_upload_file,
    OptimalConfig,
    CategoryResult,
)
from wisent.examples.scripts._pairs_ablation import (
    run_pairs_ablation,
    find_optimal_config,
)
from wisent.examples.scripts._category_analysis import (
    analyze_category_results,
)


def generate_cross_model_comparison(all_model_results: Dict[str, "DiscoveryResults"]) -> Dict[str, Any]:
    """
    Generate cross-model comparison for each category.
    
    Identifies which model best represents each concept.
    """
    if not all_model_results:
        return {}
    
    comparison = {}
    
    # Get all categories across models
    all_categories = set()
    for results in all_model_results.values():
        all_categories.update(results.categories.keys())
    
    for category in all_categories:
        cat_comparison = {
            "best_model": None,
            "best_accuracy": 0.0,
            "models": {},
            "consensus_verdict": None,
            "consensus_is_linear": None,
        }
        
        verdicts = []
        linearities = []
        
        for model_name, results in all_model_results.items():
            if category not in results.categories:
                continue
                
            cat_result = results.categories[category]
            
            # Store per-model results
            cat_comparison["models"][model_name] = {
                "signal_exists": cat_result.signal_exists,
                "is_linear": cat_result.is_linear,
                "avg_linear_accuracy": cat_result.avg_linear_probe_accuracy,
                "avg_best_nonlinear": cat_result.avg_best_nonlinear,
                "recommendation": cat_result.recommendation,
                "optimal_layer": cat_result.optimal_config.optimal_layer if cat_result.optimal_config else None,
                "optimal_strategy": cat_result.optimal_config.optimal_strategy if cat_result.optimal_config else None,
                "min_pairs": cat_result.optimal_config.min_pairs_for_stable_signal if cat_result.optimal_config else None,
                "k_directions": cat_result.optimal_config.num_directions_needed if cat_result.optimal_config else None,
                "verdict": cat_result.optimal_config.verdict if cat_result.optimal_config else cat_result.signal_verdict,
            }
            
            # Track best model (by signal strength)
            accuracy = cat_result.avg_best_nonlinear
            if accuracy > cat_comparison["best_accuracy"]:
                cat_comparison["best_accuracy"] = accuracy
                cat_comparison["best_model"] = model_name
            
            # Collect for consensus
            if cat_result.optimal_config:
                verdicts.append(cat_result.optimal_config.verdict)
                linearities.append(cat_result.optimal_config.is_linear)
            elif cat_result.signal_verdict:
                verdicts.append(cat_result.signal_verdict)
                linearities.append(cat_result.is_linear)
        
        # Determine consensus
        if verdicts:
            from collections import Counter
            cat_comparison["consensus_verdict"] = Counter(verdicts).most_common(1)[0][0]
        if linearities:
            cat_comparison["consensus_is_linear"] = sum(linearities) / len(linearities) >= 0.5
        
        comparison[category] = cat_comparison
    
    return comparison


def run_discovery(model_filter: Optional[str] = None, samples_per_benchmark: int = PAIR_GENERATORS_DEFAULT_N, with_nonsense_baseline: bool = False, with_pairs_ablation: bool = False):
    """Run full category direction discovery."""
    print("=" * SEPARATOR_WIDTH_WIDE)
    print("CATEGORY DIRECTION DISCOVERY")
    print("=" * SEPARATOR_WIDTH_WIDE)
    
    # Load categories
    categories = load_categorized_benchmarks()
    category_info = load_category_directions()
    
    print(f"Categories: {list(categories.keys())}")
    print(f"Total benchmarks: {sum(len(b) for b in categories.values())}")
    
    # Get search space config
    search_space = GeometrySearchSpace()
    search_space.config.pairs_per_benchmark = samples_per_benchmark
    
    # Filter models if specified
    if model_filter:
        models_to_test = [model_filter]
    else:
        models_to_test = search_space.models
    
    print(f"\nModels to test: {models_to_test}")
    print(f"Strategies: {[s.value for s in search_space.strategies]}")
    print(f"Pairs per benchmark: {search_space.config.pairs_per_benchmark}")
    
    # Output directory
    output_dir = Path("/tmp/direction_discovery")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_model_results = {}
    
    # Run for each model
    for model_name in models_to_test:
        model_results = run_discovery_for_model(model_name, output_dir, with_nonsense_baseline, with_pairs_ablation)
        if model_results:
            all_model_results[model_name] = model_results
    
    # Save overall summary (only if running all models)
    if not model_filter and all_model_results:
        overall_file = output_dir / "discovery_summary.json"
        overall = {
            "models": list(all_model_results.keys()),
            "categories": list(categories.keys()),
            "results": {}
        }
        for model_name, results in all_model_results.items():
            overall["results"][model_name] = {
                cat: {
                    "has_unified_direction": r.has_unified_direction,
                    "dominant_structure": r.dominant_structure,
                    "recommendation": r.recommendation,
                    "avg_linear_score": r.avg_linear_score,
                    # NEW: Add optimal config summary
                    "optimal_config": {
                        "layer": r.optimal_config.optimal_layer if r.optimal_config else None,
                        "strategy": r.optimal_config.optimal_strategy if r.optimal_config else None,
                        "min_pairs": r.optimal_config.min_pairs_for_stable_signal if r.optimal_config else None,
                        "k_directions": r.optimal_config.num_directions_needed if r.optimal_config else None,
                        "is_linear": r.optimal_config.is_linear if r.optimal_config else None,
                        "verdict": r.optimal_config.verdict if r.optimal_config else None,
                    } if r.optimal_config else None,
                }
                for cat, r in results.categories.items()
            }
        
        # Add cross-model comparison
        overall["cross_model_comparison"] = generate_cross_model_comparison(all_model_results)
        
        with open(overall_file, "w") as f:
            json.dump(overall, f, indent=JSON_INDENT)
    
    print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
    print("DISCOVERY COMPLETE")
    print("=" * SEPARATOR_WIDTH_WIDE)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover unified directions for skill categories")
    parser.add_argument("--model", type=str, default=None, help="Specific model to test (for parallel execution)")
    parser.add_argument("--samples-per-benchmark", type=int, required=True, help="Number of samples per benchmark")
    parser.add_argument("--with-nonsense-baseline", action="store_true", 
                        help="Compare against random token baseline (requires generating activations through the model)")
    parser.add_argument("--with-pairs-ablation", action="store_true",
                        help="Run ablation on number of pairs to find minimum needed for stable signal")
    parser.add_argument("--full-diagnosis", action="store_true",
                        help="Run full diagnosis (enables both --with-nonsense-baseline and --with-pairs-ablation)")
    args = parser.parse_args()
    
    # --full-diagnosis enables both
    with_nonsense = args.with_nonsense_baseline or args.full_diagnosis
    with_pairs = args.with_pairs_ablation or args.full_diagnosis
    
    run_discovery(
        model_filter=args.model, 
        samples_per_benchmark=args.samples_per_benchmark,
        with_nonsense_baseline=with_nonsense,
        with_pairs_ablation=with_pairs,
    )
