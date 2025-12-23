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
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

from wisent.core.geometry_search_space import (
    GeometrySearchSpace,
    GeometrySearchConfig,
)
from wisent.core.geometry_runner import (
    GeometryRunner,
    GeometrySearchResults,
    GeometryTestResult,
)
from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
    GeometryAnalysisConfig,
    StructureType,
)
from wisent.core.models.wisent_model import WisentModel


def load_categorized_benchmarks() -> Dict[str, List[str]]:
    """Load benchmarks grouped by category."""
    params_dir = Path(__file__).parent.parent.parent / "parameters" / "lm_eval"
    with open(params_dir / "working_benchmarks_categorized.json") as f:
        return json.load(f)


def load_category_directions() -> Dict[str, Dict]:
    """Load hypothesized directions for each category."""
    params_dir = Path(__file__).parent.parent.parent / "parameters" / "lm_eval"
    with open(params_dir / "category_directions.json") as f:
        return json.load(f)


@dataclass
class CategoryResult:
    """Result for a single category."""
    category: str
    description: str
    benchmarks_tested: List[str]
    total_tests: int
    structure_distribution: Dict[str, int]
    structure_percentages: Dict[str, float]
    dominant_structure: str
    has_unified_direction: bool
    recommendation: str
    avg_linear_score: float
    avg_cohens_d: float
    best_config: Optional[Dict[str, Any]] = None


@dataclass 
class DiscoveryResults:
    """Results from full discovery run."""
    model: str
    categories: Dict[str, CategoryResult] = field(default_factory=dict)
    
    def summary(self) -> str:
        lines = [
            f"Model: {self.model}",
            f"Categories analyzed: {len(self.categories)}",
            "",
        ]
        
        unified = []
        cone = []
        mixed = []
        
        for name, cat in self.categories.items():
            if cat.has_unified_direction:
                unified.append(name)
            elif cat.dominant_structure == "cone":
                cone.append(name)
            else:
                mixed.append(name)
        
        if unified:
            lines.append(f"Unified direction exists ({len(unified)}):")
            for name in unified:
                cat = self.categories[name]
                lines.append(f"  {name}: {cat.structure_percentages.get('linear', 0):.1f}% linear")
        
        if cone:
            lines.append(f"\nCone structure ({len(cone)}):")
            for name in cone:
                cat = self.categories[name]
                lines.append(f"  {name}: {cat.structure_percentages.get('cone', 0):.1f}% cone")
        
        if mixed:
            lines.append(f"\nMixed/other ({len(mixed)}):")
            for name in mixed:
                cat = self.categories[name]
                lines.append(f"  {name}: {cat.dominant_structure}")
        
        return "\n".join(lines)


def analyze_category_results(results: GeometrySearchResults, category: str, description: str, benchmarks: List[str]) -> CategoryResult:
    """Analyze geometry results for a category."""
    if not results.results:
        return CategoryResult(
            category=category,
            description=description,
            benchmarks_tested=benchmarks,
            total_tests=0,
            structure_distribution={},
            structure_percentages={},
            dominant_structure="error",
            has_unified_direction=False,
            recommendation="No results",
            avg_linear_score=0.0,
            avg_cohens_d=0.0,
        )
    
    dist = results.get_structure_distribution()
    total = sum(dist.values())
    
    percentages = {k: 100 * v / total for k, v in dist.items()} if total > 0 else {}
    
    linear_pct = percentages.get("linear", 0)
    cone_pct = percentages.get("cone", 0)
    orthogonal_pct = percentages.get("orthogonal", 0)
    
    # Determine dominant structure
    dominant = max(dist.items(), key=lambda x: x[1])[0] if dist else "unknown"
    
    # Determine if unified direction exists
    has_unified = linear_pct > 50
    
    # Recommendation
    if has_unified:
        recommendation = "CAA"
    elif cone_pct > 30:
        recommendation = "PRISM"
    elif orthogonal_pct > 50:
        recommendation = "TITAN"
    else:
        recommendation = "TITAN"
    
    # Best config
    best = results.get_best_by_linear_score(1)
    best_config = None
    if best:
        b = best[0]
        best_config = {
            "benchmark": b.benchmark,
            "strategy": b.strategy,
            "layers": b.layers,
            "linear_score": b.linear_score,
            "cohens_d": b.cohens_d,
        }
    
    # Averages
    avg_linear = sum(r.linear_score for r in results.results) / len(results.results)
    avg_cohens_d = sum(r.cohens_d for r in results.results) / len(results.results)
    
    return CategoryResult(
        category=category,
        description=description,
        benchmarks_tested=benchmarks,
        total_tests=total,
        structure_distribution=dist,
        structure_percentages=percentages,
        dominant_structure=dominant,
        has_unified_direction=has_unified,
        recommendation=recommendation,
        avg_linear_score=avg_linear,
        avg_cohens_d=avg_cohens_d,
        best_config=best_config,
    )


def run_discovery_for_model(model_name: str, output_dir: Path):
    """Run discovery for a single model."""
    categories = load_categorized_benchmarks()
    category_info = load_category_directions()
    search_space = GeometrySearchSpace()
    
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name}")
    print("=" * 70)
    
    try:
        model = WisentModel(model_name, device="cuda")
        print(f"Loaded: {model.num_layers} layers, hidden={model.hidden_size}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    
    cache_dir = f"/tmp/wisent_direction_cache_{model_name.replace('/', '_')}"
    
    model_results = DiscoveryResults(model=model_name)
    
    # Run for each category
    for cat_name, benchmarks in categories.items():
        print(f"\n{'-' * 50}")
        print(f"Category: {cat_name.upper()} ({len(benchmarks)} benchmarks)")
        print("-" * 50)
        
        info = category_info.get(cat_name, {})
        description = info.get("description", "")
        print(f"Description: {description}")
        
        # Create search space for this category
        cat_config = GeometrySearchConfig(
            pairs_per_benchmark=search_space.config.pairs_per_benchmark,
            max_layer_combo_size=search_space.config.max_layer_combo_size,
            cache_dir=cache_dir,
        )
        
        cat_space = GeometrySearchSpace(
            models=[model_name],
            strategies=search_space.strategies,
            benchmarks=benchmarks,
            config=cat_config,
        )
        
        # Run geometry search
        runner = GeometryRunner(cat_space, model, cache_dir=cache_dir)
        
        try:
            results = runner.run(show_progress=True)
            cat_result = analyze_category_results(results, cat_name, description, benchmarks)
            model_results.categories[cat_name] = cat_result
            
            print(f"\n  Structure: {cat_result.dominant_structure}")
            print(f"  Unified direction: {cat_result.has_unified_direction}")
            print(f"  Recommendation: {cat_result.recommendation}")
            print(f"  Avg linear score: {cat_result.avg_linear_score:.3f}")
            
            # Save per-category results
            cat_file = output_dir / f"{model_name.replace('/', '_')}_{cat_name}.json"
            results.save(str(cat_file))
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Save model summary
    summary_file = output_dir / f"{model_name.replace('/', '_')}_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "model": model_name,
            "categories": {k: asdict(v) for k, v in model_results.categories.items()}
        }, f, indent=2)
    
    print(f"\n{model_results.summary()}")
    
    # Cleanup model
    del model
    
    return model_results


def run_discovery(model_filter: Optional[str] = None):
    """Run full category direction discovery."""
    print("=" * 70)
    print("CATEGORY DIRECTION DISCOVERY")
    print("=" * 70)
    
    # Load categories
    categories = load_categorized_benchmarks()
    category_info = load_category_directions()
    
    print(f"Categories: {list(categories.keys())}")
    print(f"Total benchmarks: {sum(len(b) for b in categories.values())}")
    
    # Get search space config
    search_space = GeometrySearchSpace()
    
    # Filter models if specified
    if model_filter:
        models_to_test = [model_filter]
    else:
        models_to_test = search_space.models
    
    print(f"\nModels to test: {models_to_test}")
    print(f"Strategies: {[s.value for s in search_space.strategies]}")
    print(f"Pairs per benchmark: {search_space.config.pairs_per_benchmark}")
    
    # Output directory
    output_dir = Path("/home/ubuntu/output/direction_discovery")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_model_results = {}
    
    # Run for each model
    for model_name in models_to_test:
        model_results = run_discovery_for_model(model_name, output_dir)
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
                }
                for cat, r in results.categories.items()
            }
        
        with open(overall_file, "w") as f:
            json.dump(overall, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print("DISCOVERY COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover unified directions for skill categories")
    parser.add_argument("--model", type=str, default=None, help="Specific model to test (for parallel execution)")
    args = parser.parse_args()
    
    run_discovery(model_filter=args.model)
