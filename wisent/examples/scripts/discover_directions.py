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
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

S3_BUCKET = "wisent-bucket"
S3_PREFIX = "direction_discovery"


def s3_sync_download(model_name: str, output_dir: Path) -> None:
    """Download existing results from S3."""
    model_prefix = model_name.replace('/', '_')
    s3_path = f"s3://{S3_BUCKET}/{S3_PREFIX}/{model_prefix}/"
    try:
        subprocess.run(
            ["aws", "s3", "sync", s3_path, str(output_dir), "--quiet"],
            check=False,
            capture_output=True,
        )
        print(f"Synced existing results from S3: {s3_path}")
    except Exception as e:
        print(f"S3 download skipped: {e}")


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
    
    # Step 1: Signal detection
    avg_signal_strength: float  # MLP CV accuracy
    signal_exists: bool  # avg_signal_strength > 0.6
    
    # Step 2: Linearity check  
    avg_linear_probe_accuracy: float  # Linear probe CV accuracy
    is_linear: bool  # signal is linear (CAA will work)
    
    # NEW: Nonlinear signal metrics
    avg_knn_accuracy_k10: float  # k-NN CV accuracy
    avg_mmd_rbf: float  # Maximum Mean Discrepancy
    avg_local_dim_pos: float  # Local intrinsic dim of positive class
    avg_local_dim_neg: float  # Local intrinsic dim of negative class
    avg_fisher_max: float  # Max Fisher ratio
    avg_density_ratio: float  # Density ratio
    
    # Step 3: Geometry details (only meaningful if signal_exists)
    structure_distribution: Dict[str, int]
    structure_percentages: Dict[str, float]
    dominant_structure: str
    avg_linear_score: float
    avg_cohens_d: float
    
    # Final recommendation
    recommendation: str  # NO_SIGNAL, CAA, or NONLINEAR
    has_unified_direction: bool
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
        
        # Group by recommendation
        caa_ready = []  # Has signal AND linear
        nonlinear = []  # Has signal but NOT linear
        no_signal = []  # No signal
        
        for name, cat in self.categories.items():
            if not cat.signal_exists:
                no_signal.append(name)
            elif cat.is_linear:
                caa_ready.append(name)
            else:
                nonlinear.append(name)
        
        if caa_ready:
            lines.append(f"CAA READY - Linear signal ({len(caa_ready)}):")
            for name in sorted(caa_ready, key=lambda n: self.categories[n].avg_signal_strength, reverse=True):
                cat = self.categories[name]
                lines.append(f"  {name}: signal={cat.avg_signal_strength:.2f}, linear={cat.avg_linear_probe_accuracy:.2f}, kNN={cat.avg_knn_accuracy_k10:.2f}")
        
        if nonlinear:
            lines.append(f"\nNONLINEAR - Need different method ({len(nonlinear)}):")
            for name in nonlinear:
                cat = self.categories[name]
                lines.append(f"  {name}: signal={cat.avg_signal_strength:.2f}, linear={cat.avg_linear_probe_accuracy:.2f}, kNN={cat.avg_knn_accuracy_k10:.2f}, MMD={cat.avg_mmd_rbf:.3f}")
        
        if no_signal:
            lines.append(f"\nNO SIGNAL ({len(no_signal)}):")
            for name in no_signal:
                cat = self.categories[name]
                lines.append(f"  {name}: signal={cat.avg_signal_strength:.2f}, kNN={cat.avg_knn_accuracy_k10:.2f}")
        
        return "\n".join(lines)


def analyze_category_results(results: GeometrySearchResults, category: str, description: str, benchmarks: List[str]) -> CategoryResult:
    """Analyze geometry results for a category."""
    if not results.results:
        return CategoryResult(
            category=category,
            description=description,
            benchmarks_tested=benchmarks,
            total_tests=0,
            avg_signal_strength=0.5,
            signal_exists=False,
            avg_linear_probe_accuracy=0.5,
            is_linear=False,
            avg_knn_accuracy_k10=0.5,
            avg_mmd_rbf=0.0,
            avg_local_dim_pos=0.0,
            avg_local_dim_neg=0.0,
            avg_fisher_max=0.0,
            avg_density_ratio=1.0,
            structure_distribution={},
            structure_percentages={},
            dominant_structure="error",
            avg_linear_score=0.0,
            avg_cohens_d=0.0,
            recommendation="NO_RESULTS",
            has_unified_direction=False,
        )
    
    dist = results.get_structure_distribution()
    total = sum(dist.values())
    
    percentages = {k: 100 * v / total for k, v in dist.items()} if total > 0 else {}
    
    # Determine dominant structure
    dominant = max(dist.items(), key=lambda x: x[1])[0] if dist else "unknown"
    
    # Step 1: Signal detection (MLP CV accuracy)
    avg_signal_strength = sum(r.signal_strength for r in results.results) / len(results.results)
    signal_exists = avg_signal_strength > 0.6
    
    # Step 2: Linearity check (Linear probe CV accuracy)
    avg_linear_probe_accuracy = sum(r.linear_probe_accuracy for r in results.results) / len(results.results)
    # Signal is linear if linear probe is close to MLP accuracy
    is_linear = signal_exists and avg_linear_probe_accuracy > 0.6 and (avg_signal_strength - avg_linear_probe_accuracy) < 0.15
    
    # Step 2b: Nonlinear signal metrics
    avg_knn_accuracy_k10 = sum(r.knn_accuracy_k10 for r in results.results) / len(results.results)
    avg_mmd_rbf = sum(r.mmd_rbf for r in results.results) / len(results.results)
    avg_local_dim_pos = sum(r.local_dim_pos for r in results.results) / len(results.results)
    avg_local_dim_neg = sum(r.local_dim_neg for r in results.results) / len(results.results)
    avg_fisher_max = sum(r.fisher_max for r in results.results) / len(results.results)
    avg_density_ratio = sum(r.density_ratio for r in results.results) / len(results.results)
    
    # Step 3: Geometry details
    avg_linear_score = sum(r.linear_score for r in results.results) / len(results.results)
    avg_cohens_d = sum(r.cohens_d for r in results.results) / len(results.results)
    
    # Final recommendation
    if not signal_exists:
        recommendation = "NO_SIGNAL"
    elif is_linear:
        recommendation = "CAA"
    else:
        recommendation = "NONLINEAR"
    
    # Unified direction exists if we have linear signal
    has_unified = is_linear
    
    # Best config - prefer high signal_strength
    best = sorted(results.results, key=lambda r: r.signal_strength, reverse=True)[:1]
    best_config = None
    if best:
        b = best[0]
        best_config = {
            "benchmark": b.benchmark,
            "strategy": b.strategy,
            "layers": b.layers,
            "signal_strength": b.signal_strength,
            "linear_probe_accuracy": b.linear_probe_accuracy,
            "is_linear": b.is_linear,
        }
    
    return CategoryResult(
        category=category,
        description=description,
        benchmarks_tested=benchmarks,
        total_tests=total,
        avg_signal_strength=avg_signal_strength,
        signal_exists=signal_exists,
        avg_linear_probe_accuracy=avg_linear_probe_accuracy,
        is_linear=is_linear,
        avg_knn_accuracy_k10=avg_knn_accuracy_k10,
        avg_mmd_rbf=avg_mmd_rbf,
        avg_local_dim_pos=avg_local_dim_pos,
        avg_local_dim_neg=avg_local_dim_neg,
        avg_fisher_max=avg_fisher_max,
        avg_density_ratio=avg_density_ratio,
        structure_distribution=dist,
        structure_percentages=percentages,
        dominant_structure=dominant,
        avg_linear_score=avg_linear_score,
        avg_cohens_d=avg_cohens_d,
        recommendation=recommendation,
        has_unified_direction=has_unified,
        best_config=best_config,
    )


def run_discovery_for_model(model_name: str, output_dir: Path):
    """Run discovery for a single model with resume support."""
    categories = load_categorized_benchmarks()
    category_info = load_category_directions()
    search_space = GeometrySearchSpace()
    
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name}")
    print("=" * 70)
    
    # Download existing results from S3 for resume
    s3_sync_download(model_name, output_dir)
    
    # Check which categories are already done
    model_prefix = model_name.replace('/', '_')
    completed_categories = set()
    for cat_name in categories.keys():
        cat_file = output_dir / f"{model_prefix}_{cat_name}.json"
        if cat_file.exists() and cat_file.stat().st_size > 100:
            completed_categories.add(cat_name)
            print(f"  [SKIP] {cat_name} already completed")
    
    remaining = [c for c in categories.keys() if c not in completed_categories]
    if not remaining:
        print("All categories already completed!")
        return None
    
    print(f"\nCompleted: {len(completed_categories)}/15, Remaining: {len(remaining)}")
    print(f"Categories to run: {remaining}")
    
    try:
        model = WisentModel(model_name, device="cuda")
        print(f"Loaded: {model.num_layers} layers, hidden={model.hidden_size}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    
    cache_dir = f"/tmp/wisent_direction_cache_{model_prefix}"
    
    model_results = DiscoveryResults(model=model_name)
    
    # Run for each remaining category
    for cat_name in remaining:
        benchmarks = categories[cat_name]
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
            
            print(f"\n  Step 1 - Signal: {cat_result.avg_signal_strength:.3f} ({'EXISTS' if cat_result.signal_exists else 'NONE'})")
            print(f"  Step 2 - Linear: {cat_result.avg_linear_probe_accuracy:.3f} ({'YES' if cat_result.is_linear else 'NO'})")
            print(f"  Recommendation: {cat_result.recommendation}")
            
            # Save per-category results immediately
            cat_file = output_dir / f"{model_prefix}_{cat_name}.json"
            results.save(str(cat_file))
            print(f"  Saved: {cat_file}")
            
            # Upload to S3 immediately for durability
            s3_upload_file(cat_file, model_name)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Save/update model summary (merge with existing if any)
    summary_file = output_dir / f"{model_prefix}_summary.json"
    
    # Load existing summary if present
    existing_categories = {}
    if summary_file.exists():
        with open(summary_file) as f:
            existing = json.load(f)
            existing_categories = existing.get("categories", {})
    
    # Merge new results
    all_categories = {**existing_categories, **{k: asdict(v) for k, v in model_results.categories.items()}}
    
    with open(summary_file, "w") as f:
        json.dump({
            "model": model_name,
            "categories": all_categories
        }, f, indent=2)
    
    # Upload summary to S3
    s3_upload_file(summary_file, model_name)
    
    print(f"\n{model_results.summary()}")
    
    # Cleanup model
    del model
    
    return model_results


def run_discovery(model_filter: Optional[str] = None, samples_per_benchmark: int = 50):
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
    parser.add_argument("--samples-per-benchmark", type=int, default=50, help="Number of samples per benchmark (default: 50)")
    args = parser.parse_args()
    
    run_discovery(model_filter=args.model, samples_per_benchmark=args.samples_per_benchmark)
