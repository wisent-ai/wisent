"""
Per-model direction discovery runner.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

from wisent.core.reading.modules import (
    GeometrySearchSpace,
    GeometrySearchConfig,
)
from wisent.core.reading.modules.runner.geometry_runner import (
    GeometryRunner,
    GeometrySearchResults,
    analyze_with_nonsense_baseline,
)
from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
from wisent.core.utils.config_tools.constants import PAIR_COUNT_ABLATION_SERIES, SEPARATOR_WIDTH_WIDE, SEPARATOR_WIDTH_MEDIUM, JSON_INDENT
from wisent.core.primitives.models.wisent_model import WisentModel
from wisent.examples.scripts._discovery_utils import (
    gcs_sync_download,
    gcs_upload_file,
    load_categorized_benchmarks,
    load_category_directions,
    DiscoveryResults,
)
from wisent.examples.scripts._pairs_ablation import run_pairs_ablation
from wisent.examples.scripts._category_analysis import (
    analyze_category_results,
)


def run_discovery_for_model(model_name: str, output_dir: Path, with_nonsense_baseline: bool = False, with_pairs_ablation: bool = False):
    """Run discovery for a single model with resume support."""
    categories = load_categorized_benchmarks()
    category_info = load_category_directions()
    search_space = GeometrySearchSpace()
    
    print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
    print(f"MODEL: {model_name}")
    print("=" * SEPARATOR_WIDTH_WIDE)
    
    # Download existing results from GCS for resume
    gcs_sync_download(model_name, output_dir)
    
    # Check which categories need work
    model_prefix = model_name.replace('/', '_')
    completed_categories = set()
    needs_diagnosis = set()  # Has results but missing optimal_config
    
    for cat_name in categories.keys():
        cat_file = output_dir / f"{model_prefix}_{cat_name}.json"
        if cat_file.exists() and cat_file.stat().st_size > 100:
            # Check if it has optimal_config (full diagnosis)
            if with_nonsense_baseline or with_pairs_ablation:
                try:
                    with open(cat_file) as f:
                        existing = json.load(f)
                    # Check if optimal_config exists and has real data
                    has_diagnosis = existing.get("optimal_config") is not None
                    if not has_diagnosis:
                        needs_diagnosis.add(cat_name)
                        print(f"  [UPGRADE] {cat_name} needs full diagnosis")
                        continue
                except Exception:
                    pass
            completed_categories.add(cat_name)
            print(f"  [SKIP] {cat_name} already completed")
    
    remaining = [c for c in categories.keys() if c not in completed_categories]
    if not remaining:
        print("All categories already completed!")
        return None
    
    print(f"\nCompleted: {len(completed_categories)}/{len(categories)}, Remaining: {len(remaining)}")
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
        print(f"\n{'-' * SEPARATOR_WIDTH_MEDIUM}")
        print(f"Category: {cat_name.upper()} ({len(benchmarks)} benchmarks)")
        print("-" * SEPARATOR_WIDTH_MEDIUM)
        
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
            # Check if we can load existing results (upgrade mode)
            cat_file = output_dir / f"{model_prefix}_{cat_name}.json"
            if cat_name in needs_diagnosis and cat_file.exists():
                print(f"  [UPGRADE MODE] Loading existing results, adding diagnosis...")
                results = GeometrySearchResults.load(str(cat_file))
            else:
                results = runner.run(show_progress=True)
            
            # Optionally compute nonsense baseline for each benchmark
            nonsense_analysis = {}
            if with_nonsense_baseline:
                print(f"  Computing nonsense baseline...")
                for benchmark in benchmarks:
                    # Get cached activations for this benchmark
                    try:
                        cached = runner._get_cached_activations(
                            benchmark, 
                            ExtractionStrategy.CHAT_LAST,
                            show_progress=False
                        )
                        # Use middle layer for analysis
                        mid_layer = model.num_layers // 2
                        layer_name = str(mid_layer)
                        if layer_name in cached.get_available_layers():
                            pos_acts = cached.get_positive_activations(mid_layer)
                            neg_acts = cached.get_negative_activations(mid_layer)
                            n_pairs = min(len(pos_acts), len(neg_acts))
                            
                            # Generate nonsense baseline with SAME number of pairs
                            nonsense_pos, nonsense_neg = runner.get_nonsense_baseline(
                                n_pairs=n_pairs,
                                layer=mid_layer,
                            )
                            
                            # Analyze
                            analysis = analyze_with_nonsense_baseline(
                                pos_acts[:n_pairs], neg_acts[:n_pairs],
                                nonsense_pos, nonsense_neg,
                                benchmark
                            )
                            nonsense_analysis[benchmark] = analysis
                            print(f"    {benchmark}: {analysis['verdict']} (ICD={analysis['real_icd']['icd']:.1f}, acc={analysis['baseline_comparison']['real_accuracy']:.0%})")
                    except Exception as e:
                        print(f"    {benchmark}: SKIP ({e})")
                        nonsense_analysis[benchmark] = None
            
            # Optionally run pairs ablation
            pairs_ablation = {}
            if with_pairs_ablation:
                print(f"  Running pairs ablation...")
                mid_layer = model.num_layers // 2
                for benchmark in benchmarks:
                    try:
                        ablation = run_pairs_ablation(
                            runner, benchmark, mid_layer, 
                            ExtractionStrategy.CHAT_LAST,
                            pair_counts=list(PAIR_COUNT_ABLATION_SERIES)
                        )
                        if ablation:
                            pairs_ablation[benchmark] = ablation
                            curve = ", ".join(f"{n}:{acc:.0%}" for n, acc in sorted(ablation.items()))
                            print(f"    {benchmark}: {curve}")
                    except Exception as e:
                        print(f"    {benchmark}: SKIP ({e})")
            
            cat_result = analyze_category_results(
                results, cat_name, description, benchmarks,
                nonsense_analysis=nonsense_analysis if with_nonsense_baseline else None,
                pairs_ablation=pairs_ablation if with_pairs_ablation else None,
            )
            model_results.categories[cat_name] = cat_result
            
            print(f"\n  Step 1 - Signal: {cat_result.avg_signal_strength:.3f} ({'EXISTS' if cat_result.signal_exists else 'NONE'})")
            print(f"  Step 2 - Linear: {cat_result.avg_linear_probe_accuracy:.3f} ({'YES' if cat_result.is_linear else 'NO'})")
            if with_nonsense_baseline:
                print(f"  Step 3 - Nonsense baseline: ICD={cat_result.avg_icd:.1f}, verdict={cat_result.signal_verdict}")
            if cat_result.optimal_config:
                oc = cat_result.optimal_config
                print(f"  Optimal config: layer={oc.optimal_layer}, strategy={oc.optimal_strategy}, min_pairs={oc.min_pairs_for_stable_signal}, k_directions={oc.num_directions_needed}")
            print(f"  Recommendation: {cat_result.recommendation}")
            
            # Save per-category results immediately (cat_file defined earlier)
            results.save(str(cat_file))
            print(f"  Saved: {cat_file}")
            
            # Upload to GCS immediately for durability
            gcs_upload_file(cat_file, model_name)
            
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
        }, f, indent=JSON_INDENT)
    
    # Upload summary to GCS
    gcs_upload_file(summary_file, model_name)
    
    print(f"\n{model_results.summary()}")
    
    # Cleanup model
    del model
    
    return model_results

