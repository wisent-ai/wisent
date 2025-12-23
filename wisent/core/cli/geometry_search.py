"""Run geometry search across benchmarks to find unified goodness direction."""

import json
import sys
import os
from pathlib import Path


def execute_geometry_search(args):
    """Execute the geometry-search command."""
    print(f"\n{'='*60}")
    print("GEOMETRY SEARCH")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Pairs per benchmark: {args.pairs_per_benchmark}")
    print(f"Max layer combo size: {args.max_layer_combo_size}")
    
    # Import dependencies
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.geometry_search_space import GeometrySearchSpace, GeometrySearchConfig
    from wisent.core.geometry_runner import GeometryRunner
    from wisent.core.activations.extraction_strategy import ExtractionStrategy
    
    # Parse strategies
    if args.strategies:
        strategy_names = [s.strip() for s in args.strategies.split(',')]
        strategies = [ExtractionStrategy(s) for s in strategy_names]
        print(f"Strategies: {strategy_names}")
    else:
        strategies = None  # Use default (all 7)
        print("Strategies: all 7 default strategies")
    
    # Parse benchmarks
    if args.benchmarks:
        if args.benchmarks.endswith('.txt'):
            with open(args.benchmarks) as f:
                benchmarks = [line.strip() for line in f if line.strip()]
        else:
            benchmarks = [b.strip() for b in args.benchmarks.split(',')]
        print(f"Benchmarks: {len(benchmarks)} specified")
    else:
        benchmarks = None  # Use default (all)
        print("Benchmarks: all available")
    
    # Create config
    config = GeometrySearchConfig(
        pairs_per_benchmark=args.pairs_per_benchmark,
        max_layer_combo_size=args.max_layer_combo_size,
        random_seed=args.seed,
        cache_activations=True,
        cache_dir=args.cache_dir,
    )
    
    # Create search space
    search_space = GeometrySearchSpace(
        models=[args.model],
        strategies=strategies,
        benchmarks=benchmarks,
        config=config,
    )
    
    print(f"\n{search_space.summary()}")
    
    # Load model
    print(f"\nLoading model {args.model}...")
    model = WisentModel(args.model, device=args.device)
    print(f"Model loaded: {model.num_layers} layers, hidden_size={model.hidden_size}")
    
    # Create runner
    cache_dir = args.cache_dir or f"/tmp/wisent_geometry_cache_{args.model.replace('/', '_')}"
    runner = GeometryRunner(search_space, model, cache_dir=cache_dir)
    
    # Run search
    print(f"\nStarting geometry search...")
    results = runner.run(show_progress=True)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.save(str(output_path))
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {results.total_time_seconds / 3600:.2f} hours")
    print(f"  Extraction: {results.extraction_time_seconds / 3600:.2f} hours")
    print(f"  Testing: {results.test_time_seconds / 60:.1f} minutes")
    print(f"Benchmarks tested: {results.benchmarks_tested}")
    print(f"Strategies tested: {results.strategies_tested}")
    print(f"Layer combos tested: {results.layer_combos_tested}")
    
    print(f"\nStructure distribution:")
    for struct, count in sorted(results.get_structure_distribution().items(), key=lambda x: -x[1]):
        pct = 100 * count / results.layer_combos_tested
        print(f"  {struct}: {count} ({pct:.1f}%)")
    
    print(f"\nTop 10 by linear score:")
    for r in results.get_best_by_linear_score(10):
        print(f"  {r.benchmark}/{r.strategy} layers={r.layers}: linear={r.linear_score:.3f} best={r.best_structure}")
    
    print(f"\nTop 10 by cone score:")
    for r in results.get_best_by_structure('cone', 10):
        print(f"  {r.benchmark}/{r.strategy} layers={r.layers}: cone={r.cone_score:.3f} best={r.best_structure}")
    
    # Summary by benchmark
    print(f"\nSummary by benchmark (avg linear score):")
    by_bench = results.get_summary_by_benchmark()
    sorted_benches = sorted(by_bench.items(), key=lambda x: -x[1]['mean'])[:20]
    for bench, stats in sorted_benches:
        print(f"  {bench}: mean={stats['mean']:.3f} max={stats['max']:.3f}")
    
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    
    # Determine if unified direction exists
    dist = results.get_structure_distribution()
    total = sum(dist.values())
    linear_pct = 100 * dist.get('linear', 0) / total if total > 0 else 0
    cone_pct = 100 * dist.get('cone', 0) / total if total > 0 else 0
    orthogonal_pct = 100 * dist.get('orthogonal', 0) / total if total > 0 else 0
    
    if linear_pct > 50:
        print(f"UNIFIED LINEAR DIRECTION EXISTS ({linear_pct:.1f}% linear)")
        print("Recommendation: Use CAA with the best layer/strategy combination")
    elif cone_pct > 30:
        print(f"CONE STRUCTURE DETECTED ({cone_pct:.1f}% cone)")
        print("Recommendation: Use PRISM with multi-directional steering")
    elif orthogonal_pct > 50:
        print(f"ORTHOGONAL STRUCTURE ({orthogonal_pct:.1f}% orthogonal)")
        print("Recommendation: No unified direction - use per-benchmark directions or TITAN")
    else:
        print("MIXED STRUCTURE - no clear unified direction")
        print("Recommendation: Use TITAN for adaptive multi-component steering")
