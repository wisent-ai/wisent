"""Check linearity of a representation from contrastive pairs."""

import json
import sys


def execute_check_linearity(args):
    """Execute the check-linearity command."""
    print(f"\n{'='*60}")
    print("LINEARITY CHECK")
    print(f"{'='*60}")
    print(f"Pairs file: {args.pairs_file}")
    print(f"Model: {args.model}")
    
    # Load pairs
    with open(args.pairs_file, 'r') as f:
        data = json.load(f)
    
    pairs_data = data.get('pairs', [])
    if not pairs_data:
        print("Error: No pairs found in file")
        sys.exit(1)
    
    print(f"Loaded {len(pairs_data)} pairs")
    
    # Import dependencies
    from wisent.core.contrastive_pairs.diagnostics import (
        check_linearity,
        LinearityConfig,
    )
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse
    from wisent.core.activations.extraction_strategy import ExtractionStrategy
    
    # Build ContrastivePair objects
    pairs = []
    for p in pairs_data:
        try:
            pairs.append(ContrastivePair(
                prompt=p.get('prompt', ''),
                positive_response=PositiveResponse(
                    model_response=p['positive_response']['model_response']
                ),
                negative_response=NegativeResponse(
                    model_response=p['negative_response']['model_response']
                ),
            ))
        except (KeyError, TypeError) as e:
            if args.verbose:
                print(f"Skipping malformed pair: {e}")
            continue
    
    if len(pairs) < 10:
        print(f"Error: Need at least 10 valid pairs, got {len(pairs)}")
        sys.exit(1)
    
    print(f"Using {min(len(pairs), args.max_pairs)} pairs for analysis")
    
    # Load model
    print(f"\nLoading model {args.model}...")
    model = WisentModel(args.model, device=args.device)
    
    # Configure linearity check
    config = LinearityConfig(
        linear_threshold=args.linear_threshold,
        weak_threshold=args.weak_threshold,
        min_cohens_d=args.min_cohens_d,
        max_pairs=args.max_pairs,
        geometry_optimization_steps=args.optimization_steps,
    )
    
    if args.layers:
        config.layers_to_test = [int(l) for l in args.layers.split(',')]
    
    if args.extraction_strategy:
        config.extraction_strategies = [ExtractionStrategy(args.extraction_strategy)]
        print(f"Using extraction strategy: {args.extraction_strategy}")
    
    # Run check
    print("\nRunning linearity check...")
    result = check_linearity(pairs, model, config)
    
    # Output results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Verdict: {result.verdict.value.upper()}")
    print(f"Best linear score: {result.best_linear_score:.3f}")
    print(f"Best layer: {result.best_layer}")
    print(f"Cohen's d: {result.cohens_d:.2f}")
    print(f"Variance explained: {result.variance_explained:.3f}")
    print(f"Configurations tested: {len(result.all_results)}")
    
    print(f"\nBest configuration:")
    for k, v in result.best_config.items():
        print(f"  {k}: {v}")
    
    print(f"\nRecommendation: {result.recommendation}")
    
    # Save results if output specified
    if args.output:
        output_data = result.to_dict()
        output_data["all_results"] = result.all_results
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Verbose output
    if args.verbose:
        print(f"\n{'='*60}")
        print("ALL CONFIGURATIONS (sorted by linear score)")
        print(f"{'='*60}")
        
        sorted_results = sorted(result.all_results, key=lambda x: x['linear_score'], reverse=True)
        
        print(f"{'Linear':<8} {'d':<8} {'Layer':<6} {'Strategy':<20} {'Structure':<12} {'Norm'}")
        print("-" * 70)
        
        for r in sorted_results[:20]:
            print(f"{r['linear_score']:<8.3f} {r['cohens_d']:<8.2f} {r['layer']:<6} "
                  f"{r['extraction_strategy']:<20} {r['best_structure']:<12} {r['normalize']}")
        
        # Show best result for each structure type
        if sorted_results and 'all_structure_scores' in sorted_results[0]:
            print(f"\n{'='*60}")
            print("BEST SCORE PER STRUCTURE TYPE")
            print(f"{'='*60}")
            
            # Collect best score for each structure across all configs
            best_per_structure = {}
            for r in result.all_results:
                if 'all_structure_scores' not in r:
                    continue
                for struct_name, data in r['all_structure_scores'].items():
                    score = data['score']
                    if struct_name not in best_per_structure or score > best_per_structure[struct_name]['score']:
                        best_per_structure[struct_name] = {
                            'score': score,
                            'confidence': data['confidence'],
                            'layer': r['layer'],
                            'strategy': r['extraction_strategy'],
                        }
            
            print(f"{'Structure':<12} {'Score':<8} {'Conf':<8} {'Layer':<6} {'Strategy'}")
            print("-" * 55)
            sorted_structs = sorted(best_per_structure.items(), key=lambda x: x[1]['score'], reverse=True)
            for name, data in sorted_structs:
                print(f"{name:<12} {data['score']:<8.3f} {data['confidence']:<8.3f} {data['layer']:<6} {data['strategy']}")
    
    # Exit code based on verdict
    if result.verdict.value == "linear":
        sys.exit(0)
    elif result.verdict.value == "weakly_linear":
        sys.exit(0)
    else:
        sys.exit(1)
