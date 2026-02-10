"""RepScan CLI command for geometry analysis with concept decomposition."""

import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import json
import sys
from pathlib import Path


def execute_repscan(args):
    """Execute the repscan command."""
    print(f"\n{'='*60}")
    print("REPSCAN - Geometry Analysis with Concept Decomposition")
    print(f"{'='*60}")

    from wisent.core.geometry.repscan.repscan_with_concepts import run_repscan_with_concept_naming
    from .repscan_data_loaders import load_from_cache, load_from_json, load_from_database
    import torch

    # Load data from selected source
    if args.from_cache:
        activations_by_layer, pair_texts = load_from_cache(
            args.from_cache, task=args.task, limit=args.limit,
            database_url=args.database_url)
    elif getattr(args, 'from_json', None):
        activations_by_layer, pair_texts = load_from_json(args.from_json)
    elif args.from_database:
        activations_by_layer, pair_texts = load_from_database(args)
    else:
        print("ERROR: --from-database, --from-cache, or --from-json is required.")
        sys.exit(1)

    if not activations_by_layer:
        print("\nERROR: No activations loaded.")
        sys.exit(1)

    # Run repscan with specified steps
    print(f"\n{'='*60}")
    print(f"Running RepScan protocol (steps: {args.steps})")
    print(f"{'='*60}")

    results = run_repscan_with_concept_naming(
        activations_by_layer=activations_by_layer,
        pair_texts=pair_texts,
        generate_visualizations=args.visualizations,
        llm_model=args.llm_model,
        steps=args.steps,
    )

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nLayers: {results.get('n_layers')} (layers {results.get('layers_used', [])})")
    print(f"Total dimensions: {results.get('total_dims')}")
    print(f"Pairs analyzed: {results.get('n_pairs')}")
    rec_method = results.get('recommended_method')
    rec_conf = results.get('recommendation_confidence')
    if rec_method:
        print(f"\nRecommended method: {rec_method}")
        print(f"Confidence: {rec_conf:.2f}" if rec_conf is not None else "")

    # Print key metrics (only if geometry metrics were computed)
    metrics = results.get("metrics", {})
    if metrics:
        print(f"\n--- Key Metrics (all-layer concatenated) ---")
        for name, key in [("Signal strength", "signal_strength"), ("Linear probe", "linear_probe_accuracy"),
                           ("MLP probe", "mlp_probe_accuracy"), ("KNN", "knn_accuracy"), ("KNN PCA", "knn_pca_accuracy")]:
            print(f"{name}: {metrics.get(key, 0):.3f}")

    # Print per-layer editability profile
    editability_profile = results.get("editability_by_layer")
    if editability_profile:
        print(f"\n--- Per-Layer Editability Profile ---")
        print(f"  {'Layer':>5} {'Score':>6} {'Survival':>8} {'Decay':>8} {'PR':>6} {'Verdict'}")
        for layer in sorted(editability_profile.keys(), key=lambda x: int(x)):
            e = editability_profile[layer]
            print(f"  {layer:>5} {e['editability_score']:>6.3f} {e['steering_survival_ratio']:>8.3f}"
                  f" {e['spectral_decay_rate']:>8.4f} {e['participation_ratio']:>6.1f} {e['verdict']}")

    # Print overall (concatenated) editability analysis
    editability = results.get("editability_analysis")
    if editability:
        print(f"\n--- Overall Editability (concatenated) ---")
        print(f"Editability score: {editability.get('editability_score', 0):.3f}  |  Verdict: {editability['verdict']}")
        print(f"Editing capacity: {editability['editing_capacity']:.3f}")
        print(f"Steering survival: {editability['steering_survival_ratio']:.3f}")
        print(f"Spectral decay: {editability['spectral_decay_rate']:.4f}  |  PR: {editability.get('participation_ratio', 0):.1f}")
        for warning in editability.get("warnings", []):
            print(f"  WARNING: {warning}")

    # Print concept decomposition
    decomposition = results.get("concept_decomposition")
    if decomposition:
        n_concepts = decomposition.get("n_concepts", 1)
        print(f"\n--- Concept Decomposition ({n_concepts} concepts) ---")
        for concept in decomposition.get("concepts", []):
            print(f"\nConcept {concept['id']}: {concept.get('name', 'Unnamed')}")
            print(f"  Pairs: {concept.get('n_pairs', 0)}, Silhouette: {concept.get('silhouette_score', 0):.3f}")
            if 'optimal_layer' in concept:
                print(f"  Optimal layer: {concept['optimal_layer']} (acc: {concept.get('optimal_layer_accuracy', 0):.3f})")
            for pair in concept.get("representative_pairs", [])[:3]:
                label = pair.get("prompt", "")[:80] if isinstance(pair, dict) else f"pair index {pair}"
                print(f"    - {label}")

    # Save visualizations as PNG files
    if args.visualizations and decomposition:
        import base64
        viz_dir = Path(args.visualizations_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        visualizations = decomposition.get("visualizations", {})
        saved_count = 0
        for name, b64_data in visualizations.items():
            if b64_data and isinstance(b64_data, str):
                try:
                    with open(viz_dir / f"{name}.png", 'wb') as f:
                        f.write(base64.b64decode(b64_data))
                    saved_count += 1
                except Exception as e:
                    print(f"  Warning: Could not save {name}.png: {e}")
        if saved_count > 0:
            print(f"\nSaved {saved_count} visualizations to: {viz_dir}/")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        def serialize(obj):
            if isinstance(obj, torch.Tensor): return obj.tolist()
            elif isinstance(obj, dict): return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list): return [serialize(v) for v in obj]
            return obj
        with open(output_path, 'w') as f:
            json.dump(serialize(results), f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print(f"\n{'='*60}")
    print("REPSCAN COMPLETE")
    print(f"{'='*60}")
    return results
