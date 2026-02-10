"""RepScan CLI command for geometry analysis with concept decomposition."""

import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import json
import sys
import pickle
from pathlib import Path
from typing import Optional


def execute_repscan(args):
    """Execute the repscan command."""
    print(f"\n{'='*60}")
    print("REPSCAN - Geometry Analysis with Concept Decomposition")
    print(f"{'='*60}")

    # Import dependencies
    from wisent.core.geometry.repscan.repscan_with_concepts import (
        run_repscan_with_concept_naming,
        load_activations_from_database,
        load_pair_texts_from_database,
        load_available_layers_from_database,
    )
    from wisent.core.geometry.data.cache import get_cached_layers
    import torch

    activations_by_layer = {}
    pair_texts = None

    # Determine data source
    if args.from_cache:
        cache_path = Path(args.from_cache)
        print(f"Loading from cache file: {cache_path}")

        if not cache_path.exists():
            print(f"\nERROR: Cache file not found: {cache_path}")
            sys.exit(1)

        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        print(f"  Cache type: {type(cache_data)}")

        # Handle different cache formats
        import numpy as np
        if isinstance(cache_data, dict):
            # Check if it's layer-keyed or has 'layers' key
            if 'layers' in cache_data:
                # Format: {layers: {layer: (pos, neg)}}
                for layer, (pos, neg) in cache_data['layers'].items():
                    activations_by_layer[int(layer)] = (torch.tensor(pos), torch.tensor(neg))
            elif 'activations' in cache_data and 'labels' in cache_data:
                # Optuna cache format: activations + labels array
                activations = cache_data['activations']
                labels = cache_data['labels']
                metadata = cache_data.get('metadata', {})
                layer = metadata.get('layer_id', 0)

                # Split by labels (0=neg, 1=pos)
                if isinstance(activations, np.ndarray):
                    activations = torch.tensor(activations)
                if isinstance(labels, np.ndarray):
                    labels = torch.tensor(labels)

                pos_mask = labels == 1
                neg_mask = labels == 0

                pos_acts = activations[pos_mask]
                neg_acts = activations[neg_mask]

                # Match pairs by count
                n_pairs = min(len(pos_acts), len(neg_acts))
                if n_pairs > 0:
                    activations_by_layer[layer] = (pos_acts[:n_pairs], neg_acts[:n_pairs])
                    print(f"  Layer {layer}: {n_pairs} pairs (from {len(pos_acts)} pos, {len(neg_acts)} neg)")
            elif all(isinstance(k, int) for k in cache_data.keys()):
                # Format: {layer: (pos, neg)}
                for layer, (pos, neg) in cache_data.items():
                    if hasattr(pos, 'shape') and hasattr(neg, 'shape'):
                        activations_by_layer[layer] = (pos, neg)
            else:
                # ActivationData format from optuna cache
                for key, act_data in cache_data.items():
                    if hasattr(act_data, 'pos_activations') and hasattr(act_data, 'neg_activations'):
                        # Extract layer from key if possible
                        layer = act_data.layer if hasattr(act_data, 'layer') else 0
                        activations_by_layer[layer] = (act_data.pos_activations, act_data.neg_activations)
                    elif isinstance(act_data, dict) and 'pos' in act_data and 'neg' in act_data:
                        activations_by_layer[key if isinstance(key, int) else 0] = (
                            torch.tensor(act_data['pos']),
                            torch.tensor(act_data['neg'])
                        )

            if 'pair_texts' in cache_data:
                pair_texts = cache_data['pair_texts']

        print(f"  Loaded {len(activations_by_layer)} layers from cache")
        for layer in sorted(activations_by_layer.keys()):
            pos, neg = activations_by_layer[layer]
            print(f"    Layer {layer}: {len(pos)} pairs")

        # Load pair_texts from database if --task is provided (needed for LLM concept naming)
        if args.task and pair_texts is None:
            print(f"\nLoading pair texts from database for task: {args.task}")
            try:
                pair_texts = load_pair_texts_from_database(
                    task_name=args.task,
                    limit=args.limit or 500,
                    database_url=args.database_url,
                )
                print(f"  Loaded {len(pair_texts)} pair texts")
            except Exception as e:
                print(f"  Warning: Could not load pair texts: {e}")
                print(f"  Concepts will use generic names (concept_1, concept_2, etc.)")
                pair_texts = None

    elif args.from_database:
        if not args.task:
            print("ERROR: --task is required when using --from-database")
            sys.exit(1)

        print(f"Loading from database...")
        print(f"  Model: {args.model}")
        print(f"  Task: {args.task}")
        print(f"  Layers: {args.layers or 'all available'}")
        print(f"  Prompt format: {args.prompt_format}")
        print(f"  Extraction strategy: {args.extraction_strategy}")

        # Parse layers
        if args.layers:
            if '-' in args.layers:
                start, end = map(int, args.layers.split('-'))
                layers = list(range(start, end + 1))
            else:
                layers = [int(l.strip()) for l in args.layers.split(',')]
        else:
            # First check cache for available layers
            cached_layers = get_cached_layers(args.task, args.model)
            if cached_layers:
                print(f"  Found {len(cached_layers)} layers in cache: {cached_layers[0]}-{cached_layers[-1]}")
                layers = cached_layers
            else:
                # Fall back to database query
                print(f"  Querying available layers from database...")
                layers = load_available_layers_from_database(
                    model_name=args.model,
                    task_name=args.task,
                    extraction_strategy=args.extraction_strategy,
                    database_url=args.database_url,
                )
                print(f"  Found {len(layers)} layers: {layers[0]}-{layers[-1]}" if layers else "  No layers found")

        # Load activations for each layer
        print(f"\nLoading activations for {len(layers)} layers...")
        for layer in layers:
            try:
                pos, neg = load_activations_from_database(
                    model_name=args.model,
                    task_name=args.task,
                    layer=layer,
                    prompt_format=args.prompt_format,
                    extraction_strategy=args.extraction_strategy,
                    limit=args.limit,
                    database_url=args.database_url,
                )
                if len(pos) > 0 and len(neg) > 0:
                    activations_by_layer[layer] = (pos, neg)
                    print(f"  Layer {layer}: {len(pos)} pairs")
            except Exception as e:
                print(f"  Layer {layer}: skipped ({e})")

        # Load pair texts for concept naming
        print(f"\nLoading pair texts...")
        try:
            pair_texts = load_pair_texts_from_database(
                task_name=args.task,
                limit=args.limit or 200,
                database_url=args.database_url,
            )
            print(f"  Loaded {len(pair_texts)} pair texts")
        except Exception as e:
            print(f"  Warning: Could not load pair texts: {e}")
            pair_texts = None

    else:
        print("ERROR: --from-database or --from-cache is required.")
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
    print(f"\nLayers concatenated: {results.get('n_layers')} (layers {results.get('layers_used', [])})")
    print(f"Total dimensions: {results.get('total_dims')}")
    print(f"Pairs analyzed: {results.get('n_pairs')}")
    print(f"\nRecommended method: {results.get('recommended_method')}")
    print(f"Confidence: {results.get('recommendation_confidence', 0):.2f}")

    # Print key metrics
    metrics = results.get("metrics", {})
    print(f"\n--- Key Metrics (all-layer concatenated) ---")
    for name, key in [("Signal strength", "signal_strength"), ("Linear probe", "linear_probe_accuracy"),
                       ("MLP probe", "mlp_probe_accuracy"), ("KNN", "knn_accuracy"), ("KNN PCA", "knn_pca_accuracy")]:
        print(f"{name}: {metrics.get(key, 0):.3f}")

    # Print editability analysis
    editability = results.get("editability_analysis")
    if editability:
        print(f"\n--- Editability Analysis ---")
        print(f"Editability score: {editability.get('editability_score', 0):.3f}  |  Verdict: {editability['verdict']}")
        print(f"Editing capacity: {editability['editing_capacity']:.3f}")
        print(f"Steering survival: {editability['steering_survival_ratio']:.3f}")
        print(f"Spectral decay: {editability['spectral_decay_rate']:.4f}  |  Participation ratio: {editability.get('participation_ratio', 0):.1f}")
        for warning in editability.get("warnings", []):
            print(f"  WARNING: {warning}")

    # Print concept decomposition
    decomposition = results.get("concept_decomposition")
    if decomposition:
        n_concepts = decomposition.get("n_concepts", 1)
        print(f"\n--- Concept Decomposition ---")
        print(f"Number of concepts detected: {n_concepts}")

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
                png_path = viz_dir / f"{name}.png"
                try:
                    png_bytes = base64.b64decode(b64_data)
                    with open(png_path, 'wb') as f:
                        f.write(png_bytes)
                    saved_count += 1
                except Exception as e:
                    print(f"  Warning: Could not save {name}.png: {e}")

        if saved_count > 0:
            print(f"\nSaved {saved_count} visualizations to: {viz_dir}/")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tensors to lists for JSON serialization
        def serialize(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(v) for v in obj]
            return obj

        serialized = serialize(results)

        with open(output_path, 'w') as f:
            json.dump(serialized, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print(f"\n{'='*60}")
    print("REPSCAN COMPLETE")
    print(f"{'='*60}")

    return results
