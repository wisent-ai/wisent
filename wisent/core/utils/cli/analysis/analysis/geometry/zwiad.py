"""Zwiad CLI command for geometry analysis with concept decomposition."""

import os
# NUMBA_NUM_THREADS deliberately not set here — see runner.py.

import json
import sys
import pickle
from pathlib import Path
from typing import Optional
from wisent.core.control.steering_methods.configs.optimal import get_optimal
from wisent.core import constants as _C


def execute_zwiad(args):
    """Execute the zwiad command."""
    print(f"\n{'='*_C.SEPARATOR_WIDTH_STANDARD}")
    print("ZWIAD - Geometry Analysis with Concept Decomposition")
    print(f"{'='*_C.SEPARATOR_WIDTH_STANDARD}")

    # Import dependencies
    from wisent.core.reading.modules.modules.zwiad.zwiad_with_concepts import (
        run_zwiad_with_concept_naming,
        load_activations_from_hf,
        load_pair_texts_from_hf,
        load_available_layers_from_hf,
    )
    from wisent.core.reading.modules.utilities.data.cache import get_cached_layers
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
                layer = metadata['layer_id']

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
            print(f"\nLoading pair texts for task: {args.task}")
            try:
                pair_texts = load_pair_texts_from_hf(
                    task_name=args.task,
                    limit=None,
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
        print(f"  Extraction component: {args.extraction_component}")

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
                print(f"  Querying available layers...")
                layers = load_available_layers_from_hf(
                    model_name=args.model,
                    task_name=args.task,
                    extraction_strategy=args.extraction_strategy,
                )
                print(f"  Found {len(layers)} layers: {layers[0]}-{layers[-1]}" if layers else "  No layers found")

        # Load activations for each layer
        print(f"\nLoading activations for {len(layers)} layers...")
        for layer in layers:
            try:
                pos, neg = load_activations_from_hf(
                    model_name=args.model,
                    task_name=args.task,
                    layer=layer,
                    extraction_strategy=args.extraction_strategy,
                    limit=None,
                )
                if len(pos) > 0 and len(neg) > 0:
                    activations_by_layer[layer] = (pos, neg)
                    print(f"  Layer {layer}: {len(pos)} pairs")
            except Exception as e:
                print(f"  Layer {layer}: skipped ({e})")

        # Load pair texts for concept naming
        print(f"\nLoading pair texts...")
        try:
            pair_texts = load_pair_texts_from_hf(
                task_name=args.task,
                limit=None,
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

    print(f"\n{'='*_C.SEPARATOR_WIDTH_STANDARD}")
    print(f"Running Zwiad protocol (steps: {args.steps})")
    print(f"{'='*_C.SEPARATOR_WIDTH_STANDARD}")

    _geo = {k: getattr(args, k) for k in ("probe_small_hidden", "probe_mlp_hidden", "probe_mlp_alpha",
        "spectral_n_neighbors", "direction_n_bootstrap", "direction_subset_fraction", "direction_std_penalty",
        "consistency_w_cosine", "consistency_w_positive", "consistency_w_high_sim",
        "sparsity_threshold_fraction", "detection_threshold", "direction_moderate_similarity")}
    results = run_zwiad_with_concept_naming(
        activations_by_layer=activations_by_layer, llm_model=args.llm_model,
        concept_naming_n_samples=args.concept_naming_n_samples, min_clusters=getattr(args, 'min_clusters', None),
        pair_texts=pair_texts, generate_visualizations=not args.no_visualizations,
        steps=args.steps, output_path=args.output,
        cv_folds=args.cv_folds, min_concept_pairs=args.min_concept_pairs,
        zwiad_score_primary=args.zwiad_score_primary, zwiad_score_secondary=args.zwiad_score_secondary,
        zwiad_score_tertiary=args.zwiad_score_tertiary,
        zwiad_editability_threshold=args.zwiad_editability_threshold,
        zwiad_przelom_bonus_max=args.zwiad_przelom_bonus_max, **_geo,
    )

    _sep = '=' * _C.SEPARATOR_WIDTH_STANDARD
    print(f"\n{_sep}\nRESULTS\n{_sep}")
    print(f"\nLayers: {results.get('n_layers')} ({results.get('layers_used', [])})  dims: {results.get('total_dims')}  pairs: {results.get('n_pairs')}")
    conf = results.get('recommendation_confidence')
    print(f"Recommended: {results.get('recommended_method')}  confidence: {f'{conf:.2f}' if conf is not None else 'N/A'}")
    metrics = results.get("metrics", {})
    if metrics:
        print(f"\n--- Key Metrics ---")
        for _mk in ("signal_strength", "linear_probe_accuracy", "mlp_probe_accuracy", "knn_accuracy", "knn_pca_accuracy"):
            print(f"  {_mk}: {metrics[_mk]:.3f}")

    decomposition = results.get("concept_decomposition")
    if decomposition:
        n_concepts = decomposition.get("n_concepts", 1)
        print(f"\n--- Concept Decomposition ---")
        print(f"Number of concepts detected: {n_concepts}")

        for concept in decomposition.get("concepts", []):
            print(f"\nConcept {concept['id']}: {concept.get('name', 'Unnamed')}")
            print(f"  Pairs: {concept['n_pairs']}")
            print(f"  Silhouette: {concept['silhouette_score']:.3f}")
            if 'optimal_layer' in concept:
                print(f"  Optimal layer: {concept['optimal_layer']} (acc: {concept['optimal_layer_accuracy']:.3f})")

            rep_pairs = concept.get("representative_pairs", [])
            if rep_pairs:
                print(f"  Representative pairs:")
                for pair in rep_pairs[:_C.DISPLAY_TOP_N_TINY]:
                    if isinstance(pair, dict):
                        prompt = pair.get("prompt", "")[:_C.DISPLAY_TRUNCATION_ERROR]
                        print(f"    - {prompt}...")
                    elif isinstance(pair, int):
                        print(f"    - pair index {pair}")

    if not args.no_visualizations and decomposition:
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

    results["extraction_component"] = getattr(args, "extraction_component", get_optimal("extraction_component"))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import numpy as _np
        def serialize(obj):
            if isinstance(obj, torch.Tensor): return obj.tolist()
            if isinstance(obj, _np.ndarray): return obj.tolist()
            if isinstance(obj, dict): return {str(k): serialize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [serialize(v) for v in obj]
            return obj.item() if hasattr(obj, "item") else obj

        serialized = serialize(results)

        with open(output_path, 'w') as f:
            json.dump(serialized, f, indent=_C.JSON_INDENT)
        print(f"\nResults saved to: {output_path}")
    print("\n" + "="*_C.SEPARATOR_WIDTH_STANDARD + "\nZWIAD COMPLETE\n" + "="*_C.SEPARATOR_WIDTH_STANDARD)
    return results
