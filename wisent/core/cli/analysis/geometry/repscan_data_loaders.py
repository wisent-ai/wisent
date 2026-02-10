"""Data loading functions for repscan CLI - handles cache, JSON, and database sources."""

import json
import sys
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import numpy as np


def load_from_cache(cache_path_str: str, task: Optional[str] = None,
                    limit: Optional[int] = None, database_url: Optional[str] = None,
                    ) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], Optional[dict]]:
    """Load activations from a pickle cache file. Handles multiple cache formats."""
    cache_path = Path(cache_path_str)
    print(f"Loading from cache file: {cache_path}")
    if not cache_path.exists():
        print(f"\nERROR: Cache file not found: {cache_path}"); sys.exit(1)
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    print(f"  Cache type: {type(cache_data)}")
    activations_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    pair_texts = None
    if isinstance(cache_data, dict):
        if 'layers' in cache_data:
            for layer, (pos, neg) in cache_data['layers'].items():
                activations_by_layer[int(layer)] = (torch.tensor(pos), torch.tensor(neg))
        elif 'activations' in cache_data and 'labels' in cache_data:
            activations = cache_data['activations']
            labels = cache_data['labels']
            metadata = cache_data.get('metadata', {})
            layer = metadata.get('layer_id', 0)
            if isinstance(activations, np.ndarray):
                activations = torch.tensor(activations)
            if isinstance(labels, np.ndarray):
                labels = torch.tensor(labels)
            pos_acts, neg_acts = activations[labels == 1], activations[labels == 0]
            n_pairs = min(len(pos_acts), len(neg_acts))
            if n_pairs > 0:
                activations_by_layer[layer] = (pos_acts[:n_pairs], neg_acts[:n_pairs])
                print(f"  Layer {layer}: {n_pairs} pairs")
        elif all(isinstance(k, int) for k in cache_data.keys()):
            for layer, (pos, neg) in cache_data.items():
                if hasattr(pos, 'shape') and hasattr(neg, 'shape'):
                    activations_by_layer[layer] = (pos, neg)
        else:
            for key, act_data in cache_data.items():
                if hasattr(act_data, 'pos_activations') and hasattr(act_data, 'neg_activations'):
                    layer = act_data.layer if hasattr(act_data, 'layer') else 0
                    activations_by_layer[layer] = (act_data.pos_activations, act_data.neg_activations)
                elif isinstance(act_data, dict) and 'pos' in act_data and 'neg' in act_data:
                    activations_by_layer[key if isinstance(key, int) else 0] = (
                        torch.tensor(act_data['pos']), torch.tensor(act_data['neg']))
        if 'pair_texts' in cache_data:
            pair_texts = cache_data['pair_texts']
    _print_layer_summary(activations_by_layer, "cache")
    if task and pair_texts is None:
        pair_texts = _try_load_pair_texts(task, limit, database_url)
    return activations_by_layer, pair_texts


def load_from_json(json_path_str: str) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], Optional[dict]]:
    """Load activations from get-activations JSON output."""
    from wisent.core.geometry.repscan.repscan_with_concepts import load_activations_from_json
    json_path = Path(json_path_str)
    print(f"Loading from JSON: {json_path}")
    if not json_path.exists():
        print(f"\nERROR: JSON file not found: {json_path}"); sys.exit(1)
    activations_by_layer, pair_texts = load_activations_from_json(str(json_path))
    _print_layer_summary(activations_by_layer, "JSON")
    return activations_by_layer, pair_texts


def load_from_database(args) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], Optional[dict]]:
    """Load activations from Supabase database."""
    from wisent.core.geometry.repscan.repscan_with_concepts import (
        load_activations_from_database, load_available_layers_from_database,
    )
    from wisent.core.geometry.data.cache import get_cached_layers
    if not args.task:
        print("ERROR: --task is required when using --from-database"); sys.exit(1)
    print(f"Loading from database: {args.model} / {args.task} / layers={args.layers or 'all'}")
    if args.layers:
        if '-' in args.layers:
            start, end = map(int, args.layers.split('-'))
            layers = list(range(start, end + 1))
        else:
            layers = [int(l.strip()) for l in args.layers.split(',')]
    else:
        cached_layers = get_cached_layers(args.task, args.model)
        if cached_layers:
            print(f"  Found {len(cached_layers)} layers in cache: {cached_layers[0]}-{cached_layers[-1]}")
            layers = cached_layers
        else:
            print(f"  Querying available layers from database...")
            layers = load_available_layers_from_database(
                model_name=args.model, task_name=args.task,
                extraction_strategy=args.extraction_strategy, database_url=args.database_url,
            )
            print(f"  Found {len(layers)} layers" if layers else "  No layers found")
    activations_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    print(f"\nLoading activations for {len(layers)} layers...")
    for layer in layers:
        try:
            pos, neg = load_activations_from_database(
                model_name=args.model, task_name=args.task, layer=layer,
                prompt_format=args.prompt_format, extraction_strategy=args.extraction_strategy,
                limit=args.limit, database_url=args.database_url,
            )
            if len(pos) > 0 and len(neg) > 0:
                activations_by_layer[layer] = (pos, neg)
                print(f"  Layer {layer}: {len(pos)} pairs")
        except Exception as e:
            print(f"  Layer {layer}: skipped ({e})")
    pair_texts = _try_load_pair_texts(args.task, args.limit, args.database_url)
    return activations_by_layer, pair_texts


def _print_layer_summary(activations_by_layer: dict, source: str) -> None:
    """Print summary of loaded layers."""
    print(f"  Loaded {len(activations_by_layer)} layers from {source}")
    for layer in sorted(activations_by_layer.keys()):
        pos, neg = activations_by_layer[layer]
        print(f"    Layer {layer}: {len(pos)} pairs")


def _try_load_pair_texts(task: str, limit: Optional[int], database_url: Optional[str]) -> Optional[dict]:
    """Try loading pair texts from database; return None on failure."""
    from wisent.core.geometry.repscan.repscan_with_concepts import load_pair_texts_from_database
    print(f"\nLoading pair texts from database for task: {task}")
    try:
        pair_texts = load_pair_texts_from_database(
            task_name=task, limit=limit or 500, database_url=database_url)
        print(f"  Loaded {len(pair_texts)} pair texts")
        return pair_texts
    except Exception as e:
        print(f"  Warning: Could not load pair texts: {e}")
        return None
