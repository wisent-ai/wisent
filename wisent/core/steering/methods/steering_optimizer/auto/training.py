"""
Training helpers for auto steering optimization.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

from wisent.core.constants import GROM_NUM_DIRECTIONS, TECZA_NUM_DIRECTIONS

logger = logging.getLogger(__name__)


def train_recommended_method(
    wisent_model: Any,
    pairs: List[Any],
    method: str,
    layer: int,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Train the recommended steering method.

    Args:
        wisent_model: WisentModel instance
        pairs: List of contrastive pairs
        method: Recommended method name (CAA, GROM, TECZA, TETNO)
        layer: Layer parameter (not used for multi-layer training)
        verbose: Enable verbose output

    Returns:
        Dictionary with method name, layer count, and trained result
    """
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations import ExtractionStrategy

    num_layers = wisent_model.num_layers
    all_layers = [str(i) for i in range(1, num_layers + 1)]

    if verbose:
        print(f"   Collecting activations for {len(pairs)} pairs...")

    collector = ActivationCollector(model=wisent_model)
    enriched_pairs = []

    for i, pair in enumerate(pairs):
        enriched = collector.collect(pair, strategy=ExtractionStrategy.CHAT_LAST, layers=all_layers)
        enriched_pairs.append(enriched)
        if verbose and (i + 1) % 25 == 0:
            print(f"     {i + 1}/{len(pairs)} pairs processed")

    pair_set = ContrastivePairSet(pairs=enriched_pairs, name=f"{method.lower()}_training")

    if verbose:
        print(f"   Collected activations for {len(enriched_pairs)} pairs")

    if method == "CAA":
        return _train_caa(pair_set, verbose)
    elif method == "GROM":
        return _train_grom(wisent_model, pair_set, all_layers, verbose)
    elif method == "TECZA":
        return _train_tecza(wisent_model, pair_set, verbose)
    elif method == "TETNO":
        return _train_tetno(wisent_model, pair_set, all_layers, verbose)
    else:
        logger.warning(f"Unknown method {method}, falling back to CAA")
        return _train_caa(pair_set, verbose)


def _train_caa(pair_set: Any, verbose: bool) -> Dict[str, Any]:
    """Train CAA method."""
    from wisent.core.steering_methods.methods.caa import CAAMethod

    caa_method = CAAMethod()
    result = caa_method.train(pair_set)

    if verbose:
        print(f"   CAA trained on {len(pair_set.pairs)} pairs")
        print(f"     Layers: {len(result.directions)}")

    return {"method": "CAA", "layers": len(result.directions), "result": result}


def _train_grom(wisent_model: Any, pair_set: Any, all_layers: List[str], verbose: bool) -> Dict[str, Any]:
    """Train GROM method."""
    from wisent.core.steering_methods.methods.grom import GROMMethod

    layer_indices = [int(l) for l in all_layers]
    grom_method = GROMMethod(
        model=wisent_model,
        num_directions=GROM_NUM_DIRECTIONS,
        manifold_method="pca",
        steering_layers=layer_indices,
        sensor_layer=layer_indices[0],
    )

    result = grom_method.train_grom(pair_set)

    if verbose:
        print(f"   GROM trained on {len(pair_set.pairs)} pairs")
        print(f"     Layers: {len(result.layer_order)}")
        print(f"     Directions per layer: {result.directions[result.layer_order[0]].shape[0]}")

    return {"method": "GROM", "layers": len(result.layer_order), "result": result}


def _train_tecza(wisent_model: Any, pair_set: Any, verbose: bool) -> Dict[str, Any]:
    """Train TECZA method."""
    from wisent.core.steering_methods.methods.advanced import TECZAMethod

    tecza_method = TECZAMethod(
        model=wisent_model.hf_model,
        num_directions=TECZA_NUM_DIRECTIONS,
    )

    result = tecza_method.train(pair_set)

    if verbose:
        num_dirs = next(iter(result.directions.values())).shape[0]
        print(f"   TECZA trained on {len(pair_set.pairs)} pairs")
        print(f"     Layers: {len(result.directions)}")
        print(f"     Directions per layer: {num_dirs}")

    return {"method": "TECZA", "layers": len(result.directions), "result": result}


def _train_tetno(wisent_model: Any, pair_set: Any, all_layers: List[str], verbose: bool) -> Dict[str, Any]:
    """Train TETNO method."""
    from wisent.core.steering_methods.methods.advanced import TETNOMethod

    layer_indices = [int(l) for l in all_layers]
    tetno_method = TETNOMethod(
        model=wisent_model.hf_model,
        steering_layers=layer_indices,
        sensor_layer=layer_indices[0],
    )

    result = tetno_method.train_tetno(pair_set)

    if verbose:
        print(f"   TETNO trained on {len(pair_set.pairs)} pairs")
        print(f"     Layers: {len(result.behavior_vectors)}")
        print(f"     Optimal threshold: {result.optimal_threshold:.3f}")

    return {"method": "TETNO", "layers": len(result.behavior_vectors), "result": result}
