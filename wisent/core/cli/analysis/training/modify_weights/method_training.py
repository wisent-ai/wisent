"""Method training and auto-selection for weight modification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Dict, Any, Tuple

import torch

if TYPE_CHECKING:
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair

def get_all_layers(model) -> List[str]:
    """Get all layer indices as strings for a model (1-indexed for collector API)."""
    if hasattr(model, 'hf_model'):
        config = model.hf_model.config
    elif hasattr(model, 'config'):
        config = model.config
    else:
        return [str(i) for i in range(1, 37)]

    num_layers = getattr(config, 'num_hidden_layers', None) or \
                 getattr(config, 'n_layer', None) or \
                 getattr(config, 'num_layers', None) or 36

    return [str(i) for i in range(1, num_layers + 1)]

def auto_select_steering_method(
    pairs: List["ContrastivePair"],
    model: "WisentModel",
    verbose: bool = False,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """
    Automatically select the best steering method based on zwiad geometry analysis.

    Returns:
        tuple: (steering_method, modification_method, metrics)
    """
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations import ExtractionStrategy
    from wisent.core.geometry import (
        compute_geometry_metrics,
        compute_recommendation,
        compute_concept_coherence,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("AUTO-SELECTING STEERING METHOD (zwiad)")
        print("=" * 60)
        print("   Analyzing activation geometry...")

    collector = ActivationCollector(model=model)
    sample_pairs = pairs[:min(50, len(pairs))]

    num_layers = model.num_layers if hasattr(model, 'num_layers') else 36
    analysis_layer = str(int(num_layers * 0.75))

    pos_activations = []
    neg_activations = []

    for pair in sample_pairs:
        enriched = collector.collect(
            pair,
            strategy=ExtractionStrategy.default(),
            layers=[analysis_layer]
        )

        if enriched.positive_response.layers_activations.get(analysis_layer) is not None:
            pos_activations.append(enriched.positive_response.layers_activations[analysis_layer])
        if enriched.negative_response.layers_activations.get(analysis_layer) is not None:
            neg_activations.append(enriched.negative_response.layers_activations[analysis_layer])

    if len(pos_activations) < 10 or len(neg_activations) < 10:
        if verbose:
            print("   Warning: Insufficient activations for analysis, defaulting to GROM")
        return "grom", "grom", None

    pos_tensor = torch.stack(pos_activations)
    neg_tensor = torch.stack(neg_activations)

    metrics = compute_geometry_metrics(pos_tensor, neg_tensor, n_folds=3)
    recommendation = compute_recommendation(metrics)
    recommended_method = recommendation.get("recommended_method", "GROM").upper()
    confidence = recommendation.get("confidence", 0.5)
    reasoning = recommendation.get("reasoning", "")

    coherence = compute_concept_coherence(pos_tensor, neg_tensor)

    if verbose:
        print(f"\n   Repscan Analysis Results:")
        print(f"   - Linear probe accuracy: {metrics.get('linear_probe_accuracy', 0):.3f}")
        print(f"   - Signal strength:       {metrics.get('signal_strength', 0):.3f}")
        print(f"   - Concept coherence:     {coherence:.3f}")
        print(f"   - Steerability score:    {metrics.get('steer_steerability_score', 0):.3f}")
        print(f"   - ICD:                   {metrics.get('icd_icd', 0):.1f}")
        print(f"   - Recommendation:        {recommended_method} (confidence={confidence:.2f})")
        print(f"       Reasoning: {reasoning}")

    if recommended_method == "CAA":
        steering_method = "caa"
        modification_method = "directional"
    elif recommended_method == "TECZA":
        steering_method = "tecza"
        modification_method = "tecza"
    else:
        steering_method = "grom"
        modification_method = "grom"

    if verbose:
        print(f"\n   Selected: {steering_method.upper()} / {modification_method}")
        print("=" * 60 + "\n")

    return steering_method, modification_method, metrics

def train_grom_for_task(args, model: "WisentModel", pairs: List["ContrastivePair"]):
    """Train GROM on contrastive pairs and return the GROMResult."""
    from wisent.core.steering_methods.methods.grom import GROMMethod
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations import ExtractionStrategy

    if args.verbose:
        print("\nTraining GROM steering method...")

    pair_set = ContrastivePairSet(
        name=getattr(args, 'trait_label', 'steering'),
        pairs=pairs,
        task_type=args.task if hasattr(args, 'task') else None,
    )

    if args.verbose:
        print("  Collecting activations for GROM training...")

    if args.layers is None:
        layers = get_all_layers(model)
    else:
        layers = [str(l) for l in str(args.layers).split(',')]
    strategy = ExtractionStrategy.CHAT_LAST

    collector = ActivationCollector(model=model)
    enriched_pairs = []
    for i, pair in enumerate(pair_set.pairs):
        enriched_pair = collector.collect(pair, strategy=strategy, layers=layers)
        enriched_pairs.append(enriched_pair)
        if args.verbose and (i + 1) % 10 == 0:
            print(f"    Collected {i + 1}/{len(pair_set.pairs)} pairs")

    pair_set.pairs = enriched_pairs
    if args.verbose:
        print(f"  Collected activations for {len(enriched_pairs)} pairs")

    layer_indices = [int(l) for l in layers]
    grom_method = GROMMethod(
        model=model,
        num_directions=getattr(args, 'grom_num_directions', 8),
        manifold_method="pca",
        steering_layers=layer_indices,
        sensor_layer=layer_indices[0],
    )

    grom_result = grom_method.train_grom(pair_set)

    if args.verbose:
        print(f"GROM trained on {len(pairs)} pairs")
        print(f"  Layers: {len(grom_result.layer_order)}")
        print(f"  Directions per layer: {grom_result.directions[grom_result.layer_order[0]].shape[0]}")

    return grom_result

def train_tetno_for_task(args, wisent_model: "WisentModel", pairs: List["ContrastivePair"]):
    """Train TETNO steering for a task."""
    from wisent.core.steering_methods.methods.advanced import TETNOMethod
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations import ExtractionStrategy

    model = wisent_model.hf_model
    if args.layers:
        layers = args.layers.split(',')
    else:
        layers = get_all_layers(wisent_model)

    if args.verbose:
        print(f"  Collecting activations for TETNO training...")

    collector = ActivationCollector(model=wisent_model)
    enriched_pairs = []

    for i, pair in enumerate(pairs):
        enriched = collector.collect(pair, strategy=ExtractionStrategy.default(), layers=layers)
        enriched_pairs.append(enriched)
        if args.verbose and (i + 1) % 10 == 0:
            print(f"    Collected {i + 1}/{len(pairs)} pairs")

    pair_set = ContrastivePairSet(pairs=enriched_pairs, name="tetno_training")

    if args.verbose:
        print(f"  Collected activations for {len(enriched_pairs)} pairs")

    layer_indices = [int(l) for l in layers]
    tetno_method = TETNOMethod(
        model=model,
        steering_layers=layer_indices,
        sensor_layer=layer_indices[0],
    )

    tetno_result = tetno_method.train_tetno(pair_set)

    if args.verbose:
        print(f"TETNO trained on {len(pairs)} pairs")
        print(f"  Layers: {len(tetno_result.behavior_vectors)}")
        print(f"  Optimal threshold: {tetno_result.optimal_threshold:.3f}")

    return tetno_result

def train_tecza_for_task(args, wisent_model: "WisentModel", pairs: List["ContrastivePair"]):
    """Train TECZA steering for a task."""
    from wisent.core.steering_methods.methods.advanced import TECZAMethod
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations import ExtractionStrategy

    model = wisent_model.hf_model
    if args.layers:
        layers = args.layers.split(',')
    else:
        layers = get_all_layers(wisent_model)

    if args.verbose:
        print(f"  Collecting activations for TECZA training...")

    collector = ActivationCollector(model=wisent_model)
    enriched_pairs = []

    for i, pair in enumerate(pairs):
        enriched = collector.collect(pair, strategy=ExtractionStrategy.default(), layers=layers)
        enriched_pairs.append(enriched)
        if args.verbose and (i + 1) % 10 == 0:
            print(f"    Collected {i + 1}/{len(pairs)} pairs")

    pair_set = ContrastivePairSet(pairs=enriched_pairs, name="tecza_training")

    if args.verbose:
        print(f"  Collected activations for {len(enriched_pairs)} pairs")

    num_directions = getattr(args, 'tecza_num_directions', 3)
    tecza_method = TECZAMethod(model=model, num_directions=num_directions)

    tecza_result = tecza_method.train(pair_set)

    if args.verbose:
        num_dirs = next(iter(tecza_result.directions.values())).shape[0]
        print(f"TECZA trained on {len(pairs)} pairs")
        print(f"  Layers: {len(tecza_result.directions)}")
        print(f"  Directions per layer: {num_dirs}")

    return tecza_result


# Import from helpers (split to meet 300-line limit)
from wisent.core.cli.analysis.training.modify_weights._helpers.method_training_helpers import (  # noqa: E402
    train_nurt_for_task,
    train_szlak_for_task,
    train_wicher_for_task,
)
