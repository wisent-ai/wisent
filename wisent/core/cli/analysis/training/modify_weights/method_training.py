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
    Automatically select the best steering method based on repscan geometry analysis.

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
        print("AUTO-SELECTING STEERING METHOD (repscan)")
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
            print("   Warning: Insufficient activations for analysis, defaulting to TITAN")
        return "titan", "titan", None

    pos_tensor = torch.stack(pos_activations)
    neg_tensor = torch.stack(neg_activations)

    metrics = compute_geometry_metrics(pos_tensor, neg_tensor, n_folds=3)
    recommendation = compute_recommendation(metrics)
    recommended_method = recommendation.get("recommended_method", "TITAN").upper()
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
    elif recommended_method == "PRISM":
        steering_method = "prism"
        modification_method = "prism"
    else:
        steering_method = "titan"
        modification_method = "titan"

    if verbose:
        print(f"\n   Selected: {steering_method.upper()} / {modification_method}")
        print("=" * 60 + "\n")

    return steering_method, modification_method, metrics


def train_titan_for_task(args, model: "WisentModel", pairs: List["ContrastivePair"]):
    """Train TITAN on contrastive pairs and return the TITANResult."""
    from wisent.core.steering_methods.methods.titan import TITANMethod
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations import ExtractionStrategy

    if args.verbose:
        print("\nTraining TITAN steering method...")

    pair_set = ContrastivePairSet(
        name=getattr(args, 'trait_label', 'steering'),
        pairs=pairs,
        task_type=args.task if hasattr(args, 'task') else None,
    )

    if args.verbose:
        print("  Collecting activations for TITAN training...")

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
    titan_method = TITANMethod(
        model=model,
        num_directions=getattr(args, 'titan_num_directions', 8),
        manifold_method="pca",
        steering_layers=layer_indices,
        sensor_layer=layer_indices[0],
    )

    titan_result = titan_method.train_titan(pair_set)

    if args.verbose:
        print(f"TITAN trained on {len(pairs)} pairs")
        print(f"  Layers: {len(titan_result.layer_order)}")
        print(f"  Directions per layer: {titan_result.directions[titan_result.layer_order[0]].shape[0]}")

    return titan_result


def train_pulse_for_task(args, wisent_model: "WisentModel", pairs: List["ContrastivePair"]):
    """Train PULSE steering for a task."""
    from wisent.core.steering_methods.methods.advanced import PULSEMethod
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations import ExtractionStrategy

    model = wisent_model.hf_model
    if args.layers:
        layers = args.layers.split(',')
    else:
        layers = get_all_layers(wisent_model)

    if args.verbose:
        print(f"  Collecting activations for PULSE training...")

    collector = ActivationCollector(model=wisent_model)
    enriched_pairs = []

    for i, pair in enumerate(pairs):
        enriched = collector.collect(pair, strategy=ExtractionStrategy.default(), layers=layers)
        enriched_pairs.append(enriched)
        if args.verbose and (i + 1) % 10 == 0:
            print(f"    Collected {i + 1}/{len(pairs)} pairs")

    pair_set = ContrastivePairSet(pairs=enriched_pairs, name="pulse_training")

    if args.verbose:
        print(f"  Collected activations for {len(enriched_pairs)} pairs")

    layer_indices = [int(l) for l in layers]
    pulse_method = PULSEMethod(
        model=model,
        steering_layers=layer_indices,
        sensor_layer=layer_indices[0],
    )

    pulse_result = pulse_method.train_pulse(pair_set)

    if args.verbose:
        print(f"PULSE trained on {len(pairs)} pairs")
        print(f"  Layers: {len(pulse_result.behavior_vectors)}")
        print(f"  Optimal threshold: {pulse_result.optimal_threshold:.3f}")

    return pulse_result


def train_prism_for_task(args, wisent_model: "WisentModel", pairs: List["ContrastivePair"]):
    """Train PRISM steering for a task."""
    from wisent.core.steering_methods.methods.advanced import PRISMMethod
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations import ExtractionStrategy

    model = wisent_model.hf_model
    if args.layers:
        layers = args.layers.split(',')
    else:
        layers = get_all_layers(wisent_model)

    if args.verbose:
        print(f"  Collecting activations for PRISM training...")

    collector = ActivationCollector(model=wisent_model)
    enriched_pairs = []

    for i, pair in enumerate(pairs):
        enriched = collector.collect(pair, strategy=ExtractionStrategy.default(), layers=layers)
        enriched_pairs.append(enriched)
        if args.verbose and (i + 1) % 10 == 0:
            print(f"    Collected {i + 1}/{len(pairs)} pairs")

    pair_set = ContrastivePairSet(pairs=enriched_pairs, name="prism_training")

    if args.verbose:
        print(f"  Collected activations for {len(enriched_pairs)} pairs")

    num_directions = getattr(args, 'prism_num_directions', 3)
    prism_method = PRISMMethod(model=model, num_directions=num_directions)

    prism_result = prism_method.train(pair_set)

    if args.verbose:
        num_dirs = next(iter(prism_result.directions.values())).shape[0]
        print(f"PRISM trained on {len(pairs)} pairs")
        print(f"  Layers: {len(prism_result.directions)}")
        print(f"  Directions per layer: {num_dirs}")

    return prism_result
