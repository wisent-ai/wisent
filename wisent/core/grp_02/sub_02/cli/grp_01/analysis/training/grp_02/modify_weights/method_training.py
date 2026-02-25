"""Method training and auto-selection for weight modification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Dict, Any, Tuple

import torch
from wisent.core.constants import CLASSIFIER_THRESHOLD, DEFAULT_SCORE, GROM_NUM_DIRECTIONS, TECZA_NUM_DIRECTIONS, GEOMETRY_CV_FOLDS, DEFAULT_METHOD_LAYER_RANGE_END, NUM_LAYERS_LARGE_MODEL, LAYER_SAMPLING_DIVISOR_TRAINING, METHOD_SELECTION_SAMPLE_SIZE, MIN_LAYER_ACTIVATIONS_FOR_GEOMETRY, SEPARATOR_WIDTH_STANDARD, PROGRESS_LOG_INTERVAL_10

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
        return [str(i) for i in range(1, DEFAULT_METHOD_LAYER_RANGE_END)]

    num_layers = getattr(config, 'num_hidden_layers', None) or \
                 getattr(config, 'n_layer', None) or \
                 getattr(config, 'num_layers', None) or NUM_LAYERS_LARGE_MODEL

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
        print("\n" + "=" * SEPARATOR_WIDTH_STANDARD)
        print("AUTO-SELECTING STEERING METHOD (zwiad)")
        print("=" * SEPARATOR_WIDTH_STANDARD)
        print("   Analyzing activation geometry...")

    collector = ActivationCollector(model=model)
    sample_pairs = pairs[:min(METHOD_SELECTION_SAMPLE_SIZE, len(pairs))]

    num_layers = model.num_layers if hasattr(model, 'num_layers') else NUM_LAYERS_LARGE_MODEL
    candidate_layers = list(range(0, num_layers, max(1, num_layers // LAYER_SAMPLING_DIVISOR_TRAINING)))
    if (num_layers - 1) not in candidate_layers:
        candidate_layers.append(num_layers - 1)
    candidate_layer_strs = [str(l) for l in candidate_layers]

    # Collect activations from all candidate layers
    layer_pos = {l: [] for l in candidate_layer_strs}
    layer_neg = {l: [] for l in candidate_layer_strs}

    for pair in sample_pairs:
        enriched = collector.collect(
            pair,
            strategy=ExtractionStrategy.default(),
            layers=candidate_layer_strs,
        )
        for l in candidate_layer_strs:
            if enriched.positive_response.layers_activations.get(l) is not None:
                layer_pos[l].append(enriched.positive_response.layers_activations[l])
            if enriched.negative_response.layers_activations.get(l) is not None:
                layer_neg[l].append(enriched.negative_response.layers_activations[l])

    # Pick the layer with the strongest signal
    best_lpa, best_metrics, best_layer = -1.0, None, candidate_layer_strs[0]
    for l in candidate_layer_strs:
        if len(layer_pos[l]) < MIN_LAYER_ACTIVATIONS_FOR_GEOMETRY or len(layer_neg[l]) < MIN_LAYER_ACTIVATIONS_FOR_GEOMETRY:
            continue
        pt = torch.stack(layer_pos[l])
        nt = torch.stack(layer_neg[l])
        m = compute_geometry_metrics(pt, nt, n_folds=GEOMETRY_CV_FOLDS)
        lpa = m.get('linear_probe_accuracy', 0.0)
        if lpa > best_lpa:
            best_lpa, best_metrics, best_layer = lpa, m, l

    if best_metrics is None:
        if verbose:
            print("   Warning: Insufficient activations for analysis, defaulting to GROM")
        return "grom", "grom", None

    pos_tensor = torch.stack(layer_pos[best_layer])
    neg_tensor = torch.stack(layer_neg[best_layer])
    metrics = best_metrics
    recommendation = compute_recommendation(metrics)
    recommended_method = recommendation.get("recommended_method", "GROM").upper()
    confidence = recommendation.get("confidence", CLASSIFIER_THRESHOLD)
    reasoning = recommendation.get("reasoning", "")

    coherence = compute_concept_coherence(pos_tensor, neg_tensor)

    if verbose:
        print(f"\n   Repscan Analysis Results:")
        print(f"   - Linear probe accuracy: {metrics.get('linear_probe_accuracy', DEFAULT_SCORE):.3f}")
        print(f"   - Signal strength:       {metrics.get('signal_strength', DEFAULT_SCORE):.3f}")
        print(f"   - Concept coherence:     {coherence:.3f}")
        print(f"   - Steerability score:    {metrics.get('steer_steerability_score', DEFAULT_SCORE):.3f}")
        print(f"   - ICD:                   {metrics.get('icd_icd', DEFAULT_SCORE):.1f}")
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
        print("=" * SEPARATOR_WIDTH_STANDARD + "\n")

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
        if args.verbose and (i + 1) % PROGRESS_LOG_INTERVAL_10 == 0:
            print(f"    Collected {i + 1}/{len(pair_set.pairs)} pairs")

    pair_set.pairs = enriched_pairs
    if args.verbose:
        print(f"  Collected activations for {len(enriched_pairs)} pairs")

    layer_indices = [int(l) for l in layers]
    grom_method = GROMMethod(
        model=model,
        num_directions=getattr(args, 'grom_num_directions', GROM_NUM_DIRECTIONS),
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
        if args.verbose and (i + 1) % PROGRESS_LOG_INTERVAL_10 == 0:
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
        if args.verbose and (i + 1) % PROGRESS_LOG_INTERVAL_10 == 0:
            print(f"    Collected {i + 1}/{len(pairs)} pairs")

    pair_set = ContrastivePairSet(pairs=enriched_pairs, name="tecza_training")

    if args.verbose:
        print(f"  Collected activations for {len(enriched_pairs)} pairs")

    num_directions = getattr(args, 'tecza_num_directions', TECZA_NUM_DIRECTIONS)
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
