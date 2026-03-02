"""
Training functions for nurt, szlak, and wicher steering methods.

Split from method_training.py to meet 300-line limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from wisent.core.utils.config_tools.constants import PROGRESS_LOG_INTERVAL_10

if TYPE_CHECKING:
    from wisent.core.primitives.models.wisent_model import WisentModel
    from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair


def train_nurt_for_task(args, wisent_model: "WisentModel", pairs: List["ContrastivePair"]):
    """Train Concept Flow on contrastive pairs and return a NurtSteeringObject."""
    from wisent.core.control.steering_methods.methods.nurt import NurtMethod
    from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.primitives.model_interface.core.activations.activations_collector import ActivationCollector
    from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
    from wisent.core.control.steering_methods.steering_object import SteeringObjectMetadata
    from wisent.core.utils.cli.steering.core.create_nurt import (
        _create_nurt_steering_object,
    )
    from wisent.core.utils.cli.analysis.training.modify_weights.method_training import get_all_layers

    layers = [str(l) for l in str(args.layers).split(',')] if args.layers else get_all_layers(wisent_model)
    if args.verbose:
        print(f"  Collecting activations for Concept Flow training...")

    collector = ActivationCollector(model=wisent_model)
    layer_acts = {l: {"positive": [], "negative": []} for l in layers}

    for i, pair in enumerate(pairs):
        enriched = collector.collect(pair, strategy=ExtractionStrategy.CHAT_LAST, layers=layers)
        for l in layers:
            pa = enriched.positive_response.layers_activations.get(l)
            na = enriched.negative_response.layers_activations.get(l)
            if pa is not None:
                layer_acts[l]["positive"].append(pa)
            if na is not None:
                layer_acts[l]["negative"].append(na)
        if args.verbose and (i + 1) % PROGRESS_LOG_INTERVAL_10 == 0:
            print(f"    Collected {i + 1}/{len(pairs)} pairs")

    meta = SteeringObjectMetadata(
        method="nurt", model_name=args.model,
        benchmark=getattr(args, 'task', 'unknown'), category="steering",
        extraction_strategy="chat_last", num_pairs=len(pairs),
        layers=[int(l) for l in layers], hidden_dim=0,
    )
    return _create_nurt_steering_object(meta, layer_acts, layers, args)


def train_szlak_for_task(args, wisent_model: "WisentModel", pairs: List["ContrastivePair"]):
    """Train SZLAK on contrastive pairs and return a SzlakSteeringObject."""
    from wisent.core.primitives.model_interface.core.activations.activations_collector import ActivationCollector
    from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
    from wisent.core.control.steering_methods.steering_object import SteeringObjectMetadata
    from wisent.core.control.steering_methods.methods.szlak.create import (
        _create_szlak_steering_object,
    )
    from wisent.core.utils.cli.analysis.training.modify_weights.method_training import get_all_layers

    layers = [str(l) for l in str(args.layers).split(',')] if args.layers else get_all_layers(wisent_model)
    if args.verbose:
        print(f"  Collecting activations for SZLAK training...")

    collector = ActivationCollector(model=wisent_model)
    layer_acts = {l: {"positive": [], "negative": []} for l in layers}

    for i, pair in enumerate(pairs):
        enriched = collector.collect(pair, strategy=ExtractionStrategy.CHAT_LAST, layers=layers)
        for l in layers:
            pa = enriched.positive_response.layers_activations.get(l)
            na = enriched.negative_response.layers_activations.get(l)
            if pa is not None:
                layer_acts[l]["positive"].append(pa)
            if na is not None:
                layer_acts[l]["negative"].append(na)
        if args.verbose and (i + 1) % PROGRESS_LOG_INTERVAL_10 == 0:
            print(f"    Collected {i + 1}/{len(pairs)} pairs")

    meta = SteeringObjectMetadata(
        method="szlak", model_name=args.model,
        benchmark=getattr(args, 'task', 'unknown'), category="steering",
        extraction_strategy="chat_last", num_pairs=len(pairs),
        layers=[int(l) for l in layers], hidden_dim=0,
    )
    return _create_szlak_steering_object(meta, layer_acts, layers, args)


def train_wicher_for_task(args, wisent_model: "WisentModel", pairs: List["ContrastivePair"]):
    """Train WICHER on contrastive pairs and return a WicherSteeringObject."""
    from wisent.core.primitives.model_interface.core.activations.activations_collector import ActivationCollector
    from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
    from wisent.core.control.steering_methods.steering_object import SteeringObjectMetadata
    from wisent.core.control.steering_methods.methods.wicher.create import (
        _create_wicher_steering_object,
    )
    from wisent.core.utils.cli.analysis.training.modify_weights.method_training import get_all_layers

    layers = [str(l) for l in str(args.layers).split(',')] if args.layers else get_all_layers(wisent_model)
    if args.verbose:
        print(f"  Collecting activations for WICHER training...")

    collector = ActivationCollector(model=wisent_model)
    layer_acts = {l: {"positive": [], "negative": []} for l in layers}

    for i, pair in enumerate(pairs):
        enriched = collector.collect(pair, strategy=ExtractionStrategy.CHAT_LAST, layers=layers)
        for l in layers:
            pa = enriched.positive_response.layers_activations.get(l)
            na = enriched.negative_response.layers_activations.get(l)
            if pa is not None:
                layer_acts[l]["positive"].append(pa)
            if na is not None:
                layer_acts[l]["negative"].append(na)
        if args.verbose and (i + 1) % PROGRESS_LOG_INTERVAL_10 == 0:
            print(f"    Collected {i + 1}/{len(pairs)} pairs")

    meta = SteeringObjectMetadata(
        method="wicher", model_name=args.model,
        benchmark=getattr(args, 'task', 'unknown'), category="steering",
        extraction_strategy="chat_last", num_pairs=len(pairs),
        layers=[int(l) for l in layers], hidden_dim=0,
    )
    return _create_wicher_steering_object(meta, layer_acts, layers, args)
