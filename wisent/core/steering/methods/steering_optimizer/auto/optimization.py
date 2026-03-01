"""Auto steering optimization using zwiad geometry analysis."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch

from wisent.core.config_manager import ModelConfigManager
from wisent.core.constants import (
    DEFAULT_LIMIT, DEFAULT_SCORE, BLEND_DEFAULT,
    AUTO_MAX_TIME_MINUTES, AUTO_MIN_PAIRS, AUTO_SAMPLE_SIZE,
    AUTO_N_FOLDS, AUTO_DEFAULT_STRENGTHS, AUTO_MIN_PAIRS_SPLIT,
    AUTO_LAYER_DIVISOR, SEPARATOR_WIDTH_WIDE,
)
from .training import train_recommended_method
from .grid_search import run_grid_search

logger = logging.getLogger(__name__)


def run_auto_steering_optimization(
    model_name: str,
    task_name: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    device: str = None,
    verbose: bool = False,
    use_classification_config: bool = True,
    max_time_minutes: float = AUTO_MAX_TIME_MINUTES,
    methods_to_test: Optional[List[str]] = None,
    strength_range: Optional[List[float]] = None,
    layer_range: Optional[str] = None
) -> Dict[str, Any]:
    """Automatically optimize steering using zwiad geometry analysis."""
    from wisent.core.models.wisent_model import WisentModel

    if not task_name:
        return {"error": "Task name is required for auto steering optimization"}

    if verbose:
        print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}\nAUTO STEERING OPTIMIZATION (zwiad)\n{'=' * SEPARATOR_WIDTH_WIDE}")
        print(f"   Model: {model_name}\n   Task: {task_name}\n{'=' * SEPARATOR_WIDTH_WIDE}\n")
        print("Loading model...", flush=True)

    wisent_model = WisentModel(model_name, device=device)
    num_layers = wisent_model.num_layers

    if verbose:
        print(f"Model loaded with {num_layers} layers\n")
        print(f"Generating contrastive pairs for {task_name}...", flush=True)

    pairs = _generate_pairs(task_name, limit)
    if not pairs or len(pairs) < AUTO_MIN_PAIRS:
        return {"error": f"Could not generate enough pairs for {task_name}"}
    if verbose:
        print(f"Generated {len(pairs)} contrastive pairs\n")

    # Run zwiad analysis
    recommended_method, confidence, reasoning, metrics, coherence = _run_zwiad_analysis(
        wisent_model, pairs, num_layers, verbose
    )

    # Determine search space
    layers_to_test = _get_layers_to_test(layer_range, num_layers)
    if strength_range is None:
        strength_range = list(AUTO_DEFAULT_STRENGTHS)

    if verbose:
        print(f"\nGRID SEARCH for {recommended_method}")
        print(f"   Layers: {layers_to_test}\n   Strengths: {strength_range}")
        print(f"\nTraining {recommended_method} steering vectors...")

    steering_result = train_recommended_method(
        wisent_model=wisent_model, pairs=pairs,
        method=recommended_method, layer=layers_to_test[0], verbose=verbose,
    )

    eval_pairs = pairs[len(pairs)//2:] if len(pairs) > AUTO_MIN_PAIRS_SPLIT else pairs
    grid_results, best_layer, best_strength, best_config = run_grid_search(
        wisent_model=wisent_model, steering_result=steering_result,
        recommended_method=recommended_method, layers_to_test=layers_to_test,
        strength_range=strength_range, eval_pairs=eval_pairs,
        task_name=task_name, verbose=verbose,
    )

    best_score = best_config.get('score', DEFAULT_SCORE) if best_config else DEFAULT_SCORE
    _save_config(model_name, task_name, recommended_method, best_layer,
                 best_strength, best_score, confidence, reasoning,
                 metrics, coherence, layers_to_test, strength_range, grid_results)

    if verbose:
        print(f"\nSteering optimization complete!")
        print(f"   Method: {recommended_method}, Layer: {best_layer}, Strength: {best_strength}, Score: {best_score:.3f}")

    return {
        'model_name': model_name, 'task_name': task_name,
        'recommended_method': recommended_method,
        'optimal_layer': best_layer, 'optimal_strength': best_strength,
        'best_score': best_score, 'confidence': confidence, 'reasoning': reasoning,
        'zwiad_metrics': {
            'linear_probe_accuracy': metrics.get('linear_probe_accuracy', DEFAULT_SCORE),
            'signal_strength': metrics.get('signal_strength', DEFAULT_SCORE),
            'steerability_score': metrics.get('steer_steerability_score', DEFAULT_SCORE),
            'icd': metrics.get('icd_icd', DEFAULT_SCORE),
            'concept_coherence': coherence,
        },
        'grid_search_results': grid_results,
        'steering_result': steering_result,
        'optimization_date': datetime.now().isoformat(),
        'config_saved': True,
    }


def _generate_pairs(task_name: str, limit: int) -> List:
    """Generate contrastive pairs for the task."""
    from wisent.extractors.lm_eval._registry.lm_task_pairs_generation import build_contrastive_pairs
    try:
        return build_contrastive_pairs(task_name=task_name, limit=limit)
    except Exception as e:
        logger.error(f"Failed to generate pairs for {task_name}: {e}")
        return []


def _run_zwiad_analysis(wisent_model: Any, pairs: List, num_layers: int, verbose: bool) -> tuple:
    """Run zwiad geometry analysis on collected activations."""
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations import ExtractionStrategy
    from wisent.core.geometry import compute_geometry_metrics, compute_recommendation, compute_concept_coherence

    if verbose:
        print("Collecting activations for geometry analysis...", flush=True)

    candidate_layers = list(range(0, num_layers, max(1, num_layers // AUTO_LAYER_DIVISOR)))
    if (num_layers - 1) not in candidate_layers:
        candidate_layers.append(num_layers - 1)
    candidate_layer_strs = [str(l) for l in candidate_layers]

    collector = ActivationCollector(model=wisent_model)
    sample_pairs = pairs[:min(AUTO_SAMPLE_SIZE, len(pairs))]
    layer_pos = {l: [] for l in candidate_layer_strs}
    layer_neg = {l: [] for l in candidate_layer_strs}

    for pair in sample_pairs:
        enriched = collector.collect(pair, strategy=ExtractionStrategy.CHAT_LAST, layers=candidate_layer_strs)
        for l in candidate_layer_strs:
            pos_act = enriched.positive_response.layers_activations.get(l)
            neg_act = enriched.negative_response.layers_activations.get(l)
            if pos_act is not None:
                layer_pos[l].append(pos_act)
            if neg_act is not None:
                layer_neg[l].append(neg_act)

    # Pick the layer with the strongest linear probe signal
    best_lpa, best_layer = -1.0, candidate_layer_strs[0]
    best_metrics = None
    for l in candidate_layer_strs:
        if len(layer_pos[l]) < AUTO_MIN_PAIRS or len(layer_neg[l]) < AUTO_MIN_PAIRS:
            continue
        pt = torch.stack(layer_pos[l])
        nt = torch.stack(layer_neg[l])
        m = compute_geometry_metrics(pt, nt, n_folds=AUTO_N_FOLDS)
        lpa = m.get('linear_probe_accuracy', 0.0)
        if lpa > best_lpa:
            best_lpa, best_metrics, best_layer = lpa, m, l

    if best_metrics is None:
        return "GROM", BLEND_DEFAULT, "Insufficient activations", {}, DEFAULT_SCORE

    if verbose:
        print(f"Analyzed {len(candidate_layers)} layers, best: layer {best_layer} (lpa={best_lpa:.3f})\n")

    pos_tensor = torch.stack(layer_pos[best_layer])
    neg_tensor = torch.stack(layer_neg[best_layer])
    metrics = best_metrics
    recommendation = compute_recommendation(metrics)
    coherence = compute_concept_coherence(pos_tensor, neg_tensor)

    recommended_method = recommendation.get("recommended_method", "GROM").upper()
    confidence = recommendation.get("confidence", BLEND_DEFAULT)
    reasoning = recommendation.get("reasoning", "")

    if verbose:
        print(f"   Repscan: Linear accuracy: {metrics.get('linear_probe_accuracy', DEFAULT_SCORE):.3f}")
        print(f"   Recommendation: {recommended_method} (confidence={confidence:.2f})")

    return recommended_method, confidence, reasoning, metrics, coherence


def _get_layers_to_test(layer_range: Optional[str], num_layers: int) -> List[int]:
    """Get layers to test based on layer_range or default."""
    if layer_range:
        if '-' in layer_range:
            start, end = map(int, layer_range.split('-'))
            return list(range(start, end + 1))
        elif ',' in layer_range:
            return [int(x.strip()) for x in layer_range.split(',')]
        return [int(layer_range)]
    return list(range(num_layers))


def _save_config(
    model_name: str, task_name: str, method: str, layer: int,
    strength: float, score: float, confidence: float, reasoning: str,
    metrics: Dict, coherence: float, layers_tested: List[int],
    strengths_tested: List[float], grid_results: List[Dict]
):
    """Save optimization results to model config."""
    config_manager = ModelConfigManager()
    config = config_manager.load_model_config(model_name) or {
        'model_name': model_name,
        'created_date': datetime.now().isoformat(),
        'config_version': '2.0'
    }

    if 'steering_optimization' not in config:
        config['steering_optimization'] = {}

    config['steering_optimization'].update({
        'best_method': method, 'best_layer': layer,
        'best_strength': strength, 'best_score': score,
        'optimization_date': datetime.now().isoformat(),
        'zwiad_metrics': {
            'linear_probe_accuracy': metrics.get('linear_probe_accuracy', DEFAULT_SCORE),
            'signal_strength': metrics.get('signal_strength', DEFAULT_SCORE),
            'steerability_score': metrics.get('steer_steerability_score', DEFAULT_SCORE),
            'icd': metrics.get('icd_icd', DEFAULT_SCORE),
            'concept_coherence': coherence,
        },
        'confidence': confidence, 'reasoning': reasoning,
        'grid_search': {
            'layers_tested': layers_tested,
            'strengths_tested': strengths_tested,
            'total_combinations': len(grid_results),
        }
    })

    if 'task_specific_steering' not in config:
        config['task_specific_steering'] = {}

    config['task_specific_steering'][task_name] = {
        'method': method, 'layer': layer, 'strength': strength,
        'score': score, 'confidence': confidence,
    }
    config_manager.save_model_config(model_name, **config)
