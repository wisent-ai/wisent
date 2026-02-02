"""Auto steering optimization using repscan geometry analysis."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch

from wisent.core.config_manager import ModelConfigManager
from .training import train_recommended_method
from .grid_search import run_grid_search

logger = logging.getLogger(__name__)


def run_auto_steering_optimization(
    model_name: str,
    task_name: Optional[str] = None,
    limit: int = 100,
    device: str = None,
    verbose: bool = False,
    use_classification_config: bool = True,
    max_time_minutes: float = 60.0,
    methods_to_test: Optional[List[str]] = None,
    strength_range: Optional[List[float]] = None,
    layer_range: Optional[str] = None
) -> Dict[str, Any]:
    """Automatically optimize steering using repscan geometry analysis."""
    from wisent.core.models.wisent_model import WisentModel

    if not task_name:
        return {"error": "Task name is required for auto steering optimization"}

    if verbose:
        print(f"\n{'='*70}\nAUTO STEERING OPTIMIZATION (repscan)\n{'='*70}")
        print(f"   Model: {model_name}\n   Task: {task_name}\n{'='*70}\n")
        print("Loading model...", flush=True)

    wisent_model = WisentModel(model_name, device=device)
    num_layers = wisent_model.num_layers
    if verbose:
        print(f"Model loaded with {num_layers} layers\n")
        print(f"Generating contrastive pairs for {task_name}...", flush=True)

    pairs = _generate_pairs(task_name, limit)
    if not pairs or len(pairs) < 10:
        return {"error": f"Could not generate enough pairs for {task_name}"}
    if verbose:
        print(f"Generated {len(pairs)} contrastive pairs\n")

    # Run repscan analysis
    recommended_method, confidence, reasoning, metrics, coherence = _run_repscan_analysis(
        wisent_model, pairs, num_layers, verbose
    )

    # Determine search space
    layers_to_test = _get_layers_to_test(layer_range, num_layers)
    if strength_range is None:
        strength_range = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    if verbose:
        print(f"\nGRID SEARCH for {recommended_method}")
        print(f"   Layers: {layers_to_test}\n   Strengths: {strength_range}")
        print(f"\nTraining {recommended_method} steering vectors...")

    steering_result = train_recommended_method(
        wisent_model=wisent_model, pairs=pairs,
        method=recommended_method, layer=layers_to_test[0], verbose=verbose,
    )

    eval_pairs = pairs[len(pairs)//2:] if len(pairs) > 20 else pairs
    grid_results, best_layer, best_strength, best_config = run_grid_search(
        wisent_model=wisent_model, steering_result=steering_result,
        recommended_method=recommended_method, layers_to_test=layers_to_test,
        strength_range=strength_range, eval_pairs=eval_pairs,
        task_name=task_name, verbose=verbose,
    )

    best_score = best_config.get('score', 0.0) if best_config else 0.0
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
        'repscan_metrics': {
            'linear_probe_accuracy': metrics.get('linear_probe_accuracy', 0),
            'signal_strength': metrics.get('signal_strength', 0),
            'steerability_score': metrics.get('steer_steerability_score', 0),
            'icd': metrics.get('icd_icd', 0),
            'concept_coherence': coherence,
        },
        'grid_search_results': grid_results,
        'steering_result': steering_result,
        'optimization_date': datetime.now().isoformat(),
        'config_saved': True,
    }


def _generate_pairs(task_name: str, limit: int) -> List:
    """Generate contrastive pairs for the task."""
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import build_contrastive_pairs
    try:
        return build_contrastive_pairs(task_name=task_name, limit=limit)
    except Exception as e:
        logger.error(f"Failed to generate pairs for {task_name}: {e}")
        return []


def _run_repscan_analysis(wisent_model: Any, pairs: List, num_layers: int, verbose: bool) -> tuple:
    """Run repscan geometry analysis on collected activations."""
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations import ExtractionStrategy
    from wisent.core.geometry import compute_geometry_metrics, compute_recommendation, compute_concept_coherence

    if verbose:
        print("Collecting activations for geometry analysis...", flush=True)

    analysis_layer = str(int(num_layers * 0.75))
    collector = ActivationCollector(model=wisent_model)
    sample_pairs = pairs[:min(50, len(pairs))]
    pos_activations, neg_activations = [], []

    for pair in sample_pairs:
        enriched = collector.collect(pair, strategy=ExtractionStrategy.CHAT_LAST, layers=[analysis_layer])
        pos_act = enriched.positive_response.layers_activations.get(analysis_layer)
        neg_act = enriched.negative_response.layers_activations.get(analysis_layer)
        if pos_act is not None:
            pos_activations.append(pos_act)
        if neg_act is not None:
            neg_activations.append(neg_act)

    if len(pos_activations) < 10 or len(neg_activations) < 10:
        return "TITAN", 0.5, "Insufficient activations", {}, 0.0

    if verbose:
        print(f"Collected {len(pos_activations)} pos and {len(neg_activations)} neg activations\n")

    pos_tensor = torch.stack(pos_activations)
    neg_tensor = torch.stack(neg_activations)
    metrics = compute_geometry_metrics(pos_tensor, neg_tensor, n_folds=3)
    recommendation = compute_recommendation(metrics)
    coherence = compute_concept_coherence(pos_tensor, neg_tensor)

    recommended_method = recommendation.get("recommended_method", "TITAN").upper()
    confidence = recommendation.get("confidence", 0.5)
    reasoning = recommendation.get("reasoning", "")

    if verbose:
        print(f"   Repscan: Linear accuracy: {metrics.get('linear_probe_accuracy', 0):.3f}")
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
    return list(range(num_layers // 2, num_layers))


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
        'repscan_metrics': {
            'linear_probe_accuracy': metrics.get('linear_probe_accuracy', 0),
            'signal_strength': metrics.get('signal_strength', 0),
            'steerability_score': metrics.get('steer_steerability_score', 0),
            'icd': metrics.get('icd_icd', 0),
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
