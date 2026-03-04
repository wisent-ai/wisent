"""Auto steering optimization using zwiad geometry analysis."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch

from wisent.core.utils.config_tools.config import ModelConfigManager
from wisent.core.utils.config_tools.constants import (
    SEPARATOR_WIDTH_WIDE, ARCHITECTURE_MODULE_LIMIT,
)
from .training import train_recommended_method
from .grid_search import run_grid_search

logger = logging.getLogger(__name__)


def run_auto_steering_optimization(
    model_name: str,
    limit: int,
    min_norm_threshold: float,
    task_name: Optional[str] = None,
    device: str = None,
    verbose: bool = False,
    use_classification_config: bool = True,
    max_time_minutes: float = None,
    methods_to_test: Optional[List[str]] = None,
    strength_range: tuple[float, ...] = None,
    layer_range: Optional[str] = None,
    min_clusters: int = None,
    grom_params: Dict[str, Any] = None,
    tetno_params: Dict[str, Any] = None,
    tecza_params: Optional[Dict[str, Any]] = None,
    auto_min_pairs: int = None,
    auto_sample_size: int = None,
    auto_n_folds: int = None,
    auto_min_pairs_split: int = None,
    auto_layer_divisor: int = None, spectral_n_neighbors: int = None,
    subsample_threshold: int = None, pca_dims_limit: int = None, *, train_ratio: float,
) -> Dict[str, Any]:
    """Automatically optimize steering using zwiad geometry analysis."""
    _required = {"max_time_minutes": max_time_minutes, "strength_range": strength_range,
        "auto_min_pairs": auto_min_pairs, "auto_sample_size": auto_sample_size,
        "auto_n_folds": auto_n_folds, "auto_min_pairs_split": auto_min_pairs_split,
        "auto_layer_divisor": auto_layer_divisor, "spectral_n_neighbors": spectral_n_neighbors,
        "subsample_threshold": subsample_threshold, "pca_dims_limit": pca_dims_limit, "train_ratio": train_ratio}
    for _name, _val in _required.items():
        if _val is None:
            raise ValueError(f"{_name} is required")
    from wisent.core.primitives.models.wisent_model import WisentModel
    if not task_name:
        return {"error": "Task name is required for auto steering optimization"}
    if verbose:
        sep = '=' * SEPARATOR_WIDTH_WIDE
        print(f"\n{sep}\nAUTO STEERING OPTIMIZATION (zwiad)\n{sep}")
        print(f"   Model: {model_name}\n   Task: {task_name}\n{sep}\nLoading model...", flush=True)
    wisent_model = WisentModel(model_name, device=device)
    num_layers = wisent_model.num_layers
    if verbose:
        print(f"Model loaded with {num_layers} layers\nGenerating contrastive pairs for {task_name}...", flush=True)
    pairs = _generate_pairs(task_name, limit, train_ratio=train_ratio)
    if not pairs or len(pairs) < auto_min_pairs:
        return {"error": f"Could not generate enough pairs for {task_name}"}
    if verbose:
        print(f"Generated {len(pairs)} contrastive pairs\n")
    recommended_method, confidence, reasoning, metrics, coherence = _run_zwiad_analysis(
        wisent_model, pairs, num_layers, min_clusters=min_clusters, verbose=verbose,
        spectral_n_neighbors=spectral_n_neighbors, subsample_threshold=subsample_threshold, pca_dims_limit=pca_dims_limit,
    )

    # Determine search space
    layers_to_test = _get_layers_to_test(layer_range, num_layers)
    if verbose:
        print(f"\nGRID SEARCH for {recommended_method}")
        print(f"   Layers: {layers_to_test}\n   Strengths: {strength_range}")
        print(f"\nTraining {recommended_method} steering vectors...")

    first_layer = next(iter(layers_to_test))
    if grom_params is None:
        raise ValueError("grom_params dict is required with all GROM hyperparameters")
    if tetno_params is None:
        raise ValueError("tetno_params dict is required with all TETNO hyperparameters")
    grom_kw = {f"grom_{k}": v for k, v in grom_params.items()}
    tetno_kw = {f"tetno_{k}": v for k, v in tetno_params.items()}
    steering_result = train_recommended_method(
        wisent_model=wisent_model, pairs=pairs,
        method=recommended_method, layer=first_layer, verbose=verbose,
        **tetno_kw,
        **grom_kw,
        **(tecza_params or {}),
    )

    eval_pairs = pairs[len(pairs)//2:] if len(pairs) > auto_min_pairs_split else pairs
    grid_results, best_layer, best_strength, best_config, method_params = run_grid_search(
        wisent_model=wisent_model, steering_result=steering_result,
        recommended_method=recommended_method, layers_to_test=layers_to_test,
        strength_range=strength_range, eval_pairs=eval_pairs,
        task_name=task_name, min_norm_threshold=min_norm_threshold, verbose=verbose,
    )

    if not best_config:
        raise ValueError("Grid search returned no results. Check input data.")
    best_score = best_config['score']
    _save_config(model_name, task_name, recommended_method, best_layer,
                 best_strength, best_score, confidence, reasoning,
                 metrics, coherence, layers_to_test, strength_range, grid_results,
                 method_params=method_params)

    if verbose:
        print(f"\nSteering optimization complete! Method: {recommended_method}, Layer: {best_layer}, Strength: {best_strength}, Score: {best_score:.3f}")

    zwiad = {
        'linear_probe_accuracy': metrics['linear_probe_accuracy'],
        'signal_strength': metrics['signal_strength'],
        'steerability_score': metrics['steer_steerability_score'],
        'icd': metrics['icd_icd'], 'concept_coherence': coherence,
    }
    return {
        'model_name': model_name, 'task_name': task_name,
        'recommended_method': recommended_method,
        'optimal_layer': best_layer, 'optimal_strength': best_strength,
        'best_score': best_score, 'confidence': confidence, 'reasoning': reasoning,
        'zwiad_metrics': zwiad, 'grid_search_results': grid_results,
        'steering_result': steering_result,
        'optimization_date': datetime.now().isoformat(), 'config_saved': True,
    }


def _generate_pairs(task_name: str, limit: int, *, train_ratio: float) -> List:
    """Generate contrastive pairs for the task."""
    from wisent.extractors.lm_eval.lm_task_pairs_generation import build_contrastive_pairs
    try:
        return build_contrastive_pairs(task_name=task_name, limit=limit, train_ratio=train_ratio)
    except Exception as e:
        logger.error(f"Failed to generate pairs for {task_name}: {e}")
        return []


def _run_zwiad_analysis(wisent_model: Any, pairs: List, num_layers: int, min_clusters: int, verbose: bool, *, spectral_n_neighbors: int, subsample_threshold: int, pca_dims_limit: int) -> tuple:
    """Run zwiad geometry analysis on collected activations."""
    from wisent.core.primitives.model_interface.core.activations.activations_collector import ActivationCollector
    from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
    from wisent.core.reading.modules import compute_geometry_metrics, compute_recommendation, compute_concept_coherence

    if verbose:
        print("Collecting activations for geometry analysis...", flush=True)

    candidate_layers = list(range(0, num_layers, max(1, num_layers // auto_layer_divisor)))
    if (num_layers - 1) not in candidate_layers:
        candidate_layers.append(num_layers - 1)
    candidate_layer_strs = [str(l) for l in candidate_layers]

    collector = ActivationCollector(model=wisent_model, architecture_module_limit=ARCHITECTURE_MODULE_LIMIT)
    sample_pairs = pairs[:min(auto_sample_size, len(pairs))]
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
        if len(layer_pos[l]) < auto_min_pairs or len(layer_neg[l]) < auto_min_pairs:
            continue
        pt = torch.stack(layer_pos[l])
        nt = torch.stack(layer_neg[l])
        m = compute_geometry_metrics(pt, nt, min_clusters=min_clusters, n_folds=auto_n_folds, spectral_n_neighbors=spectral_n_neighbors, subsample_threshold=subsample_threshold, pca_dims_limit=pca_dims_limit)
        lpa = m.get('linear_probe_accuracy', 0.0)
        if lpa > best_lpa:
            best_lpa, best_metrics, best_layer = lpa, m, l

    if best_metrics is None:
        raise ValueError("Insufficient activations for zwiad analysis. No layers had enough data.")

    if verbose:
        print(f"Analyzed {len(candidate_layers)} layers, best: layer {best_layer} (lpa={best_lpa:.3f})\n")

    pos_tensor = torch.stack(layer_pos[best_layer])
    neg_tensor = torch.stack(layer_neg[best_layer])
    metrics = best_metrics
    recommendation = compute_recommendation(metrics)
    coherence = compute_concept_coherence(pos_tensor, neg_tensor)

    recommended_method = recommendation["recommended_method"].upper()
    confidence = recommendation["confidence"]
    reasoning = recommendation["reasoning"]

    if verbose:
        print(f"   Repscan: Linear accuracy: {metrics['linear_probe_accuracy']:.3f}")
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
    strengths_tested: List[float], grid_results: List[Dict],
    method_params: Optional[Dict[str, Any]] = None,
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
            'linear_probe_accuracy': metrics['linear_probe_accuracy'],
            'signal_strength': metrics['signal_strength'],
            'steerability_score': metrics['steer_steerability_score'],
            'icd': metrics['icd_icd'],
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

    task_entry = {
        'method': method, 'layer': layer, 'strength': strength,
        'score': score, 'confidence': confidence,
    }
    if method_params:
        # Filter out non-serializable values (e.g. None layers lists)
        serializable_params = {}
        for k, v in method_params.items():
            if v is not None and not callable(v):
                try:
                    json.dumps(v)
                    serializable_params[k] = v
                except (TypeError, ValueError):
                    pass
        task_entry['method_params'] = serializable_params
    config['task_specific_steering'][task_name] = task_entry
    config.setdefault('optimization_method', 'auto')
    config_manager.save_model_config(model_name, **config)

    # Also persist method_params via the typed config system
    if method_params:
        try:
            from wisent.core.utils.config_tools.config.convenience import save_steering_config
            save_steering_config(
                model_name=model_name,
                method=method,
                token_aggregation="chat_last",
                prompt_strategy="default",
                normalize_mode="l2",
                strategy="auto",
                optimization_method="auto",
                metric="accuracy",
                direction_weighting="primary_only",
                task_name=task_name,
                layer=layer,
                strength=strength,
                score=score,
                method_params=method_params if method_params else None,
            )
        except Exception as e:
            logger.warning(f"Could not persist method_params via config system: {e}")
