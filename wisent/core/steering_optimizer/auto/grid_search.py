"""
Grid search for auto steering optimization.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def run_grid_search(
    wisent_model: Any,
    steering_result: Dict[str, Any],
    recommended_method: str,
    layers_to_test: List[int],
    strength_range: List[float],
    eval_pairs: List[Any],
    task_name: str,
    verbose: bool = False
) -> Tuple[List[Dict], int, float, Dict]:
    """
    Run grid search over layers and strengths.

    Args:
        wisent_model: WisentModel instance
        steering_result: Result from training
        recommended_method: Method name (CAA, TITAN, PRISM)
        layers_to_test: List of layers to test
        strength_range: List of strength values to test
        eval_pairs: Evaluation pairs
        task_name: Task name for evaluation
        verbose: Enable verbose output

    Returns:
        Tuple of (grid_results, best_layer, best_strength, best_config)
    """
    from wisent.core.evaluators.rotator import EvaluatorRotator
    from wisent.core.models import get_generate_kwargs

    if verbose:
        print(f"\n Evaluating {len(layers_to_test) * len(strength_range)} configurations...")

    # Initialize evaluator
    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.oracles")
    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
    evaluator = EvaluatorRotator(evaluator=None, task_name=task_name, autoload=False)

    grid_results = []
    best_score = -1
    best_layer = layers_to_test[0]
    best_strength = 1.0
    best_config = None

    combo_idx = 0
    total_combos = len(layers_to_test) * len(strength_range)

    for layer in layers_to_test:
        steering_vector = _get_steering_vector(steering_result, recommended_method, layer)
        if steering_vector is None:
            continue

        steering_vector = steering_vector / (steering_vector.norm() + 1e-8)

        for strength in strength_range:
            combo_idx += 1

            # Apply steering
            wisent_model.set_steering_from_raw(
                {str(layer): steering_vector},
                scale=strength,
                normalize=False
            )

            # Evaluate
            correct, total = _evaluate_pairs(
                wisent_model, eval_pairs, evaluator, task_name
            )

            wisent_model.clear_steering()

            score = correct / total if total > 0 else 0
            result = {
                'layer': layer,
                'strength': strength,
                'score': score,
                'correct': correct,
                'total': total,
            }
            grid_results.append(result)

            if verbose:
                bar = "" * int(score * 20)
                print(f"   [{combo_idx:3d}/{total_combos}] L{layer:2d} S{strength:.2f}: {score:.3f} {bar}")

            if score > best_score:
                best_score = score
                best_layer = layer
                best_strength = strength
                best_config = result

    if verbose:
        print(f"\n   Best: Layer {best_layer}, Strength {best_strength:.2f}, Score {best_score:.3f}")

    return grid_results, best_layer, best_strength, best_config or {}


def _get_steering_vector(
    steering_result: Dict[str, Any],
    method: str,
    layer: int
) -> Optional[torch.Tensor]:
    """Extract steering vector for a specific layer from training result."""
    layer_key_simple = str(layer)
    layer_key_prefixed = f"layer_{layer}"

    result = steering_result.get('result')
    if result is None or not hasattr(result, 'directions'):
        return None

    if method == "CAA":
        directions = result.directions
        if layer_key_prefixed in directions:
            return directions[layer_key_prefixed]
        elif layer_key_simple in directions:
            return directions[layer_key_simple]

    elif method == "TITAN":
        if layer_key_simple in result.directions:
            dirs = result.directions[layer_key_simple]
            weights = result.direction_weights[layer_key_simple]
            weights_norm = weights / (weights.sum() + 1e-8)
            return (dirs * weights_norm.unsqueeze(-1)).sum(dim=0)
        elif layer_key_prefixed in result.directions:
            dirs = result.directions[layer_key_prefixed]
            weights = result.direction_weights[layer_key_prefixed]
            weights_norm = weights / (weights.sum() + 1e-8)
            return (dirs * weights_norm.unsqueeze(-1)).sum(dim=0)

    elif method == "PRISM":
        if layer_key_simple in result.directions:
            return result.directions[layer_key_simple][0]
        elif layer_key_prefixed in result.directions:
            return result.directions[layer_key_prefixed][0]

    return None


def _evaluate_pairs(
    wisent_model: Any,
    eval_pairs: List[Any],
    evaluator: Any,
    task_name: str
) -> Tuple[int, int]:
    """Evaluate model on pairs and return (correct, total)."""
    from wisent.core.models import get_generate_kwargs

    correct = 0
    total = 0
    eval_subset = eval_pairs[:min(30, len(eval_pairs))]

    for pair in eval_subset:
        messages = [{"role": "user", "content": pair.prompt}]
        response = wisent_model.generate(
            [messages],
            **get_generate_kwargs(max_new_tokens=256),
        )[0]

        eval_kwargs = {
            'response': response,
            'expected': pair.positive_response.model_response,
            'question': pair.prompt,
            'choices': [pair.negative_response.model_response, pair.positive_response.model_response],
            'task_name': task_name,
        }

        if hasattr(pair, 'metadata') and pair.metadata:
            for key, value in pair.metadata.items():
                if value is not None and key not in eval_kwargs:
                    eval_kwargs[key] = value

        result = evaluator.evaluate(**eval_kwargs)
        if result.ground_truth == "TRUTHFUL":
            correct += 1
        total += 1

    return correct, total
