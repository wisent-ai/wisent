"""
Common utilities for research analysis.

Shared database access, data structures, and metric computation functions.
Uses wisent library functions where possible.

Re-exports from common_db and common_data for backward compatibility.
"""

from typing import Dict, Optional

import numpy as np
import torch

# Import wisent library functions
from wisent.core.control.steering_methods.registry import SteeringMethodRegistry, list_steering_methods
from wisent.core.reading.modules import compute_geometry_metrics as wisent_compute_geometry_metrics

# Re-export everything from submodules
from common_data import ActivationData, BenchmarkResults
from wisent.core.utils.config_tools.constants import ZERO_THRESHOLD
from common_db import (
    DB_CONFIG,
    RESEARCH_MODELS,
    get_model_info,
    get_all_models_with_activations,
    bytes_to_vector,
    load_activations_from_db,
)

# All available steering methods - from wisent registry
STEERING_METHODS = list_steering_methods()


def compute_geometry_metrics(pos_activations: np.ndarray, neg_activations: np.ndarray) -> Dict[str, float]:
    """
    Compute geometry metrics using wisent's comprehensive geometry analysis.

    Args:
        pos_activations: [N, D] array of positive activations
        neg_activations: [N, D] array of negative activations

    Returns:
        Dict of geometry metrics from wisent.core.reading.modules including:
        - signal_strength, linear_probe_accuracy, mlp_probe_accuracy
        - icd_* metrics (intrinsic concept dimensionality)
        - direction_* metrics (stability, consistency)
        - steer_* metrics (steerability analysis)
        - concept_coherence, n_concepts
        - recommended_method, recommendation_confidence
    """
    N = min(len(pos_activations), len(neg_activations))
    pos = pos_activations[:N]
    neg = neg_activations[:N]

    # Convert to torch tensors for wisent's geometry metrics
    pos_tensor = torch.tensor(pos, dtype=torch.float32)
    neg_tensor = torch.tensor(neg, dtype=torch.float32)

    # Use wisent's comprehensive geometry metrics
    metrics = wisent_compute_geometry_metrics(pos_tensor, neg_tensor)

    return metrics


def compute_steering_accuracy(
    train_pos: np.ndarray,
    train_neg: np.ndarray,
    test_pos: np.ndarray,
    test_neg: np.ndarray,
    method: str = "caa"
) -> float:
    """
    Compute steering accuracy using train/test split.

    Train a steering vector on train set, evaluate on test set.
    Accuracy = how often steering direction correctly classifies pos vs neg.
    """
    steering_vector = compute_steering_vector(train_pos, train_neg, method)

    if steering_vector is None:
        return 0.5

    # Normalize steering vector
    norm = np.linalg.norm(steering_vector)
    if norm < ZERO_THRESHOLD:
        return 0.5
    steering_vector = steering_vector / norm

    # Evaluate: does projection onto steering vector separate pos from neg?
    test_pos_proj = test_pos @ steering_vector
    test_neg_proj = test_neg @ steering_vector

    # Accuracy: positive examples should have higher projection
    correct = 0
    total = 0
    for p in test_pos_proj:
        for n in test_neg_proj:
            if p > n:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.5


def compute_steering_vector(
    train_pos: np.ndarray,
    train_neg: np.ndarray,
    method: str = "caa"
) -> Optional[np.ndarray]:
    """
    Compute steering vector using wisent's SteeringMethodRegistry.

    Args:
        train_pos: Training positive activations [N, D]
        train_neg: Training negative activations [N, D]
        method: One of the methods from SteeringMethodRegistry

    Returns:
        Steering vector [D] or None if failed
    """
    try:
        pos_tensor = torch.tensor(train_pos, dtype=torch.float32)
        neg_tensor = torch.tensor(train_neg, dtype=torch.float32)

        method_instance = SteeringMethodRegistry.create_method_instance(method)

        pos_list = [pos_tensor[i] for i in range(len(pos_tensor))]
        neg_list = [neg_tensor[i] for i in range(len(neg_tensor))]

        result = method_instance.train_for_layer(pos_list, neg_list)

        if isinstance(result, torch.Tensor):
            if result.dim() == 2:
                return result[0].cpu().numpy()
            return result.cpu().numpy()
        elif isinstance(result, dict):
            if "steering_vector" in result:
                return result["steering_vector"].cpu().numpy()
            elif "directions" in result:
                return result["directions"][0].cpu().numpy()
        return None

    except Exception as e:
        print(f"{method.upper()} failed: {e}")
        if method != "caa":
            return _compute_caa_fallback(train_pos, train_neg)
        return None


def _compute_caa_fallback(train_pos: np.ndarray, train_neg: np.ndarray) -> np.ndarray:
    """Simple CAA fallback if wisent method fails."""
    return train_pos.mean(axis=0) - train_neg.mean(axis=0)
