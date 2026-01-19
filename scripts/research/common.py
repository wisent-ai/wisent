"""
Common utilities for research analysis.

Shared database access, data structures, and metric computation functions.
Uses wisent library functions where possible.
"""

import os
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import psycopg2
import torch

# Import wisent library functions
from wisent.core.steering_methods.registry import SteeringMethodRegistry, list_steering_methods
from wisent.core.geometry import compute_geometry_metrics as wisent_compute_geometry_metrics

# All available steering methods - from wisent registry
STEERING_METHODS = list_steering_methods()

# Database configuration - uses environment variables with fallback
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "aws-0-eu-west-2.pooler.supabase.com"),
    "port": int(os.environ.get("DB_PORT", 5432)),
    "database": os.environ.get("DB_NAME", "postgres"),
    "user": os.environ.get("DB_USER", "postgres.rbqjqnouluslojmmnuqi"),
    "password": os.environ.get("DB_PASSWORD", "BsKuEnPFLCFurN4a"),
    "options": "-c statement_timeout=0",  # No timeout
}

# Models to analyze
RESEARCH_MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-3.2-1B-Instruct",
    "openai/gpt-oss-20b",
    "Qwen/Qwen3-8B",
]


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get model info from database including number of layers."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute('SELECT id, "numLayers" FROM "Model" WHERE name = %s', (model_name,))
    result = cur.fetchone()
    cur.close()
    conn.close()

    if not result:
        raise ValueError(f"Model {model_name} not found in database")

    return {"id": result[0], "num_layers": result[1]}


def get_all_models_with_activations() -> List[Dict[str, Any]]:
    """Get all models that have activations in the database."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute('''
        SELECT DISTINCT m.name, m."numLayers"
        FROM "Model" m
        JOIN "Activation" a ON m.id = a."modelId"
    ''')
    results = cur.fetchall()
    cur.close()
    conn.close()

    return [{"name": r[0], "num_layers": r[1]} for r in results]


def bytes_to_vector(data: bytes) -> np.ndarray:
    """Convert binary data back to float vector."""
    num_floats = len(data) // 4
    return np.array(struct.unpack(f'{num_floats}f', data))


@dataclass
class ActivationData:
    """Container for activation data from a single pair."""
    pair_id: int
    set_id: int
    benchmark: str
    layer: int
    strategy: str
    positive_activation: np.ndarray
    negative_activation: np.ndarray


@dataclass
class BenchmarkResults:
    """Results for a single benchmark."""
    name: str
    num_pairs: int
    strategies: Dict[str, Dict[str, float]] = field(default_factory=dict)
    geometry_metrics: Dict[str, float] = field(default_factory=dict)
    best_strategy: str = ""
    best_accuracy: float = 0.0


def load_activations_from_db(model_name: str, layer: int = None) -> Dict[str, List[ActivationData]]:
    """
    Load activations from database grouped by benchmark.

    Args:
        model_name: Name of the model (e.g., 'Qwen/Qwen3-8B')
        layer: Layer to load (default: middle layer)

    Returns:
        Dict[benchmark_name -> List[ActivationData]]
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Get model ID
    cur.execute('SELECT id, "numLayers" FROM "Model" WHERE name = %s', (model_name,))
    result = cur.fetchone()
    if not result:
        raise ValueError(f"Model {model_name} not found in database")
    model_id, num_layers = result

    # Default to middle layer if not specified
    if layer is None:
        layer = num_layers // 2

    print(f"Loading activations for model {model_name} (id={model_id}), layer {layer}")

    # Query activations with benchmark info
    print(f"Querying activations for model_id={model_id}, layer={layer}...")
    cur.execute('''
        SELECT
            a."contrastivePairId",
            a."contrastivePairSetId",
            cps.name as benchmark,
            a.layer,
            a."extractionStrategy",
            a."activationData",
            a."isPositive"
        FROM "Activation" a
        JOIN "ContrastivePairSet" cps ON a."contrastivePairSetId" = cps.id
        WHERE a."modelId" = %s AND a.layer = %s
        ORDER BY a."contrastivePairSetId", a."contrastivePairId", a."extractionStrategy"
    ''', (model_id, layer))

    rows = cur.fetchall()
    print(f"Fetched {len(rows)} activation rows")

    # Group by benchmark and pair
    raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for row in rows:
        pair_id, set_id, benchmark, layer, strategy, activation_bytes, is_positive = row
        key = "positive" if is_positive else "negative"
        raw_data[benchmark][(pair_id, set_id)][strategy][key] = bytes_to_vector(activation_bytes)

    # Convert to ActivationData objects
    result = defaultdict(list)
    for benchmark, pairs in raw_data.items():
        for (pair_id, set_id), strategies in pairs.items():
            for strategy, activations in strategies.items():
                if "positive" in activations and "negative" in activations:
                    result[benchmark].append(ActivationData(
                        pair_id=pair_id,
                        set_id=set_id,
                        benchmark=benchmark,
                        layer=layer,
                        strategy=strategy,
                        positive_activation=activations["positive"],
                        negative_activation=activations["negative"],
                    ))

    cur.close()
    conn.close()

    return dict(result)


def compute_geometry_metrics(pos_activations: np.ndarray, neg_activations: np.ndarray, include_expensive: bool = False) -> Dict[str, float]:
    """
    Compute geometry metrics using wisent's comprehensive geometry analysis.

    Args:
        pos_activations: [N, D] array of positive activations
        neg_activations: [N, D] array of negative activations
        include_expensive: Whether to include computationally expensive metrics

    Returns:
        Dict of geometry metrics from wisent.core.geometry including:
        - signal_strength, linear_probe_accuracy, mlp_probe_accuracy
        - icd_* metrics (intrinsic concept dimensionality)
        - direction_* metrics (stability, consistency)
        - steer_* metrics (steerability analysis)
        - concept_coherence, n_concepts
        - And if include_expensive: mmd_rbf, density_ratio, fisher_*, intrinsic_dim_*, knn_*, signal_to_noise
        - recommended_method, recommendation_confidence
    """
    N = min(len(pos_activations), len(neg_activations))
    pos = pos_activations[:N]
    neg = neg_activations[:N]

    # Convert to torch tensors for wisent's geometry metrics
    pos_tensor = torch.tensor(pos, dtype=torch.float32)
    neg_tensor = torch.tensor(neg, dtype=torch.float32)

    # Use wisent's comprehensive geometry metrics
    metrics = wisent_compute_geometry_metrics(pos_tensor, neg_tensor, include_expensive=include_expensive)

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

    Args:
        train_pos: Training positive activations
        train_neg: Training negative activations
        test_pos: Test positive activations
        test_neg: Test negative activations
        method: One of "caa", "hyperplane", "mlp", "prism", "pulse", "titan"

    Returns:
        Pairwise accuracy (probability that pos > neg on steering direction)
    """
    steering_vector = compute_steering_vector(train_pos, train_neg, method)

    if steering_vector is None:
        return 0.5

    # Normalize steering vector
    norm = np.linalg.norm(steering_vector)
    if norm < 1e-10:
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
        method: One of the methods from SteeringMethodRegistry (caa, hyperplane, mlp, prism, pulse, titan)

    Returns:
        Steering vector [D] or None if failed
    """
    try:
        # Convert numpy arrays to torch tensors
        pos_tensor = torch.tensor(train_pos, dtype=torch.float32)
        neg_tensor = torch.tensor(train_neg, dtype=torch.float32)

        # Create method instance using wisent's registry
        method_instance = SteeringMethodRegistry.create_method_instance(method)

        # Train for layer - pass as lists of tensors
        pos_list = [pos_tensor[i] for i in range(len(pos_tensor))]
        neg_list = [neg_tensor[i] for i in range(len(neg_tensor))]

        result = method_instance.train_for_layer(pos_list, neg_list)

        # Handle different return types from different methods
        if isinstance(result, torch.Tensor):
            # PRISM/TITAN may return multiple directions - take first
            if result.dim() == 2:
                return result[0].cpu().numpy()
            return result.cpu().numpy()
        elif isinstance(result, dict):
            # Some methods return dicts with steering_vector or directions
            if "steering_vector" in result:
                return result["steering_vector"].cpu().numpy()
            elif "directions" in result:
                return result["directions"][0].cpu().numpy()
        return None

    except Exception as e:
        print(f"{method.upper()} failed: {e}")
        # Fallback to CAA if method fails
        if method != "caa":
            return _compute_caa_fallback(train_pos, train_neg)
        return None


def _compute_caa_fallback(train_pos: np.ndarray, train_neg: np.ndarray) -> np.ndarray:
    """Simple CAA fallback if wisent method fails."""
    return train_pos.mean(axis=0) - train_neg.mean(axis=0)
