"""
Unified dataset splitting utilities.

This module provides consistent train/test splitting across all benchmarks,
regardless of their original split structure. All data is pooled together
and split using our own deterministic 80/20 split.
"""

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple


# Default split configuration
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_SEED = 42


def get_all_docs_from_task(task: Any) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Extract ALL documents from an lm-eval task, combining all available splits.

    Args:
        task: An lm-eval task object

    Returns:
        Tuple of (all_docs, split_counts) where split_counts shows how many
        docs came from each original split
    """
    all_docs = []
    split_counts = {}

    split_methods = [
        ("training_docs", "has_training_docs"),
        ("validation_docs", "has_validation_docs"),
        ("test_docs", "has_test_docs"),
        ("fewshot_docs", "has_fewshot_docs"),
    ]

    for docs_method, has_method in split_methods:
        if hasattr(task, has_method):
            try:
                has_docs = getattr(task, has_method)
                if callable(has_docs) and has_docs():
                    docs_iter = getattr(task, docs_method)()
                    if docs_iter is not None:
                        docs = list(docs_iter)
                        if docs:
                            split_counts[docs_method] = len(docs)
                            all_docs.extend(docs)
            except Exception:
                # Skip splits that fail to load
                continue

    return all_docs, split_counts


def create_deterministic_split(
    all_docs: List[Any],
    benchmark_name: str,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[Any], List[Any]]:
    """
    Create a deterministic train/test split from all documents.

    Uses benchmark name + seed to create reproducible shuffling,
    ensuring the same split every time for the same benchmark.

    Args:
        all_docs: All documents from the benchmark
        benchmark_name: Name of the benchmark (used for deterministic seeding)
        train_ratio: Ratio of data for training (default 0.8)
        seed: Base random seed (default 42)

    Returns:
        Tuple of (train_docs, test_docs)
    """
    if not all_docs:
        return [], []

    n = len(all_docs)

    # Create deterministic seed based on benchmark name
    combined_seed = int(hashlib.md5(
        f"{benchmark_name}_{seed}".encode()
    ).hexdigest()[:8], 16)

    # Shuffle indices deterministically
    rng = random.Random(combined_seed)
    indices = list(range(n))
    rng.shuffle(indices)

    # Split
    n_train = int(n * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_docs = [all_docs[i] for i in train_indices]
    test_docs = [all_docs[i] for i in test_indices]

    return train_docs, test_docs


def get_train_docs(
    task: Any,
    benchmark_name: Optional[str] = None,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """
    Get training documents from a task using our custom split.

    This combines all available splits and returns only the training portion.
    Use this for contrastive pair generation.

    Args:
        task: An lm-eval task object
        benchmark_name: Name of the benchmark (defaults to task name)
        train_ratio: Ratio of data for training (default 0.8)
        seed: Random seed for reproducibility

    Returns:
        List of training documents
    """
    if benchmark_name is None:
        benchmark_name = getattr(task, 'NAME', getattr(task, 'TASK_NAME', str(type(task).__name__)))

    all_docs, _ = get_all_docs_from_task(task)
    train_docs, _ = create_deterministic_split(all_docs, benchmark_name, train_ratio, seed)

    return train_docs


def get_test_docs(
    task: Any,
    benchmark_name: Optional[str] = None,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """
    Get test documents from a task using our custom split.

    This combines all available splits and returns only the test portion.
    Use this for evaluation.

    Args:
        task: An lm-eval task object
        benchmark_name: Name of the benchmark (defaults to task name)
        train_ratio: Ratio of data for training (default 0.8)
        seed: Random seed for reproducibility

    Returns:
        List of test documents
    """
    if benchmark_name is None:
        benchmark_name = getattr(task, 'NAME', getattr(task, 'TASK_NAME', str(type(task).__name__)))

    all_docs, _ = get_all_docs_from_task(task)
    _, test_docs = create_deterministic_split(all_docs, benchmark_name, train_ratio, seed)

    return test_docs


def get_split_info(
    task: Any,
    benchmark_name: Optional[str] = None,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Get information about the split for a task.

    Args:
        task: An lm-eval task object
        benchmark_name: Name of the benchmark
        train_ratio: Ratio of data for training
        seed: Random seed

    Returns:
        Dictionary with split information
    """
    if benchmark_name is None:
        benchmark_name = getattr(task, 'NAME', getattr(task, 'TASK_NAME', str(type(task).__name__)))

    all_docs, original_splits = get_all_docs_from_task(task)
    train_docs, test_docs = create_deterministic_split(all_docs, benchmark_name, train_ratio, seed)

    return {
        "benchmark_name": benchmark_name,
        "total_samples": len(all_docs),
        "train_samples": len(train_docs),
        "test_samples": len(test_docs),
        "train_ratio": train_ratio,
        "seed": seed,
        "original_splits": original_splits,
    }
