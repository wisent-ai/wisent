"""
Abstract base class for benchmark configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any


class BenchmarkConfig(ABC):
    """Abstract base class for benchmark configuration.

    Each benchmark should inherit from this class and implement:
    - get_benchmark_name(): Return benchmark name
    - get_data_config(): Return data split configuration
    - get_baseline_config(): Return baseline evaluation configuration
    - get_optimization_config(): Return optimization configuration
    """

    @staticmethod
    @abstractmethod
    def get_benchmark_name() -> str:
        """Return benchmark name.

        Returns:
            Benchmark name (e.g., "boolq", "cb", "gsm8k", "sst2")
        """
        pass

    @staticmethod
    @abstractmethod
    def get_data_config() -> Dict[str, Any]:
        """Return data configuration.

        Returns:
            Dictionary with data split sources (train_val_source, test_source)
        """
        pass

    @staticmethod
    @abstractmethod
    def get_baseline_config() -> Dict[str, Any]:
        """Return baseline evaluation configuration.

        Returns:
            Dictionary with baseline parameters (num_test, max_new_tokens)
        """
        pass

    @staticmethod
    @abstractmethod
    def get_optimization_config() -> Dict[str, Any]:
        """Return optimization configuration.

        Returns:
            Dictionary with optimization parameters (num_train, num_val, num_test, n_trials, n_runs)
        """
        pass
