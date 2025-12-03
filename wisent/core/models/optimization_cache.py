"""
Optimization cache for storing and retrieving optimal steering parameters.

Stores optimal configurations from optimization runs, allowing:
- Caching results to avoid re-running expensive optimizations
- Setting default parameters for specific model/task combinations
- Retrieving cached parameters via CLI or programmatically
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


# Default cache location
DEFAULT_CACHE_DIR = os.path.expanduser("~/.wisent/optimization_cache")


@dataclass
class OptimizationResult:
    """Represents a cached optimization result."""
    model: str
    task: str
    layer: int
    strength: float
    method: str = "CAA"
    token_aggregation: str = "average"
    prompt_strategy: str = "question_only"
    score: float = 0.0
    metric: str = "accuracy"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class OptimizationCache:
    """Manages cached optimization results."""

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "optimization_cache.json"
        self.defaults_file = self.cache_dir / "defaults.json"
        self._cache: Dict[str, OptimizationResult] = {}
        self._defaults: Dict[str, str] = {}  # Maps model/task key to cache key
        self._load()

    def _make_key(self, model: str, task: str, method: str = "CAA") -> str:
        """Create a unique key for a model/task/method combination."""
        # Normalize model name (remove slashes)
        model_normalized = model.replace("/", "_").replace("\\", "_")
        return f"{model_normalized}::{task}::{method}"

    def _load(self) -> None:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    data = json.load(f)
                self._cache = {
                    k: OptimizationResult.from_dict(v)
                    for k, v in data.items()
                }
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load optimization cache: {e}")
                self._cache = {}

        if self.defaults_file.exists():
            try:
                with open(self.defaults_file) as f:
                    self._defaults = json.load(f)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load defaults: {e}")
                self._defaults = {}

    def _save(self) -> None:
        """Save cache to disk."""
        with open(self.cache_file, "w") as f:
            json.dump(
                {k: v.to_dict() for k, v in self._cache.items()},
                f,
                indent=2
            )

        with open(self.defaults_file, "w") as f:
            json.dump(self._defaults, f, indent=2)

    def store(
        self,
        model: str,
        task: str,
        layer: int,
        strength: float,
        method: str = "CAA",
        token_aggregation: str = "average",
        prompt_strategy: str = "question_only",
        score: float = 0.0,
        metric: str = "accuracy",
        metadata: Optional[Dict[str, Any]] = None,
        set_as_default: bool = False
    ) -> str:
        """
        Store an optimization result in the cache.

        Args:
            model: Model name/path
            task: Task name
            layer: Optimal layer
            strength: Optimal strength
            method: Steering method (CAA, DAC, etc.)
            token_aggregation: Token aggregation strategy
            prompt_strategy: Prompt construction strategy
            score: Achieved score
            metric: Metric used for optimization
            metadata: Additional metadata to store
            set_as_default: Whether to set this as the default for model/task

        Returns:
            Cache key for the stored result
        """
        key = self._make_key(model, task, method)

        result = OptimizationResult(
            model=model,
            task=task,
            layer=layer,
            strength=strength,
            method=method,
            token_aggregation=token_aggregation,
            prompt_strategy=prompt_strategy,
            score=score,
            metric=metric,
            metadata=metadata or {}
        )

        self._cache[key] = result

        if set_as_default:
            default_key = self._make_key(model, task, "*")  # Wildcard for method
            self._defaults[default_key] = key

        self._save()
        return key

    def get(
        self,
        model: str,
        task: str,
        method: str = "CAA"
    ) -> Optional[OptimizationResult]:
        """
        Retrieve a cached optimization result.

        Args:
            model: Model name/path
            task: Task name
            method: Steering method

        Returns:
            OptimizationResult if found, None otherwise
        """
        key = self._make_key(model, task, method)
        return self._cache.get(key)

    def get_default(
        self,
        model: str,
        task: str
    ) -> Optional[OptimizationResult]:
        """
        Get the default optimization result for a model/task.

        Args:
            model: Model name/path
            task: Task name

        Returns:
            Default OptimizationResult if set, None otherwise
        """
        default_key = self._make_key(model, task, "*")
        if default_key in self._defaults:
            cache_key = self._defaults[default_key]
            return self._cache.get(cache_key)
        return None

    def set_default(self, model: str, task: str, method: str = "CAA") -> bool:
        """
        Set an existing cached result as the default.

        Args:
            model: Model name/path
            task: Task name
            method: Steering method of the result to set as default

        Returns:
            True if successful, False if result not found
        """
        key = self._make_key(model, task, method)
        if key not in self._cache:
            return False

        default_key = self._make_key(model, task, "*")
        self._defaults[default_key] = key
        self._save()
        return True

    def exists(self, model: str, task: str, method: str = "CAA") -> bool:
        """Check if a cached result exists."""
        key = self._make_key(model, task, method)
        return key in self._cache

    def list_cached(
        self,
        model: Optional[str] = None,
        task: Optional[str] = None
    ) -> List[OptimizationResult]:
        """
        List cached results, optionally filtered by model/task.

        Args:
            model: Filter by model (optional)
            task: Filter by task (optional)

        Returns:
            List of matching OptimizationResults
        """
        results = []
        for result in self._cache.values():
            if model and result.model != model:
                continue
            if task and result.task != task:
                continue
            results.append(result)
        return results

    def delete(self, model: str, task: str, method: str = "CAA") -> bool:
        """
        Delete a cached result.

        Args:
            model: Model name/path
            task: Task name
            method: Steering method

        Returns:
            True if deleted, False if not found
        """
        key = self._make_key(model, task, method)
        if key in self._cache:
            del self._cache[key]

            # Also remove from defaults if it was the default
            default_key = self._make_key(model, task, "*")
            if self._defaults.get(default_key) == key:
                del self._defaults[default_key]

            self._save()
            return True
        return False

    def clear(self) -> int:
        """
        Clear all cached results.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache = {}
        self._defaults = {}
        self._save()
        return count


# Global cache instance
_cache: Optional[OptimizationCache] = None


def get_cache() -> OptimizationCache:
    """Get the global optimization cache instance."""
    global _cache
    if _cache is None:
        _cache = OptimizationCache()
    return _cache


def store_optimization(
    model: str,
    task: str,
    layer: int,
    strength: float,
    method: str = "CAA",
    token_aggregation: str = "average",
    prompt_strategy: str = "question_only",
    score: float = 0.0,
    metric: str = "accuracy",
    metadata: Optional[Dict[str, Any]] = None,
    set_as_default: bool = False
) -> str:
    """Convenience function to store an optimization result."""
    return get_cache().store(
        model=model,
        task=task,
        layer=layer,
        strength=strength,
        method=method,
        token_aggregation=token_aggregation,
        prompt_strategy=prompt_strategy,
        score=score,
        metric=metric,
        metadata=metadata,
        set_as_default=set_as_default
    )


def get_cached_optimization(
    model: str,
    task: str,
    method: str = "CAA",
    use_default: bool = True
) -> Optional[OptimizationResult]:
    """
    Convenience function to get a cached optimization result.

    Args:
        model: Model name/path
        task: Task name
        method: Steering method
        use_default: If True and exact match not found, try to get default

    Returns:
        OptimizationResult if found, None otherwise
    """
    cache = get_cache()
    result = cache.get(model, task, method)
    if result is None and use_default:
        result = cache.get_default(model, task)
    return result
