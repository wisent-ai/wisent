"""Backward-compatible OptimizationCache class."""

from __future__ import annotations

from typing import Optional, Dict, Any, List

from ..convenience import get_config_manager, save_steering_config, get_steering_config
from .optimization_result import OptimizationResult, store_optimization, get_cached_optimization


class OptimizationCache:
    """Backward-compatible wrapper class for the unified config manager."""

    def __init__(self):
        self._manager = get_config_manager()
        self._defaults: Dict[str, str] = {}

    def _make_key(self, model: str, task: str, method: str = "CAA") -> str:
        model_normalized = model.replace("/", "_").replace("\\", "_")
        return f"{model_normalized}::{task}::{method}"

    def store(
        self, model: str, task: str, layer: int, strength: float,
        method: str = "CAA", token_aggregation: str = "average",
        prompt_strategy: str = "question_only", score: float = 0.0,
        metric: str = "accuracy", metadata: Optional[Dict[str, Any]] = None,
        set_as_default: bool = False,
    ) -> str:
        """Store an optimization result."""
        return store_optimization(
            model=model, task=task, layer=layer, strength=strength,
            method=method, token_aggregation=token_aggregation,
            prompt_strategy=prompt_strategy, score=score, metric=metric,
            metadata=metadata, set_as_default=set_as_default,
        )

    def get(self, model: str, task: str, method: str = "CAA") -> Optional[OptimizationResult]:
        """Get a cached optimization result."""
        return get_cached_optimization(model, task, method, use_default=False)

    def get_default(self, model: str, task: str) -> Optional[OptimizationResult]:
        """Get the default optimization result for a model/task."""
        return get_cached_optimization(model, task, "*", use_default=True)

    def set_default(self, model: str, task: str, method: str = "CAA") -> bool:
        """Set a cached result as the default."""
        steering = get_steering_config(model, task)
        if steering is None:
            return False
        save_steering_config(
            model_name=model, task_name=task, layer=steering.layer,
            strength=steering.strength, method=steering.method,
            token_aggregation=steering.token_aggregation,
            prompt_strategy=steering.prompt_strategy, score=steering.score,
            metric=steering.metric, set_as_default=True,
        )
        return True

    def exists(self, model: str, task: str, method: str = "CAA") -> bool:
        """Check if a cached result exists."""
        return get_cached_optimization(model, task, method, use_default=False) is not None

    def list_cached(
        self, model: Optional[str] = None, task: Optional[str] = None,
    ) -> List[OptimizationResult]:
        """List cached results, optionally filtered."""
        results = []
        models = [model] if model else self._manager.list_models()

        for m in models:
            config = self._manager.get_model_config(m)
            if config.default_steering and not task:
                results.append(OptimizationResult(
                    model=m, task="(default)", layer=config.default_steering.layer,
                    strength=config.default_steering.strength, method=config.default_steering.method,
                    token_aggregation=config.default_steering.token_aggregation,
                    prompt_strategy=config.default_steering.prompt_strategy,
                    score=config.default_steering.score, metric=config.default_steering.metric,
                ))
            for task_name, task_config in config.tasks.items():
                if task and task_name != task:
                    continue
                if task_config.steering:
                    results.append(OptimizationResult(
                        model=m, task=task_name, layer=task_config.steering.layer,
                        strength=task_config.steering.strength, method=task_config.steering.method,
                        token_aggregation=task_config.steering.token_aggregation,
                        prompt_strategy=task_config.steering.prompt_strategy,
                        score=task_config.steering.score, metric=task_config.steering.metric,
                    ))
        return results

    def delete(self, model: str, task: str, method: str = "CAA") -> bool:
        """Delete a cached result."""
        config = self._manager.get_model_config(model)
        if task in config.tasks and config.tasks[task].steering:
            config.tasks[task].steering = None
            self._manager._save_model_config(config)
            return True
        return False

    def clear(self) -> int:
        """Clear all cached steering results."""
        count = 0
        for model in self._manager.list_models():
            config = self._manager.get_model_config(model)
            if config.default_steering:
                config.default_steering = None
                count += 1
            for task_config in config.tasks.values():
                if task_config.steering:
                    task_config.steering = None
                    count += 1
            self._manager._save_model_config(config)
        return count

    def _save(self) -> None:
        """No-op for compatibility - config manager auto-saves."""
        pass


# Global cache instance for backward compatibility
_legacy_cache: Optional[OptimizationCache] = None


def get_cache() -> OptimizationCache:
    """Get the global optimization cache instance (backward compatible)."""
    global _legacy_cache
    if _legacy_cache is None:
        _legacy_cache = OptimizationCache()
    return _legacy_cache
