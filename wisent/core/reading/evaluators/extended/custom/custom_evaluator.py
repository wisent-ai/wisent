"""
Custom evaluator classes for user-defined evaluation functions.

This module provides flexible evaluation interfaces:
- CustomEvaluator: Base class for custom evaluators
- CallableEvaluator: Wrap any Python callable as an evaluator
- APIEvaluator: Base class for API-based evaluators (e.g., GPTZero)
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from wisent.core.utils.config_tools.constants import SCORE_RANGE_MIN, SCORE_RANGE_MAX
from wisent.core.reading.evaluators.core.atoms import BaseEvaluator, EvalResult

# Re-export from helpers
from wisent.core.reading.evaluators.custom._custom_evaluator_helpers import (
    create_custom_evaluator,
)

__all__ = [
    "CustomEvaluator",
    "CallableEvaluator",
    "APIEvaluator",
    "create_custom_evaluator",
    "EvaluatorProtocol",
]

logger = logging.getLogger(__name__)


class EvaluatorProtocol(Protocol):
    """Protocol for custom evaluator functions."""
    def __call__(self, response: str, **kwargs) -> Union[float, Dict[str, Any]]: ...


@dataclass
class CustomEvaluatorConfig:
    """Configuration for custom evaluators."""
    name: Optional[str] = None
    description: str = "Custom evaluator"
    invert_score: bool = False
    score_key: Optional[str] = None
    normalize: bool = True


class CustomEvaluator(ABC):
    """Abstract base class for custom evaluators."""

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[CustomEvaluatorConfig] = None,
    ):
        self.name = name
        self.description = description
        self.config = config or CustomEvaluatorConfig(name=name, description=description)

    @abstractmethod
    def evaluate_response(self, response: str, **kwargs) -> Union[float, Dict[str, Any]]:
        """Evaluate a single response."""
        pass

    def evaluate_batch(
        self, responses: List[str], **kwargs,
    ) -> List[Dict[str, Any]]:
        """Evaluate a batch of responses."""
        results = []
        for response in responses:
            result = self.evaluate_response(response, **kwargs)
            if isinstance(result, (int, float)):
                result = {"score": float(result)}
            results.append(result)
        return results

    def _normalize_result(self, result: Union[float, Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize evaluation result to standard format."""
        if isinstance(result, (int, float)):
            score = float(result)
            result = {"score": score}
        else:
            score = result.get(self.config.score_key, result.get("score", 0.0))

        if self.config.normalize:
            score = (score - SCORE_RANGE_MIN) / (SCORE_RANGE_MAX - SCORE_RANGE_MIN)
            score = max(SCORE_RANGE_MIN, min(SCORE_RANGE_MAX, score))

        if self.config.invert_score:
            score = 1.0 - score

        result["score"] = score
        return result

    def __call__(self, response: str, **kwargs) -> Dict[str, Any]:
        """Make the evaluator callable."""
        result = self.evaluate_response(response, **kwargs)
        return self._normalize_result(result)


class CallableEvaluator(CustomEvaluator):
    """Wrap any callable as an evaluator."""

    def __init__(
        self,
        fn: Callable[[str], Union[float, Dict[str, Any]]],
        name: str,
        description: str = "Callable evaluator",
        config: Optional[CustomEvaluatorConfig] = None,
    ):
        super().__init__(name=name, description=description, config=config)
        self._fn = fn

    def evaluate_response(self, response: str, **kwargs) -> Union[float, Dict[str, Any]]:
        return self._fn(response, **kwargs) if kwargs else self._fn(response)


class APIEvaluator(CustomEvaluator):
    """Base class for API-based evaluators."""

    def __init__(
        self,
        name: str,
        description: str,
        api_timeout: int,
        max_retries: int,
        retry_delay: float,
        config: Optional[CustomEvaluatorConfig] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_responses: bool = True,
    ):
        super().__init__(name=name, description=description, config=config)
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = api_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_responses = cache_responses
        self._cache: Dict[str, Dict[str, Any]] = {}

    @abstractmethod
    def _call_api(self, response: str, **kwargs) -> Dict[str, Any]:
        """Make the actual API call. Override in subclasses."""
        pass

    def evaluate_response(self, response: str, **kwargs) -> Dict[str, Any]:
        """Evaluate response with caching and retries."""
        import hashlib
        import time

        cache_key = hashlib.md5(response.encode()).hexdigest()
        if self.cache_responses and cache_key in self._cache:
            return self._cache[cache_key]

        result = self._call_api(response, **kwargs)
        if self.cache_responses:
            self._cache[cache_key] = result
        return result

    def evaluate_batch(self, responses: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Evaluate batch with optional batched API call."""
        if hasattr(self, '_call_api_batch'):
            return self._call_api_batch(responses, **kwargs)
        return super().evaluate_batch(responses, **kwargs)
