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

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult

__all__ = [
    "CustomEvaluator",
    "CallableEvaluator",
    "APIEvaluator",
    "create_custom_evaluator",
    "EvaluatorProtocol",
]

logger = logging.getLogger(__name__)


class EvaluatorProtocol(Protocol):
    """Protocol for custom evaluator functions.
    
    Custom evaluators must accept a response string and return either:
    - A float score in [0, 1] (higher = better)
    - A dict with at least a 'score' key
    
    Example:
        def my_evaluator(response: str) -> float:
            # Call your API, run your logic, etc.
            return 0.85
        
        def my_evaluator_detailed(response: str) -> dict:
            return {
                'score': 0.85,
                'confidence': 0.9,
                'details': 'Response is human-like'
            }
    """
    def __call__(self, response: str, **kwargs) -> Union[float, Dict[str, Any]]: ...


@dataclass
class CustomEvaluatorConfig:
    """Configuration for custom evaluators."""
    name: str = "custom"
    description: str = "Custom evaluator"
    invert_score: bool = False  # If True, higher original score = worse
    score_key: str = "score"  # Key to extract from dict responses
    min_score: float = 0.0
    max_score: float = 1.0
    normalize: bool = True  # Normalize to [0, 1]


class CustomEvaluator(ABC):
    """Abstract base class for custom evaluators.
    
    Subclass this to create custom evaluators that can be used with
    optimize-steering and optimize-weights commands.
    
    Example:
        class MyCustomEvaluator(CustomEvaluator):
            def __init__(self, api_key: str):
                super().__init__(name="my_evaluator", description="My API evaluator")
                self.api_key = api_key
            
            def evaluate_response(self, response: str, **kwargs) -> float:
                # Call your API
                result = my_api.analyze(response, key=self.api_key)
                return result['score']
    """
    
    def __init__(
        self,
        name: str = "custom",
        description: str = "Custom evaluator",
        config: Optional[CustomEvaluatorConfig] = None,
    ):
        self.name = name
        self.description = description
        self.config = config or CustomEvaluatorConfig(name=name, description=description)
    
    @abstractmethod
    def evaluate_response(self, response: str, **kwargs) -> Union[float, Dict[str, Any]]:
        """Evaluate a single response.
        
        Args:
            response: The model response to evaluate
            **kwargs: Additional context (prompt, expected, etc.)
        
        Returns:
            Either a float score in [0, 1] or a dict with 'score' key
        """
        pass
    
    def evaluate_batch(
        self,
        responses: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Evaluate a batch of responses.
        
        Default implementation calls evaluate_response for each.
        Override for batch optimization (e.g., batched API calls).
        
        Args:
            responses: List of model responses
            **kwargs: Additional context passed to each evaluation
        
        Returns:
            List of result dicts with at least 'score' key
        """
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
            score = (score - self.config.min_score) / (self.config.max_score - self.config.min_score)
            score = max(0.0, min(1.0, score))
        
        if self.config.invert_score:
            score = 1.0 - score
        
        result["score"] = score
        return result
    
    def __call__(self, response: str, **kwargs) -> Dict[str, Any]:
        """Make the evaluator callable."""
        result = self.evaluate_response(response, **kwargs)
        return self._normalize_result(result)


class CallableEvaluator(CustomEvaluator):
    """Wrap any callable as an evaluator.
    
    Example:
        def my_score_fn(response: str) -> float:
            return len(response) / 1000  # Score based on length
        
        evaluator = CallableEvaluator(my_score_fn)
        score = evaluator("Hello world")['score']
    """
    
    def __init__(
        self,
        fn: Callable[[str], Union[float, Dict[str, Any]]],
        name: str = "callable",
        description: str = "Callable evaluator",
        config: Optional[CustomEvaluatorConfig] = None,
    ):
        super().__init__(name=name, description=description, config=config)
        self._fn = fn
    
    def evaluate_response(self, response: str, **kwargs) -> Union[float, Dict[str, Any]]:
        return self._fn(response, **kwargs) if kwargs else self._fn(response)


class APIEvaluator(CustomEvaluator):
    """Base class for API-based evaluators.
    
    Provides common functionality for evaluators that call external APIs:
    - Rate limiting
    - Retries with exponential backoff
    - Response caching
    
    Subclass and implement _call_api() for your specific API.
    """
    
    def __init__(
        self,
        name: str = "api",
        description: str = "API-based evaluator",
        config: Optional[CustomEvaluatorConfig] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache_responses: bool = True,
    ):
        super().__init__(name=name, description=description, config=config)
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_responses = cache_responses
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    @abstractmethod
    def _call_api(self, response: str, **kwargs) -> Dict[str, Any]:
        """Make the actual API call.
        
        Override this in subclasses.
        
        Args:
            response: Text to analyze
            **kwargs: Additional parameters
        
        Returns:
            API response as dict
        """
        pass
    
    def evaluate_response(self, response: str, **kwargs) -> Dict[str, Any]:
        """Evaluate response with caching and retries."""
        import hashlib
        import time
        
        cache_key = hashlib.md5(response.encode()).hexdigest()
        if self.cache_responses and cache_key in self._cache:
            return self._cache[cache_key]
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = self._call_api(response, **kwargs)
                if self.cache_responses:
                    self._cache[cache_key] = result
                return result
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    time.sleep(delay)
        
        raise RuntimeError(f"API call failed after {self.max_retries} attempts: {last_error}")
    
    def evaluate_batch(self, responses: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Evaluate batch with optional batched API call.
        
        Override _call_api_batch() for APIs that support batch requests.
        """
        if hasattr(self, '_call_api_batch'):
            return self._call_api_batch(responses, **kwargs)
        return super().evaluate_batch(responses, **kwargs)


def create_custom_evaluator(
    evaluator_spec: Union[str, Callable, CustomEvaluator, Dict[str, Any]],
    **kwargs,
) -> CustomEvaluator:
    """Create a custom evaluator from various inputs.
    
    Args:
        evaluator_spec: One of:
            - String path to a Python module with 'evaluator' or 'create_evaluator' function
              e.g., "my_evaluators.gptzero" or "path/to/evaluator.py:my_fn"
            - A callable that takes response str and returns score
            - A CustomEvaluator instance
            - A dict with 'module' and optional 'function' keys
        **kwargs: Additional arguments passed to the evaluator
    
    Returns:
        CustomEvaluator instance
    
    Examples:
        # From callable
        evaluator = create_custom_evaluator(lambda r: len(r) / 100)
        
        # From module path
        evaluator = create_custom_evaluator("my_evaluators.gptzero")
        
        # From file path with function name
        evaluator = create_custom_evaluator("./my_eval.py:score_humanness")
        
        # From dict
        evaluator = create_custom_evaluator({
            "module": "my_evaluators.gptzero",
            "function": "create_evaluator",
            "api_key": "xxx"
        })
    """
    if isinstance(evaluator_spec, CustomEvaluator):
        return evaluator_spec
    
    if callable(evaluator_spec) and not isinstance(evaluator_spec, str):
        return CallableEvaluator(evaluator_spec, **kwargs)
    
    if isinstance(evaluator_spec, dict):
        module_path = evaluator_spec.get("module")
        function_name = evaluator_spec.get("function", "create_evaluator")
        eval_kwargs = {k: v for k, v in evaluator_spec.items() if k not in ("module", "function")}
        eval_kwargs.update(kwargs)
        return _load_evaluator_from_module(module_path, function_name, eval_kwargs)
    
    if isinstance(evaluator_spec, str):
        if ":" in evaluator_spec:
            module_path, function_name = evaluator_spec.rsplit(":", 1)
        else:
            module_path = evaluator_spec
            function_name = None
        return _load_evaluator_from_module(module_path, function_name, kwargs)
    
    raise ValueError(f"Invalid evaluator_spec type: {type(evaluator_spec)}")


def _load_evaluator_from_module(
    module_path: str,
    function_name: Optional[str],
    kwargs: Dict[str, Any],
) -> CustomEvaluator:
    """Load evaluator from a Python module path or file path."""
    import sys
    from pathlib import Path
    
    if module_path.endswith(".py") or "/" in module_path or "\\" in module_path:
        path = Path(module_path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluator file not found: {module_path}")
        
        spec = importlib.util.spec_from_file_location("custom_eval", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["custom_eval"] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)
    
    if function_name:
        fn = getattr(module, function_name)
    elif hasattr(module, "create_evaluator"):
        fn = module.create_evaluator
    elif hasattr(module, "evaluator"):
        fn = module.evaluator
    elif hasattr(module, "evaluate"):
        fn = module.evaluate
    else:
        fn_candidates = [
            name for name in dir(module)
            if callable(getattr(module, name))
            and not name.startswith("_")
            and "eval" in name.lower()
        ]
        if fn_candidates:
            fn = getattr(module, fn_candidates[0])
        else:
            raise AttributeError(
                f"Module {module_path} has no evaluator function. "
                "Define 'create_evaluator', 'evaluator', 'evaluate', or specify function name with ':'."
            )
    
    if isinstance(fn, type) and issubclass(fn, CustomEvaluator):
        return fn(**kwargs)
    
    if isinstance(fn, CustomEvaluator):
        return fn
    
    if callable(fn):
        try:
            result = fn(**kwargs)
            if isinstance(result, CustomEvaluator):
                return result
            if callable(result):
                return CallableEvaluator(result, name=module_path.split(".")[-1])
        except TypeError:
            pass
        return CallableEvaluator(fn, name=module_path.split(".")[-1])
    
    raise TypeError(f"Cannot create evaluator from {fn}")
