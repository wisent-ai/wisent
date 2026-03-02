"""Factory functions for custom evaluators.

Extracted from custom_evaluator.py to keep file under 300 lines.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict, Optional, Union

logger = logging.getLogger(__name__)


def create_custom_evaluator(
    evaluator_spec: Union[str, Callable, Any, Dict[str, Any]],
    **kwargs,
):
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
    """
    from wisent.core.reading.evaluators.custom.custom_evaluator import (
        CustomEvaluator, CallableEvaluator,
    )

    if isinstance(evaluator_spec, CustomEvaluator):
        return evaluator_spec

    if callable(evaluator_spec) and not isinstance(evaluator_spec, str):
        return CallableEvaluator(evaluator_spec, name="callable", **kwargs)

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
):
    """Load evaluator from a Python module path or file path."""
    import sys
    from pathlib import Path
    from wisent.core.reading.evaluators.custom.custom_evaluator import (
        CustomEvaluator, CallableEvaluator,
    )

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
