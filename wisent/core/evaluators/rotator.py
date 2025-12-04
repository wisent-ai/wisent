"""
Evaluator rotator for discovering and selecting evaluators.

Uses BaseRotator for common plugin discovery and resolution logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult, EvaluatorError
from wisent.core.utils.base_rotator import BaseRotator

__all__ = [
    "EvaluatorRotator",
]

logger = logging.getLogger(__name__)


class EvaluatorRotator(BaseRotator[BaseEvaluator]):
    """
    Orchestrates evaluator selection and execution with flexible discovery.

    Extends BaseRotator with evaluator-specific functionality:
    - Auto-selection based on task_name via extractor registry
    - Batch evaluation support
    - Task-aware evaluation
    """

    def __init__(
        self,
        evaluator: Union[str, BaseEvaluator, Type[BaseEvaluator], None] = None,
        task_name: Optional[str] = None,
        evaluators_location: Union[str, Path] = "wisent.core.evaluators.oracles",
        autoload: bool = True,
    ) -> None:
        """
        Initialize the evaluator rotator.

        Args:
            evaluator: Evaluator name, instance, class, or None for auto-selection.
            task_name: Optional task name for auto-selecting evaluator.
            evaluators_location: Module path or directory for evaluator discovery.
            autoload: Whether to auto-discover evaluators on init.
        """
        self._task_name = task_name

        # Initialize base class (will call _resolve via super().__init__)
        super().__init__(
            plugin=evaluator,
            location=evaluators_location,
            autoload=autoload,
        )

    def _get_registry_class(self) -> Type[BaseEvaluator]:
        return BaseEvaluator

    def _get_error_class(self) -> Type[Exception]:
        return EvaluatorError

    def _get_plugin_type_name(self) -> str:
        return "evaluator"

    # Keep static method for backward compatibility
    @staticmethod
    def discover_evaluators(location: Union[str, Path] = "wisent.core.evaluators.oracles") -> None:
        """
        Import all evaluator modules so BaseEvaluator subclasses self-register.

        Static method for backward compatibility.
        """
        # Create temporary instance just for discovery
        rotator = EvaluatorRotator.__new__(EvaluatorRotator)
        rotator._scope_prefix = ""
        rotator.discover(location)

    @staticmethod
    def list_evaluators() -> List[Dict[str, Any]]:
        """List all registered evaluators with their metadata."""
        out: List[Dict[str, Any]] = []
        for name, cls in BaseEvaluator.list_registered().items():
            out.append(
                {
                    "name": name,
                    "description": getattr(cls, "description", ""),
                    "task_names": list(getattr(cls, "task_names", ())),
                    "class": f"{cls.__module__}.{cls.__name__}",
                }
            )
        return sorted(out, key=lambda x: x["name"])

    def _resolve(
        self,
        plugin: Union[str, BaseEvaluator, Type[BaseEvaluator], None],
        **kwargs: Any,
    ) -> BaseEvaluator:
        """
        Resolve evaluator with task-based auto-selection support.

        Overrides base _resolve to add task_name-based auto-selection.
        """
        if plugin is None and self._task_name:
            # Auto-select based on task_name via extractor registry
            return self._auto_select_for_task()

        if plugin is None and not self._task_name:
            raise EvaluatorError(
                "No evaluator specified and no task_name provided. "
                "Either provide an evaluator name or a task_name for auto-selection."
            )

        # Use base class resolution for non-None cases
        return super()._resolve(plugin, **kwargs)

    def _auto_select_for_task(self) -> BaseEvaluator:
        """Auto-select evaluator based on task_name via extractor registry."""
        from wisent.core.contrastive_pairs.lm_eval_pairs.lm_extractor_registry import (
            get_extractor as get_lm_extractor,
            UnsupportedLMEvalBenchmarkError
        )
        from wisent.core.contrastive_pairs.huggingface_pairs.hf_extractor_registry import (
            get_extractor as get_hf_extractor,
            UnsupportedHuggingFaceBenchmarkError
        )

        extractor = None
        try:
            extractor = get_lm_extractor(self._task_name)
        except UnsupportedLMEvalBenchmarkError:
            try:
                extractor = get_hf_extractor(self._task_name)
            except UnsupportedHuggingFaceBenchmarkError:
                pass

        if extractor is None:
            raise EvaluatorError(
                f"No extractor registered for task '{self._task_name}'. "
                "Cannot auto-select evaluator."
            )

        # Get evaluator name from extractor
        evaluator_name = None
        if hasattr(extractor, 'evaluator_name'):
            evaluator_name = extractor.evaluator_name
        elif hasattr(extractor.__class__, '__module__'):
            try:
                import sys
                mod = sys.modules.get(extractor.__class__.__module__)
                if mod and hasattr(mod, 'evaluator_name'):
                    evaluator_name = mod.evaluator_name
            except Exception:
                pass

        if not evaluator_name:
            raise EvaluatorError(
                f"No evaluator_name defined for task '{self._task_name}'. "
                f"Extractor class: {extractor.__class__.__name__} "
                f"from module: {extractor.__class__.__module__}. "
                f"Please add 'evaluator_name' attribute to the extractor class."
            )

        logger.info(
            f"Auto-selected evaluator '{evaluator_name}' for task "
            f"'{self._task_name}' (from extractor)"
        )
        cls = BaseEvaluator.get(evaluator_name)
        return cls()

    def use(self, evaluator: Union[str, BaseEvaluator, Type[BaseEvaluator]]) -> None:
        """Switch to a different evaluator."""
        self._plugin = self._resolve(evaluator)

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """
        Evaluate a single response.

        Args:
            response: Model response to evaluate.
            expected: Expected/ground truth answer.
            **kwargs: Additional kwargs passed to evaluator.

        Returns:
            EvalResult with evaluation outcome.
        """
        kwargs.setdefault("task_name", self._task_name)
        return self._plugin.evaluate(response, expected, **kwargs)

    def evaluate_batch(
        self, responses: Sequence[str], expected_answers: Sequence[Any], **kwargs
    ) -> List[EvalResult]:
        """
        Evaluate a batch of responses.

        Args:
            responses: List of model responses.
            expected_answers: List of expected answers.
            **kwargs: Additional kwargs passed to evaluator.

        Returns:
            List of EvalResult objects.
        """
        kwargs.setdefault("task_name", self._task_name)
        return self._plugin.evaluate_batch(responses, expected_answers, **kwargs)


if __name__ == "__main__":
    rot = EvaluatorRotator(
        evaluators_location="wisent.core.evaluators.oracles",
        autoload=True,
    )

    print("Available evaluators:")
    for ev in rot.list_evaluators():
        print(f" - {ev['name']}: {ev['description']} (tasks: {', '.join(ev['task_names'])})")
