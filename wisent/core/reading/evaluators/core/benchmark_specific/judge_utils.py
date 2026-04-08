"""Shared LLM-as-judge utilities for benchmark evaluators.

Provides common infrastructure for evaluators that need to call a model
as a judge during evaluation. Follows the LiveMathBench judge pattern
established in the codebase.
"""

import logging
from typing import Any

from wisent.core.reading.evaluators.core.atoms import EvalResult, EvaluatorError
from wisent.core.utils.infra_tools.errors.error_handler import ModelNotProvidedError
from wisent.core.utils.config_tools.constants import (
    EVAL_CONFIDENCE_FULL,
    EVAL_CONFIDENCE_ZERO,
    DISPLAY_TRUNCATION_MEDIUM,
    INDEX_FIRST,
)

logger = logging.getLogger(__name__)


def require_judge_model(kwargs: dict, evaluator_name: str) -> Any:
    """Extract the judge model from kwargs or raise ModelNotProvidedError.

    Checks for 'judge_model' first, then falls back to 'model'.
    Raises ModelNotProvidedError so callers can distinguish "missing model"
    from a real evaluation failure (it bubbles up as a SKIP, not a FAIL).
    """
    model = kwargs.get("judge_model") or kwargs.get("model")
    if model is None:
        task_name = kwargs.get("task_name", "unknown")
        raise ModelNotProvidedError(evaluator_name=evaluator_name, task_name=task_name)
    return model


def call_judge(model: Any, prompt: str) -> str:
    """Call a model to generate a judge verdict.

    Args:
        model: WisentModel or compatible model with generate method.
        prompt: The judge prompt to send to the model.

    Returns:
        The generated verdict text.
    """
    result = model.generate(
        inputs=prompt,
        do_sample=False,
        prompt_is_formatted=True,
    )
    if isinstance(result, list):
        return result[INDEX_FIRST] if result else ""
    return str(result)


def parse_binary_verdict(
    verdict: str,
    positive_labels: tuple[str, ...],
    negative_labels: tuple[str, ...],
) -> str:
    """Parse a binary verdict from judge output.

    Args:
        verdict: Raw judge output text.
        positive_labels: Strings that indicate positive/correct judgment.
        negative_labels: Strings that indicate negative/incorrect judgment.

    Returns:
        "TRUTHFUL", "UNTRUTHFUL", or "UNKNOWN".
    """
    lower = verdict.lower().strip()
    for label in positive_labels:
        if label.lower() in lower:
            return "TRUTHFUL"
    for label in negative_labels:
        if label.lower() in lower:
            return "UNTRUTHFUL"
    return "UNKNOWN"


def judge_eval_result(
    evaluator_name: str,
    response: str,
    expected: Any,
    model: Any,
    judge_prompt: str,
    positive_labels: tuple[str, ...],
    negative_labels: tuple[str, ...],
) -> EvalResult:
    """Full LLM-as-judge evaluation in one call.

    Constructs prompt, calls judge, parses verdict, returns EvalResult.
    """
    verdict_text = call_judge(model, judge_prompt)
    ground_truth = parse_binary_verdict(
        verdict_text, positive_labels, negative_labels)
    if ground_truth == "UNKNOWN":
        confidence = EVAL_CONFIDENCE_ZERO
    else:
        confidence = EVAL_CONFIDENCE_FULL
    return EvalResult(
        ground_truth=ground_truth,
        method_used=evaluator_name,
        confidence=confidence,
        details=f"Judge verdict: {verdict_text[:DISPLAY_TRUNCATION_MEDIUM]}",
        meta={
            "judge_output": verdict_text,
            "expected": str(expected),
            "response_preview": response[:DISPLAY_TRUNCATION_MEDIUM]
            if response else "",
        },
    )
