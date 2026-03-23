"""Search evaluators for SealQA and BrowseComp benchmarks.

SealQA: LLM auto-rater for semantic correctness.
BrowseComp: LLM-as-judge binary CORRECT/INCORRECT.
"""

from typing import Any
import logging

from wisent.core.reading.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.utils.config_tools.constants import (
    EVAL_CONFIDENCE_FULL,
    EVAL_CONFIDENCE_ZERO,
    DISPLAY_TRUNCATION_MEDIUM,
)

logger = logging.getLogger(__name__)


class SealEvaluator(BaseEvaluator):
    """SealQA: semantic correctness via LLM auto-rater.

    Paper: SealQA. Metric: Accuracy.
    Method: LLM auto-rater (GPT-based) for semantic correctness.
    Requires 'judge_model' or 'model' kwarg.
    """

    name = "seal"
    description = "SealQA: semantic correctness via LLM judge"

    _JUDGE_PROMPT = (
        "You are evaluating search-augmented QA accuracy "
        "(per SealQA methodology).\n\n"
        "Expected answer: {expected}\n\n"
        "Model's answer: {response}\n\n"
        "Is the model's answer semantically correct? "
        "Reply with exactly CORRECT or INCORRECT."
    )

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        from wisent.core.reading.evaluators.benchmark_specific.judge_utils import (
            require_judge_model, call_judge, parse_binary_verdict,
        )
        model = require_judge_model(kwargs, self.name)
        prompt = self._JUDGE_PROMPT.format(
            expected=str(expected), response=response)
        verdict = call_judge(model, prompt)
        gt = parse_binary_verdict(
            verdict, ("correct",), ("incorrect",))
        conf = EVAL_CONFIDENCE_FULL if gt != "UNKNOWN" else EVAL_CONFIDENCE_ZERO
        return EvalResult(
            ground_truth=gt, method_used=self.name, confidence=conf,
            details=f"Judge: {verdict[:DISPLAY_TRUNCATION_MEDIUM]}",
            meta={"judge_output": verdict, "expected": str(expected)})


class BrowseCompEvaluator(BaseEvaluator):
    """Evaluator for BrowseComp benchmark.

    Paper: BrowseComp (OpenAI). Metric: Accuracy.
    Method: LLM-as-judge binary CORRECT/INCORRECT.
    Requires 'judge_model' or 'model' kwarg for judge evaluation.
    """

    name = "browsecomp"
    description = "BrowseComp: LLM-as-judge binary correctness"

    _JUDGE_PROMPT = (
        "You are evaluating whether a model's answer is correct.\n\n"
        "Question context and expected answer: {expected}\n\n"
        "Model's answer: {response}\n\n"
        "Is the model's answer correct? Reply with exactly "
        "CORRECT or INCORRECT."
    )

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        from wisent.core.reading.evaluators.benchmark_specific.judge_utils import (
            require_judge_model, call_judge, parse_binary_verdict,
        )
        model = require_judge_model(kwargs, self.name)
        prompt = self._JUDGE_PROMPT.format(
            expected=str(expected), response=response)
        verdict = call_judge(model, prompt)
        gt = parse_binary_verdict(
            verdict, ("correct",), ("incorrect",))
        conf = EVAL_CONFIDENCE_FULL if gt != "UNKNOWN" else EVAL_CONFIDENCE_ZERO
        return EvalResult(
            ground_truth=gt, method_used=self.name, confidence=conf,
            details=f"Judge: {verdict[:DISPLAY_TRUNCATION_MEDIUM]}",
            meta={"judge_output": verdict, "expected": str(expected)})
