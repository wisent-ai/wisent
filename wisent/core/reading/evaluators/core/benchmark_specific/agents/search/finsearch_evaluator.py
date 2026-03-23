"""FinSearchComp evaluator for financial search benchmark.

Paper: FinSearchComp. Metric: Accuracy.
Method: LLM-as-judge with rubric-guided judging, tolerance bands
for numerical answers. Requires 'judge_model' or 'model' kwarg.
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

_JUDGE_PROMPT = (
    "You are a financial expert evaluating answer accuracy.\n\n"
    "Expected answer: {expected}\n\n"
    "Model's answer: {response}\n\n"
    "Evaluation rubric:\n"
    "- For numerical answers: the answer is CORRECT if within a "
    "reasonable tolerance band of the expected value (typically "
    "within a few percent for financial figures)\n"
    "- For textual answers: the answer is CORRECT if it conveys "
    "the same information as the expected answer\n"
    "- Partial answers or answers with minor formatting differences "
    "should be considered CORRECT\n\n"
    "Reply with exactly CORRECT or INCORRECT."
)


class FinSearchCompEvaluator(BaseEvaluator):
    """FinSearchComp: LLM-as-judge with rubric-guided financial judging.

    Paper: FinSearchComp. Metric: Accuracy.
    Method: LLM-as-judge with rubric-guided judging and tolerance
    bands for numerical financial answers.
    """

    name = "finsearchcomp"
    description = "FinSearchComp: LLM-as-judge rubric-guided financial evaluation"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        from wisent.core.reading.evaluators.benchmark_specific.judge_utils import (
            require_judge_model, call_judge, parse_binary_verdict,
        )
        model = require_judge_model(kwargs, self.name)
        prompt = _JUDGE_PROMPT.format(
            expected=str(expected), response=response)
        verdict = call_judge(model, prompt)
        gt = parse_binary_verdict(
            verdict, ("correct",), ("incorrect",))
        conf = EVAL_CONFIDENCE_FULL if gt != "UNKNOWN" else EVAL_CONFIDENCE_ZERO
        return EvalResult(
            ground_truth=gt, method_used=self.name, confidence=conf,
            details=f"Judge: {verdict[:DISPLAY_TRUNCATION_MEDIUM]}",
            meta={"judge_output": verdict, "expected": str(expected)})
