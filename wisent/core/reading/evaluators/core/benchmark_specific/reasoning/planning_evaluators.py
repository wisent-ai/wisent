"""Planning and table-answering evaluators for PlanBench and TAG benchmarks.

PlanBench: Plan validation with exact match for verification.
TAG-Bench: Exact match of generated answer against labeled correct answer.
"""

import re
import logging
from typing import Any

from wisent.core.reading.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.utils.config_tools.constants import (
    DISPLAY_TRUNCATION_ERROR,
    EVAL_CONFIDENCE_FULL,
    EVAL_CONFIDENCE_ZERO,
    EVAL_CONTRASTIVE_CORRECT_IDX,
    EVAL_CONTRASTIVE_INCORRECT_IDX,
    EVAL_NUM_CONTRASTIVE_PAIR_SIZE,
)

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_plan_text(text: str) -> str:
    """Normalize plan/verification text for comparison.

    Strips whitespace, lowercases, removes punctuation variations, and
    collapses multiple spaces to facilitate matching across formatting styles.
    """
    text = text.strip().lower()
    text = text.replace("\t", " ").replace("\r", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    text = text.rstrip(".,;:!?")
    return text


def _text_matches(response_text: str, expected_text: str) -> bool:
    """Check if normalized response matches normalized expected."""
    return _normalize_plan_text(response_text) == _normalize_plan_text(expected_text)


def _contrastive_text_match(evaluator, choices, expected):
    """Shared contrastive evaluation using normalized text matching."""
    correct_ans = str(choices[EVAL_CONTRASTIVE_CORRECT_IDX])
    incorrect_ans = str(choices[EVAL_CONTRASTIVE_INCORRECT_IDX])
    c_match = _text_matches(correct_ans, str(expected))
    i_match = _text_matches(incorrect_ans, str(expected))
    if c_match and not i_match:
        return EvalResult(
            ground_truth="TRUTHFUL", method_used=evaluator.name,
            confidence=EVAL_CONFIDENCE_FULL,
            details="Correct choice matches expected",
            meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans})
    if i_match and not c_match:
        return EvalResult(
            ground_truth="UNTRUTHFUL", method_used=evaluator.name,
            confidence=EVAL_CONFIDENCE_FULL,
            details="Incorrect choice matches expected",
            meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans})
    # Try substring containment as secondary strategy
    norm_exp = _normalize_plan_text(str(expected))
    norm_c = _normalize_plan_text(correct_ans)
    norm_i = _normalize_plan_text(incorrect_ans)
    c_contains = norm_exp in norm_c or norm_c in norm_exp
    i_contains = norm_exp in norm_i or norm_i in norm_exp
    if c_contains and not i_contains:
        return EvalResult(
            ground_truth="TRUTHFUL", method_used=evaluator.name,
            confidence=EVAL_CONFIDENCE_FULL,
            details="Correct choice contains/contained in expected",
            meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans})
    if i_contains and not c_contains:
        return EvalResult(
            ground_truth="UNTRUTHFUL", method_used=evaluator.name,
            confidence=EVAL_CONFIDENCE_FULL,
            details="Incorrect choice contains/contained in expected",
            meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans})
    return EvalResult(
        ground_truth="UNKNOWN", method_used=evaluator.name,
        confidence=EVAL_CONFIDENCE_ZERO,
        details="Neither choice clearly matches expected",
        meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans,
              "expected": str(expected)})


_PLANBENCH_JUDGE = (
    "You are a planning evaluator (per PlanBench methodology). "
    "Evaluate whether the generated plan is valid and achieves "
    "the goal from the given initial state.\n\n"
    "Problem (initial state + goal): {prompt}\n"
    "Expected plan/answer: {expected}\n"
    "Generated plan/answer: {response}\n\n"
    "For plan generation: is the plan valid (achieves the goal "
    "from the initial state using legal actions)?\n"
    "For plan verification: does the answer correctly identify "
    "whether the given plan is valid or not?\n\n"
    "Reply with exactly CORRECT or INCORRECT."
)


class PlanBenchEvaluator(BaseEvaluator):
    """PlanBench: plan validation via LLM judge.

    Paper: PlanBench. Method: VAL plan validator for generation,
    exact match for verification. Uses LLM-as-judge for plan
    validity assessment. Metric: Accuracy %.
    """

    name = "planbench"
    description = "PlanBench: plan validation via LLM judge"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        from wisent.core.reading.evaluators.benchmark_specific.judge_utils import (
            require_judge_model, call_judge, parse_binary_verdict,
        )
        model = require_judge_model(kwargs, self.name)
        question = kwargs.get("question", str(expected))
        prompt = _PLANBENCH_JUDGE.format(
            prompt=question, expected=str(expected), response=response)
        verdict = call_judge(model, prompt)
        gt = parse_binary_verdict(
            verdict, ("correct",), ("incorrect",))
        conf = EVAL_CONFIDENCE_FULL if gt != "UNKNOWN" else EVAL_CONFIDENCE_ZERO
        return EvalResult(
            ground_truth=gt, method_used=self.name, confidence=conf,
            details=f"Judge: {verdict[:DISPLAY_TRUNCATION_ERROR]}",
            meta={"judge_output": verdict, "expected": str(expected)})


class TagEvaluator(BaseEvaluator):
    """Evaluator for TAG-Bench (Table-Answering-Generation) benchmark.

    Paper: TAG-Bench. Exact match of generated answer against labeled
    correct answer after normalization. Metric: Exact match accuracy.
    """

    name = "tag"
    description = "TAG-Bench evaluator for table-answering exact match"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate model response against expected TAG answer."""
        choices = kwargs.get("choices")
        if choices and len(choices) == EVAL_NUM_CONTRASTIVE_PAIR_SIZE:
            return _contrastive_text_match(self, choices, expected)
        return self._evaluate_direct(response, expected)

    def _evaluate_direct(self, response: str, expected: Any) -> EvalResult:
        """Direct evaluation using exact match after normalization."""
        if not response or not response.strip():
            return EvalResult(
                ground_truth="UNKNOWN", method_used=self.name,
                confidence=EVAL_CONFIDENCE_ZERO,
                details="Empty response", meta={"expected": expected})
        if isinstance(expected, list):
            expected_answers = expected
        else:
            expected_answers = [expected]
        norm_resp = _normalize_plan_text(response)
        for exp in expected_answers:
            norm_exp = _normalize_plan_text(str(exp))
            if norm_resp == norm_exp:
                return EvalResult(
                    ground_truth="TRUTHFUL", method_used=self.name,
                    confidence=EVAL_CONFIDENCE_FULL,
                    details=f"Exact match: '{norm_resp}' == '{norm_exp}'",
                    meta={"matched_answer": exp})
        first_exp = _normalize_plan_text(
            str(expected_answers[EVAL_CONTRASTIVE_CORRECT_IDX]))
        return EvalResult(
            ground_truth="UNTRUTHFUL", method_used=self.name,
            confidence=EVAL_CONFIDENCE_ZERO,
            details=f"No match: '{norm_resp}' vs '{first_exp}'",
            meta={"response_norm": norm_resp, "expected": expected})
