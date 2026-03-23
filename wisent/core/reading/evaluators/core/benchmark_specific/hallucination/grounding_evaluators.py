"""Hallucination grounding evaluators using LLM-as-judge.

- FaithBench: hallucination span classification via LLM judge
- FACTS Grounding: binary grounding check via LLM judge
- Chinese SimpleQA: correctness classification via LLM judge
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

_FAITHBENCH_JUDGE = (
    "You are a hallucination classifier (per FaithBench methodology). "
    "Classify whether the response contains hallucinated content "
    "relative to the source material.\n\n"
    "Source/expected: {expected}\n\n"
    "Response to evaluate: {response}\n\n"
    "Hallucination categories: factual fabrication, entity error, "
    "relation error, or sentence-level contradiction.\n"
    "Is the response FAITHFUL (no hallucinations) or "
    "HALLUCINATED (contains hallucinations)? Reply with exactly one."
)

_FACTS_JUDGE = (
    "You are a grounding verifier (per FACTS Grounding methodology). "
    "Determine if the response is properly grounded in the provided "
    "source material -- every claim must be supported.\n\n"
    "Source material: {expected}\n\n"
    "Response: {response}\n\n"
    "Is the response GROUNDED (all claims supported by source) or "
    "UNGROUNDED (contains unsupported claims)? Reply exactly one."
)

_CHINESE_SIMPLEQA_JUDGE = (
    "You are evaluating factual accuracy (per Chinese SimpleQA). "
    "Compare the model's answer against the reference answer.\n\n"
    "Reference answer: {expected}\n\n"
    "Model's answer: {response}\n\n"
    "Classify as:\n"
    "- CORRECT: The answer matches the reference\n"
    "- NOT_ATTEMPTED: The model refused or said it does not know\n"
    "- INCORRECT: The answer contradicts the reference\n\n"
    "Reply with exactly CORRECT, NOT_ATTEMPTED, or INCORRECT."
)


class FaithBenchEvaluator(BaseEvaluator):
    """FaithBench: hallucination span classification via LLM judge.

    Paper: FaithBench. Method: Classify hallucinated spans into
    categories with worst-pooled scoring. Uses LLM-as-judge.
    Metric: Balanced Accuracy, F-score Macro.
    """

    name = "faithbench"
    description = "FaithBench: hallucination classification via LLM judge"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        from wisent.core.reading.evaluators.benchmark_specific.judge_utils import (
            require_judge_model, call_judge, parse_binary_verdict,
        )
        model = require_judge_model(kwargs, self.name)
        prompt = _FAITHBENCH_JUDGE.format(
            expected=str(expected), response=response)
        verdict = call_judge(model, prompt)
        gt = parse_binary_verdict(
            verdict, ("faithful",), ("hallucinated",))
        conf = EVAL_CONFIDENCE_FULL if gt != "UNKNOWN" else EVAL_CONFIDENCE_ZERO
        return EvalResult(
            ground_truth=gt, method_used=self.name, confidence=conf,
            details=f"Judge: {verdict[:DISPLAY_TRUNCATION_MEDIUM]}",
            meta={"judge_output": verdict, "expected": str(expected)})


class FACTSGroundingEvaluator(BaseEvaluator):
    """FACTS Grounding: binary grounding check via LLM judge.

    Paper: FACTS Grounding. Method: Two-phase LLM-as-judge with
    frontier models for binary grounding verification.
    Metric: Average judge scores.
    """

    name = "facts_grounding"
    description = "FACTS Grounding: binary grounding via LLM judge"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        from wisent.core.reading.evaluators.benchmark_specific.judge_utils import (
            require_judge_model, call_judge, parse_binary_verdict,
        )
        model = require_judge_model(kwargs, self.name)
        prompt = _FACTS_JUDGE.format(
            expected=str(expected), response=response)
        verdict = call_judge(model, prompt)
        gt = parse_binary_verdict(
            verdict, ("grounded",), ("ungrounded",))
        conf = EVAL_CONFIDENCE_FULL if gt != "UNKNOWN" else EVAL_CONFIDENCE_ZERO
        return EvalResult(
            ground_truth=gt, method_used=self.name, confidence=conf,
            details=f"Judge: {verdict[:DISPLAY_TRUNCATION_MEDIUM]}",
            meta={"judge_output": verdict, "expected": str(expected)})


class ChineseSimpleQAEvaluator(BaseEvaluator):
    """Chinese SimpleQA: correctness classification via LLM judge.

    Paper: Chinese SimpleQA. Method: LLM-as-judge classifies
    into Correct/Not Attempted/Incorrect.
    Metric: CO rate, CGA, F-score.
    """

    name = "chinese_simpleqa"
    description = "Chinese SimpleQA: correctness via LLM judge"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        from wisent.core.reading.evaluators.benchmark_specific.judge_utils import (
            require_judge_model, call_judge, parse_binary_verdict,
        )
        model = require_judge_model(kwargs, self.name)
        prompt = _CHINESE_SIMPLEQA_JUDGE.format(
            expected=str(expected), response=response)
        verdict = call_judge(model, prompt)
        gt = parse_binary_verdict(
            verdict,
            ("correct",),
            ("incorrect", "not_attempted"))
        conf = EVAL_CONFIDENCE_FULL if gt != "UNKNOWN" else EVAL_CONFIDENCE_ZERO
        return EvalResult(
            ground_truth=gt, method_used=self.name, confidence=conf,
            details=f"Judge: {verdict[:DISPLAY_TRUNCATION_MEDIUM]}",
            meta={"judge_output": verdict, "expected": str(expected)})
