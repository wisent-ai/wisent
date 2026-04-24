"""Hallucination detection evaluators.

- HallucinationsLeaderboard: meta-benchmark averaging via LLM judge
- HaluEval: binary Yes/No classification (already correct)
- HaluLens: claim decomposition + verification via LLM judge
"""

import re
from typing import Any
import logging

from wisent.core.reading.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.utils.config_tools.constants import (
    CHANCE_LEVEL_ACCURACY,
    EVAL_CONFIDENCE_FULL,
    EVAL_CONFIDENCE_ZERO,
    DISPLAY_TRUNCATION_MEDIUM,
    SCORE_RANGE_MAX,
    SCORE_RANGE_MIN,
)

logger = logging.getLogger(__name__)

_HALLU_LEADERBOARD_JUDGE = (
    "You are evaluating answer correctness for a hallucination "
    "detection meta-benchmark (per Hallucinations Leaderboard).\n\n"
    "Expected answer: {expected}\n\n"
    "Model's answer: {response}\n\n"
    "Does the model's answer match the expected answer? Consider "
    "exact match, semantic equivalence, and factual consistency.\n"
    "Reply with exactly CORRECT or INCORRECT."
)

_HALULENS_JUDGE = (
    "You are a hallucination verifier (per HaluLens methodology). "
    "Decompose the response into individual claims and verify each "
    "against the reference.\n\n"
    "Reference: {expected}\n\n"
    "Response to verify: {response}\n\n"
    "Are all claims in the response supported by the reference?\n"
    "Reply with exactly SUPPORTED (no hallucinations) or "
    "UNSUPPORTED (contains hallucinated claims)."
)


class HallucinationsLeaderboardEvaluator(BaseEvaluator):
    """Hallucinations Leaderboard: meta-benchmark via LLM judge.

    Paper: Hallucinations Leaderboard. Method: Avg of EM, ROUGE-L,
    accuracy across sub-tasks via lm-eval-harness. This uses
    LLM-as-judge for correctness assessment.
    Metric: Averaged sub-task metrics.
    """

    name = "hallucinations_leaderboard"
    description = "Hallucinations Leaderboard: meta-benchmark via LLM judge"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        from wisent.core.reading.evaluators.benchmark_specific.judge_utils import (
            require_judge_model, call_judge, parse_binary_verdict,
        )
        model = require_judge_model(kwargs, self.name)
        prompt = _HALLU_LEADERBOARD_JUDGE.format(
            expected=str(expected), response=response)
        verdict = call_judge(model, prompt)
        gt = parse_binary_verdict(
            verdict, ("correct",), ("incorrect",))
        conf = EVAL_CONFIDENCE_FULL if gt != "UNKNOWN" else EVAL_CONFIDENCE_ZERO
        return EvalResult(
            ground_truth=gt, method_used=self.name, confidence=conf,
            details=f"Judge: {verdict[:DISPLAY_TRUNCATION_MEDIUM]}",
            meta={"judge_output": verdict, "expected": str(expected)})


class HaluEvalEvaluator(BaseEvaluator):
    """HaluEval: hallucination detection.

    Paper: HaluEval. Method: Binary classification (Yes/No) for
    whether an answer is hallucinated. In the contrastive pair
    framework, compares response text against expected answer
    using normalized text matching.
    Metric: Accuracy.
    """

    name = "halueval"
    description = "HaluEval: hallucination detection via text comparison"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        resp_norm = self.normalize_text(str(response))
        exp_norm = self.normalize_text(str(expected))
        if not resp_norm:
            return EvalResult(
                ground_truth="UNKNOWN", method_used=self.name,
                confidence=EVAL_CONFIDENCE_ZERO,
                details="Empty response")
        if resp_norm == exp_norm:
            return EvalResult(
                ground_truth="TRUTHFUL", method_used=self.name,
                confidence=SCORE_RANGE_MAX,
                details="Exact normalized match")
        if exp_norm in resp_norm or resp_norm in exp_norm:
            return EvalResult(
                ground_truth="TRUTHFUL", method_used=self.name,
                confidence=SCORE_RANGE_MAX,
                details="Containment match")
        # Token overlap
        resp_toks = set(resp_norm.split())
        exp_toks = set(exp_norm.split())
        if exp_toks:
            overlap = len(resp_toks & exp_toks) / len(exp_toks)
            if overlap >= CHANCE_LEVEL_ACCURACY:
                return EvalResult(
                    ground_truth="TRUTHFUL", method_used=self.name,
                    confidence=overlap,
                    details=f"Token overlap: {overlap:.3f}",
                    meta={"overlap": overlap})
        return EvalResult(
            ground_truth="UNTRUTHFUL", method_used=self.name,
            confidence=SCORE_RANGE_MIN,
            details="No match with expected answer")


class HaluLensEvaluator(BaseEvaluator):
    """HaluLens: claim verification via LLM judge.

    Paper: HaluLens. Method: LLM-as-judge (Llama-based) for claim
    decomposition and verification. Uses LLM-as-judge.
    Metric: Hallucination Rate, F-score@K.
    """

    name = "halulens"
    description = "HaluLens: claim verification via LLM judge"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        from wisent.core.reading.evaluators.benchmark_specific.judge_utils import (
            require_judge_model, call_judge, parse_binary_verdict,
        )
        model = require_judge_model(kwargs, self.name)
        prompt = _HALULENS_JUDGE.format(
            expected=str(expected), response=response)
        verdict = call_judge(model, prompt)
        gt = parse_binary_verdict(
            verdict, ("supported",), ("unsupported",))
        conf = EVAL_CONFIDENCE_FULL if gt != "UNKNOWN" else EVAL_CONFIDENCE_ZERO
        return EvalResult(
            ground_truth=gt, method_used=self.name, confidence=conf,
            details=f"Judge: {verdict[:DISPLAY_TRUNCATION_MEDIUM]}",
            meta={"judge_output": verdict, "expected": str(expected)})
