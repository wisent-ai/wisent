"""Miscellaneous evaluators for LongForm writing and MedConceptsQA benchmarks.

LongForm: Generation quality via METEOR score (evaluate library).
MedConceptsQA: Multiple-choice QA across medical vocabularies.
"""

import re
import logging
from typing import Any

import evaluate

from wisent.core.reading.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.utils.config_tools.constants import (
    EVAL_CONFIDENCE_FULL,
    EVAL_CONFIDENCE_ZERO,
    EVAL_CONTRASTIVE_CORRECT_IDX,
    EVAL_CONTRASTIVE_INCORRECT_IDX,
    EVAL_NUM_CONTRASTIVE_PAIR_SIZE,
    METEOR_SCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)

_LETTER_RE = re.compile(r"^\s*([A-Da-d])\s*[.):}\]]?\s*$")
_ANSWER_LETTER_RE = re.compile(
    r"(?:answer|option|choice)\s*(?:is|:)\s*([A-Da-d])", re.IGNORECASE,
)

_METEOR_METRIC = None


def _get_meteor():
    """Lazy-load the HuggingFace evaluate METEOR metric."""
    global _METEOR_METRIC
    if _METEOR_METRIC is None:
        _METEOR_METRIC = evaluate.load("meteor")
    return _METEOR_METRIC


def _compute_meteor(response: str, reference: str) -> float:
    """Compute METEOR score using the HuggingFace evaluate library.

    Uses the standard METEOR implementation which includes stemming,
    synonymy matching via WordNet, and chunk penalty.
    """
    if not response or not response.strip() or not reference or not reference.strip():
        return EVAL_CONFIDENCE_ZERO
    metric = _get_meteor()
    result = metric.compute(
        predictions=[response],
        references=[reference],
    )
    return result["meteor"]


def _extract_mc_letter(text: str) -> str | None:
    """Extract a multiple-choice letter (A/B/C/D) from response text."""
    m = _LETTER_RE.match(text)
    if m:
        return m.group(EVAL_CONTRASTIVE_INCORRECT_IDX).upper()
    m = _ANSWER_LETTER_RE.search(text)
    if m:
        return m.group(EVAL_CONTRASTIVE_INCORRECT_IDX).upper()
    stripped = text.strip()
    if stripped and stripped[EVAL_CONTRASTIVE_CORRECT_IDX].upper() in "ABCD":
        short_limit = EVAL_NUM_CONTRASTIVE_PAIR_SIZE + EVAL_CONTRASTIVE_INCORRECT_IDX
        if len(stripped) < short_limit:
            return stripped[EVAL_CONTRASTIVE_CORRECT_IDX].upper()
    return None


class LongformWritingEvaluator(BaseEvaluator):
    """Evaluator for LongForm writing benchmark.

    Paper: LongForm. Generation quality comparison against reference
    completions using simplified METEOR score (unigram precision/recall
    with stemming approximation). Metric: METEOR score.
    """

    name = "longform_writing"
    description = "LongForm evaluator using simplified METEOR score"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate model response against expected LongForm reference."""
        choices = kwargs.get("choices")
        if choices and len(choices) == EVAL_NUM_CONTRASTIVE_PAIR_SIZE:
            return self._evaluate_contrastive(choices, expected)
        return self._evaluate_direct(response, expected)

    def _evaluate_direct(self, response: str, expected: Any) -> EvalResult:
        """Direct evaluation using METEOR score."""
        if not response or not response.strip():
            return EvalResult(
                ground_truth="UNKNOWN", method_used=self.name,
                confidence=EVAL_CONFIDENCE_ZERO,
                details="Empty response", meta={"expected": expected})
        if isinstance(expected, list):
            references = expected
        else:
            references = [expected]
        best_score = EVAL_CONFIDENCE_ZERO
        best_ref = None
        for ref in references:
            score = _compute_meteor(response, str(ref))
            if score > best_score:
                best_score = score
                best_ref = ref
        if best_score >= METEOR_SCORE_THRESHOLD:
            return EvalResult(
                ground_truth="TRUTHFUL", method_used=self.name,
                confidence=best_score,
                details=f"METEOR score: {best_score:.3f}",
                meta={"meteor_score": best_score, "best_reference": best_ref})
        return EvalResult(
            ground_truth="UNTRUTHFUL", method_used=self.name,
            confidence=best_score,
            details=f"Low METEOR score: {best_score:.3f}",
            meta={"meteor_score": best_score, "best_reference": best_ref})

    def _evaluate_contrastive(self, choices: list, expected: Any) -> EvalResult:
        """Contrastive evaluation: compare METEOR scores of two choices."""
        correct_ans = str(choices[EVAL_CONTRASTIVE_CORRECT_IDX])
        incorrect_ans = str(choices[EVAL_CONTRASTIVE_INCORRECT_IDX])
        expected_str = str(expected)
        c_score = _compute_meteor(correct_ans, expected_str)
        i_score = _compute_meteor(incorrect_ans, expected_str)
        if c_score > i_score:
            return EvalResult(
                ground_truth="TRUTHFUL", method_used=self.name,
                confidence=c_score,
                details=f"Correct METEOR: {c_score:.3f} > Incorrect: {i_score:.3f}",
                meta={"correct_score": c_score, "incorrect_score": i_score})
        if i_score > c_score:
            return EvalResult(
                ground_truth="UNTRUTHFUL", method_used=self.name,
                confidence=i_score,
                details=f"Incorrect METEOR: {i_score:.3f} > Correct: {c_score:.3f}",
                meta={"correct_score": c_score, "incorrect_score": i_score})
        return EvalResult(
            ground_truth="UNKNOWN", method_used=self.name,
            confidence=EVAL_CONFIDENCE_ZERO,
            details=f"Tied METEOR scores: {c_score:.3f}",
            meta={"correct_score": c_score, "incorrect_score": i_score})


class MedConceptsQAEvaluator(BaseEvaluator):
    """Evaluator for MedConceptsQA medical vocabulary benchmark.

    Paper: MedConceptsQA. Multiple-choice QA across medical vocabularies.
    Supports letter answers (A/B/C/D) and full text answers.
    Metric: Accuracy.
    """

    name = "med_concepts_qa"
    description = "MedConceptsQA evaluator for medical multiple-choice QA"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate model response against expected MedConceptsQA answer."""
        choices = kwargs.get("choices")
        if choices and len(choices) == EVAL_NUM_CONTRASTIVE_PAIR_SIZE:
            return self._evaluate_contrastive(choices, expected)
        return self._evaluate_direct(response, expected)

    def _evaluate_direct(self, response: str, expected: Any) -> EvalResult:
        """Direct evaluation using letter and normalized text matching."""
        if not response or not response.strip():
            return EvalResult(
                ground_truth="UNKNOWN", method_used=self.name,
                confidence=EVAL_CONFIDENCE_ZERO,
                details="Empty response", meta={"expected": expected})
        expected_str = str(expected).strip()
        resp_letter = _extract_mc_letter(response)
        exp_letter = _extract_mc_letter(expected_str)
        if resp_letter is not None and exp_letter is not None:
            is_correct = resp_letter == exp_letter
            return EvalResult(
                ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
                method_used=self.name,
                confidence=EVAL_CONFIDENCE_FULL if is_correct else EVAL_CONFIDENCE_ZERO,
                details=f"Letter match: '{resp_letter}' vs '{exp_letter}'",
                meta={"response_letter": resp_letter, "expected_letter": exp_letter})
        norm_resp = self.normalize_text(response)
        norm_exp = self.normalize_text(expected_str)
        is_correct = norm_resp == norm_exp or norm_exp in norm_resp
        return EvalResult(
            ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
            method_used=self.name,
            confidence=EVAL_CONFIDENCE_FULL if is_correct else EVAL_CONFIDENCE_ZERO,
            details=f"Text comparison: response vs expected",
            meta={"response_norm": norm_resp, "expected_norm": norm_exp})

    def _evaluate_contrastive(self, choices: list, expected: Any) -> EvalResult:
        """Contrastive evaluation: compare two choices against expected."""
        correct_ans = str(choices[EVAL_CONTRASTIVE_CORRECT_IDX])
        incorrect_ans = str(choices[EVAL_CONTRASTIVE_INCORRECT_IDX])
        expected_str = str(expected).strip()
        exp_letter = _extract_mc_letter(expected_str)
        c_letter = _extract_mc_letter(correct_ans)
        i_letter = _extract_mc_letter(incorrect_ans)
        if exp_letter is not None:
            if c_letter == exp_letter and i_letter != exp_letter:
                return EvalResult(
                    ground_truth="TRUTHFUL", method_used=self.name,
                    confidence=EVAL_CONFIDENCE_FULL,
                    details="Correct choice letter matches expected",
                    meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans})
            if i_letter == exp_letter and c_letter != exp_letter:
                return EvalResult(
                    ground_truth="UNTRUTHFUL", method_used=self.name,
                    confidence=EVAL_CONFIDENCE_FULL,
                    details="Incorrect choice letter matches expected",
                    meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans})
        norm_exp = self.normalize_text(expected_str)
        norm_c = self.normalize_text(correct_ans)
        norm_i = self.normalize_text(incorrect_ans)
        c_match = norm_c == norm_exp or norm_exp in norm_c
        i_match = norm_i == norm_exp or norm_exp in norm_i
        if c_match and not i_match:
            return EvalResult(
                ground_truth="TRUTHFUL", method_used=self.name,
                confidence=EVAL_CONFIDENCE_FULL,
                details="Correct choice matches expected text",
                meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans})
        if i_match and not c_match:
            return EvalResult(
                ground_truth="UNTRUTHFUL", method_used=self.name,
                confidence=EVAL_CONFIDENCE_FULL,
                details="Incorrect choice matches expected text",
                meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans})
        return EvalResult(
            ground_truth="UNKNOWN", method_used=self.name,
            confidence=EVAL_CONFIDENCE_ZERO,
            details="Neither choice clearly matches expected",
            meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans,
                  "expected": expected_str})
