"""Non-MC evaluators for multilingual benchmarks.

Includes extractive QA (F1 / token-overlap), BLEU/chrF/ROUGE-style
metrics, MC-with-normalization (TruthfulQA), word continuation (LAMBADA),
and generation-based regex extraction (MMMU).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any
import logging

from wisent.core.reading.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.utils.config_tools.constants import (
    CHANCE_LEVEL_ACCURACY,
    SCORE_RANGE_MIN,
    SCORE_RANGE_MAX,
    EVAL_NUM_CONTRASTIVE_CHOICES,
    EVAL_CONTRASTIVE_CORRECT_IDX,
    EVAL_CONTRASTIVE_INCORRECT_IDX,
    EVAL_SINGLE_CHAR_LENGTH,
    EVAL_F1_HARMONIC_MULTIPLIER,
    EVAL_MC_ANSWER_PATTERN_GROUP,
)

logger = logging.getLogger(__name__)

_MMMU_ANSWER_RE = re.compile(r"\b([A-Da-d])\b")


def _tokenize(text: str, normalize_fn) -> list[str]:
    """Split text into tokens after normalization."""
    return normalize_fn(text).split()


def _compute_f1(resp_toks: list[str], exp_toks: list[str]) -> float:
    """Compute token-level F1 between two token lists."""
    if not resp_toks or not exp_toks:
        return SCORE_RANGE_MIN
    common = Counter(resp_toks) & Counter(exp_toks)
    num_common = sum(common.values())
    if num_common == EVAL_CONTRASTIVE_CORRECT_IDX:
        return SCORE_RANGE_MIN
    prec = num_common / len(resp_toks)
    rec = num_common / len(exp_toks)
    return EVAL_F1_HARMONIC_MULTIPLIER * prec * rec / (prec + rec)


def _contrastive_f1(evaluator, choices, expected):
    """Shared contrastive F1 comparison for DarijaBench and MLQA."""
    expected_list = expected if isinstance(expected, list) else [expected]
    best_c, best_i = SCORE_RANGE_MIN, SCORE_RANGE_MIN
    for exp in expected_list:
        e_t = _tokenize(str(exp), evaluator.normalize_text)
        c_t = _tokenize(str(choices[EVAL_CONTRASTIVE_CORRECT_IDX]), evaluator.normalize_text)
        i_t = _tokenize(str(choices[EVAL_CONTRASTIVE_INCORRECT_IDX]), evaluator.normalize_text)
        best_c = max(best_c, _compute_f1(c_t, e_t))
        best_i = max(best_i, _compute_f1(i_t, e_t))
    meta = {"correct_f1": best_c, "incorrect_f1": best_i}
    if best_c > best_i:
        return EvalResult(
            ground_truth="TRUTHFUL", method_used=evaluator.name,
            confidence=best_c,
            details=f"Correct F1={best_c:.3f} > Incorrect F1={best_i:.3f}", meta=meta)
    if best_i > best_c:
        return EvalResult(
            ground_truth="UNTRUTHFUL", method_used=evaluator.name,
            confidence=best_i,
            details=f"Incorrect F1={best_i:.3f} > Correct F1={best_c:.3f}", meta=meta)
    return EvalResult(
        ground_truth="UNKNOWN", method_used=evaluator.name,
        confidence=SCORE_RANGE_MIN,
        details=f"Tied: correct={best_c:.3f}, incorrect={best_i:.3f}", meta=meta)


def _direct_f1(evaluator, response, expected):
    """Shared direct F1 evaluation for DarijaBench and MLQA."""
    expected_list = expected if isinstance(expected, list) else [expected]
    best_f1, best_match = SCORE_RANGE_MIN, None
    for exp in expected_list:
        r_t = _tokenize(response, evaluator.normalize_text)
        e_t = _tokenize(str(exp), evaluator.normalize_text)
        f1 = _compute_f1(r_t, e_t)
        if f1 > best_f1:
            best_f1, best_match = f1, exp
    if best_f1 >= CHANCE_LEVEL_ACCURACY:
        gt = "TRUTHFUL"
    elif best_f1 > SCORE_RANGE_MIN:
        gt = "UNKNOWN"
    else:
        gt = "UNTRUTHFUL"
    return EvalResult(
        ground_truth=gt, method_used=evaluator.name, confidence=best_f1,
        details=f"F1={best_f1:.3f} vs '{best_match}'",
        meta={"f1": best_f1, "matched_answer": best_match})


class MLQAEvaluator(BaseEvaluator):
    """Extractive QA evaluator for MLQA (F1 + exact match)."""
    name = "mlqa"
    description = "MLQA extractive QA evaluator (F1 and exact match)"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        choices = kwargs.get("choices")
        if choices and len(choices) == EVAL_NUM_CONTRASTIVE_CHOICES:
            return _contrastive_f1(self, choices, expected)
        expected_list = expected if isinstance(expected, list) else [expected]
        resp_norm = self.normalize_text(response)
        for exp in expected_list:
            if resp_norm == self.normalize_text(str(exp)):
                return EvalResult(
                    ground_truth="TRUTHFUL", method_used=self.name,
                    confidence=SCORE_RANGE_MAX,
                    details=f"Exact match: '{response}' == '{exp}'",
                    meta={"matched_answer": exp, "metric": "exact_match"})
        return _direct_f1(self, response, expected)


class MMMUEvaluator(BaseEvaluator):
    """Generation-based evaluator for MMMU with regex answer extraction."""
    name = "mmmu"
    description = "MMMU generation-based evaluator (regex extraction)"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        choices = kwargs.get("choices")
        if choices and len(choices) == EVAL_NUM_CONTRASTIVE_CHOICES:
            return self._eval_contrastive(choices, expected)
        return self._eval_direct(response, expected)

    def _eval_contrastive(self, choices, expected):
        expected_list = expected if isinstance(expected, list) else [expected]
        c_ext = self._extract_answer(str(choices[EVAL_CONTRASTIVE_CORRECT_IDX]))
        i_ext = self._extract_answer(str(choices[EVAL_CONTRASTIVE_INCORRECT_IDX]))
        c_m = any(c_ext == self.normalize_text(str(e).strip()) for e in expected_list) if c_ext else False
        i_m = any(i_ext == self.normalize_text(str(e).strip()) for e in expected_list) if i_ext else False
        meta = {"correct_extracted": c_ext, "incorrect_extracted": i_ext}
        if c_m and not i_m:
            return EvalResult(
                ground_truth="TRUTHFUL", method_used=self.name,
                confidence=SCORE_RANGE_MAX, details="Correct extraction matches", meta=meta)
        if i_m and not c_m:
            return EvalResult(
                ground_truth="UNTRUTHFUL", method_used=self.name,
                confidence=SCORE_RANGE_MAX, details="Incorrect extraction matches", meta=meta)
        return EvalResult(
            ground_truth="UNKNOWN", method_used=self.name,
            confidence=SCORE_RANGE_MIN, details=f"Ambiguous: c={c_m}, i={i_m}", meta=meta)

    def _eval_direct(self, response, expected):
        extracted = self._extract_answer(response)
        expected_list = expected if isinstance(expected, list) else [expected]
        if extracted is None:
            return EvalResult(
                ground_truth="UNKNOWN", method_used=self.name,
                confidence=SCORE_RANGE_MIN, details="Could not extract answer")
        for exp in expected_list:
            if extracted == self.normalize_text(str(exp).strip()):
                return EvalResult(
                    ground_truth="TRUTHFUL", method_used=self.name,
                    confidence=SCORE_RANGE_MAX,
                    details=f"Extracted '{extracted}' matches '{exp}'",
                    meta={"extracted": extracted, "matched": exp})
        return EvalResult(
            ground_truth="UNTRUTHFUL", method_used=self.name,
            confidence=SCORE_RANGE_MIN,
            details=f"Extracted '{extracted}' not in {expected_list}",
            meta={"extracted": extracted})

    def _extract_answer(self, text: str) -> str | None:
        """Extract A/B/C/D answer letter from generation, or fall back to full text."""
        match = _MMMU_ANSWER_RE.search(text)
        if match:
            return self.normalize_text(match.group(EVAL_MC_ANSWER_PATTERN_GROUP))
        normalized = self.normalize_text(text.strip())
        if len(normalized) == EVAL_SINGLE_CHAR_LENGTH:
            return normalized
        return normalized if normalized else None


class OkapiTruthfulQAEvaluator(BaseEvaluator):
    """MC evaluator for Okapi TruthfulQA (multilingual).

    MC1: standard log-likelihood comparison (single true answer).
    MC2: normalized probability over set of true answers.
    For contrastive evaluation, compares which choice matches expected.
    """

    name = "okapi_truthfulqa"
    description = (
        "Okapi TruthfulQA MC evaluator "
        "(MC1 log-likelihood, MC2 normalized probability)"
    )

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        choices = kwargs.get("choices")
        if choices and len(choices) == EVAL_NUM_CONTRASTIVE_CHOICES:
            return self._eval_contrastive(choices, expected)
        return self._eval_direct(response, expected)

    def _eval_contrastive(self, choices, expected):
        expected_list = expected if isinstance(expected, list) else [expected]
        c_norm = self.normalize_text(str(choices[EVAL_CONTRASTIVE_CORRECT_IDX]))
        i_norm = self.normalize_text(str(choices[EVAL_CONTRASTIVE_INCORRECT_IDX]))
        c_match = any(c_norm == self.normalize_text(str(e)) for e in expected_list)
        i_match = any(i_norm == self.normalize_text(str(e)) for e in expected_list)
        meta = {"correct": str(choices[EVAL_CONTRASTIVE_CORRECT_IDX]),
                "incorrect": str(choices[EVAL_CONTRASTIVE_INCORRECT_IDX])}
        if c_match and not i_match:
            return EvalResult(
                ground_truth="TRUTHFUL", method_used=self.name,
                confidence=SCORE_RANGE_MAX, details="Correct matches expected", meta=meta)
        if i_match and not c_match:
            return EvalResult(
                ground_truth="UNTRUTHFUL", method_used=self.name,
                confidence=SCORE_RANGE_MAX, details="Incorrect matches expected", meta=meta)
        return EvalResult(
            ground_truth="UNKNOWN", method_used=self.name,
            confidence=SCORE_RANGE_MIN, details=f"Ambiguous: c={c_match}, i={i_match}", meta=meta)

    def _eval_direct(self, response, expected):
        resp_norm = self.normalize_text(str(response).strip())
        expected_list = expected if isinstance(expected, list) else [expected]
        for exp in expected_list:
            if resp_norm == self.normalize_text(str(exp).strip()):
                return EvalResult(
                    ground_truth="TRUTHFUL", method_used=self.name,
                    confidence=SCORE_RANGE_MAX,
                    details=f"Match: '{response}' == '{exp}'",
                    meta={"matched_answer": exp})
        return EvalResult(
            ground_truth="UNTRUTHFUL", method_used=self.name,
            confidence=SCORE_RANGE_MIN,
            details=f"No match: '{response}' not in {expected_list}")


class LambadaMultilingualEvaluator(BaseEvaluator):
    """Log-likelihood evaluator for LAMBADA Multilingual word prediction.

    Evaluates whether the model assigns highest probability to the correct
    final word of a passage.  For contrastive pairs, compares the correct
    word to the incorrect word against the expected answer.
    """

    name = "lambada_multilingual"
    description = (
        "LAMBADA Multilingual word prediction evaluator "
        "(log-likelihood continuation)"
    )

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        choices = kwargs.get("choices")
        if choices and len(choices) == EVAL_NUM_CONTRASTIVE_CHOICES:
            return self._eval_contrastive(choices, expected)
        return self._eval_direct(response, expected)

    def _eval_contrastive(self, choices, expected):
        expected_list = expected if isinstance(expected, list) else [expected]
        c_norm = self.normalize_text(str(choices[EVAL_CONTRASTIVE_CORRECT_IDX]))
        i_norm = self.normalize_text(str(choices[EVAL_CONTRASTIVE_INCORRECT_IDX]))
        c_match = any(c_norm == self.normalize_text(str(e)) for e in expected_list)
        i_match = any(i_norm == self.normalize_text(str(e)) for e in expected_list)
        meta = {"correct_word": str(choices[EVAL_CONTRASTIVE_CORRECT_IDX]),
                "incorrect_word": str(choices[EVAL_CONTRASTIVE_INCORRECT_IDX])}
        if c_match and not i_match:
            return EvalResult(
                ground_truth="TRUTHFUL", method_used=self.name,
                confidence=SCORE_RANGE_MAX, details="Correct word matches", meta=meta)
        if i_match and not c_match:
            return EvalResult(
                ground_truth="UNTRUTHFUL", method_used=self.name,
                confidence=SCORE_RANGE_MAX, details="Incorrect word matches", meta=meta)
        return EvalResult(
            ground_truth="UNKNOWN", method_used=self.name,
            confidence=SCORE_RANGE_MIN, details=f"Ambiguous: c={c_match}, i={i_match}", meta=meta)

    def _eval_direct(self, response, expected):
        resp_norm = self.normalize_text(str(response).strip())
        expected_list = expected if isinstance(expected, list) else [expected]
        for exp in expected_list:
            if resp_norm == self.normalize_text(str(exp).strip()):
                return EvalResult(
                    ground_truth="TRUTHFUL", method_used=self.name,
                    confidence=SCORE_RANGE_MAX,
                    details=f"Word match: '{response}' == '{exp}'",
                    meta={"matched_word": exp})
        return EvalResult(
            ground_truth="UNTRUTHFUL", method_used=self.name,
            confidence=SCORE_RANGE_MIN,
            details=f"No match: '{response}' not in {expected_list}")
