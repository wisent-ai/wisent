"""Math olympiad evaluators for CNMO and OlympiadBench benchmarks.

CNMO: LiveMathBench/CNMO with G-Pass@k metric.
OlympiadBench: SymPy-based mathematical equivalence with numerical tolerance.
"""

import re
import logging
from typing import Any

from wisent.core.reading.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.reading.evaluators.benchmark_specific.math_parsing.is_equiv import is_equiv
from wisent.core.utils.config_tools.constants import (
    COMPARE_TOL,
    DISPLAY_TRUNCATION_MEDIUM,
    EVAL_CONFIDENCE_FULL,
    EVAL_CONFIDENCE_ZERO,
    EVAL_CONTRASTIVE_CORRECT_IDX,
    EVAL_CONTRASTIVE_INCORRECT_IDX,
    EVAL_NUM_CONTRASTIVE_PAIR_SIZE,
    INDEX_FIRST,
    INDEX_LAST,
    MATH_REL_TOL,
    NEAR_ZERO_TOL,
    PERCENT_MULTIPLIER,
    SCIENTIFIC_NOTATION_BASE,
    SENSOR_LAST_OFFSET,
)

logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\{")
_HASH_ANSWER_RE = re.compile(r"####\s*(.+)")
_ANSWER_IS_RE = re.compile(
    r"(?:the\s+|final\s+)?answer\s*(?:is|:)\s*(.+?)(?:\.|$)", re.IGNORECASE,
)
_FRAC_RE = re.compile(r"\\frac\{([^}]+)\}\{([^}]+)\}")
_SCIENTIFIC_RE = re.compile(
    rf"([+-]?\d+\.?\d*)\s*[xX\u00d7]\s*{SCIENTIFIC_NOTATION_BASE}"
    r"\s*\^\s*\{?\s*([+-]?\d+)\s*\}?",
)
_PERCENT_RE = re.compile(r"([+-]?\d+\.?\d*)\s*%")


def _extract_boxed(text: str) -> str | None:
    """Extract the last \\boxed{} answer, handling nested braces."""
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    start_idx = matches[INDEX_LAST].end()
    brace_count = SENSOR_LAST_OFFSET
    idx = start_idx
    while idx < len(text) and brace_count > INDEX_FIRST:
        if text[idx] == "{":
            brace_count += SENSOR_LAST_OFFSET
        elif text[idx] == "}":
            brace_count -= SENSOR_LAST_OFFSET
        idx += SENSOR_LAST_OFFSET
    if brace_count == INDEX_FIRST:
        return text[start_idx : idx - SENSOR_LAST_OFFSET].strip()
    return None


def _extract_numerical_answer(text: str) -> str | None:
    """Extract numerical answer using \\boxed{}, ####, 'answer is' patterns."""
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed
    m = _HASH_ANSWER_RE.search(text)
    if m:
        return m.group(SENSOR_LAST_OFFSET).strip()
    m = _ANSWER_IS_RE.search(text)
    if m:
        return m.group(SENSOR_LAST_OFFSET).strip()
    return text.strip() if text else None


def _parse_number(s: str) -> float | None:
    """Parse string to float: fractions, percentages, scientific notation."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace(" ", "")
    m = _FRAC_RE.fullmatch(s)
    if m:
        try:
            return float(m.group(SENSOR_LAST_OFFSET)) / float(
                m.group(EVAL_NUM_CONTRASTIVE_PAIR_SIZE))
        except (ValueError, ZeroDivisionError):
            pass
    if "/" in s and s.count("/") == SENSOR_LAST_OFFSET:
        parts = s.split("/")
        try:
            return float(parts[INDEX_FIRST]) / float(parts[SENSOR_LAST_OFFSET])
        except (ValueError, ZeroDivisionError):
            pass
    m = _SCIENTIFIC_RE.fullmatch(s)
    if m:
        try:
            return float(m.group(SENSOR_LAST_OFFSET)) * (
                SCIENTIFIC_NOTATION_BASE ** float(
                    m.group(EVAL_NUM_CONTRASTIVE_PAIR_SIZE)))
        except (ValueError, OverflowError):
            pass
    m = _PERCENT_RE.fullmatch(s)
    if m:
        try:
            return float(m.group(SENSOR_LAST_OFFSET)) / PERCENT_MULTIPLIER
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError:
        return None


def _numbers_close(a: float, b: float) -> bool:
    """Check if two numbers are close using absolute and relative tolerance."""
    if abs(a - b) < COMPARE_TOL:
        return True
    denom = max(abs(a), abs(b))
    if denom < NEAR_ZERO_TOL:
        return abs(a - b) < COMPARE_TOL
    return abs(a - b) / denom < MATH_REL_TOL


def _try_sympy_equiv(expr_a: str, expr_b: str) -> bool | None:
    """Try SymPy-based symbolic equivalence. Returns None on failure."""
    try:
        import sympy
        parsed_a = sympy.sympify(expr_a, evaluate=True)
        parsed_b = sympy.sympify(expr_b, evaluate=True)
        diff = sympy.simplify(parsed_a - parsed_b)
        if diff == INDEX_FIRST:
            return True
        val = complex(diff.evalf())
        return abs(val) < COMPARE_TOL
    except Exception:
        return None


def _contrastive_numeric(evaluator, choices, expected):
    """Shared contrastive evaluation for math olympiad evaluators."""
    correct_ans = str(choices[EVAL_CONTRASTIVE_CORRECT_IDX])
    incorrect_ans = str(choices[EVAL_CONTRASTIVE_INCORRECT_IDX])
    exp_num = _parse_number(str(expected))
    c_num = _parse_number(correct_ans)
    i_num = _parse_number(incorrect_ans)
    if exp_num is not None and c_num is not None and i_num is not None:
        c_match = _numbers_close(c_num, exp_num)
        i_match = _numbers_close(i_num, exp_num)
        if c_match and not i_match:
            return EvalResult(
                ground_truth="TRUTHFUL", method_used=evaluator.name,
                confidence=EVAL_CONFIDENCE_FULL,
                details="Correct choice matches expected numerically",
                meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans})
        if i_match and not c_match:
            return EvalResult(
                ground_truth="UNTRUTHFUL", method_used=evaluator.name,
                confidence=EVAL_CONFIDENCE_FULL,
                details="Incorrect choice matches expected numerically",
                meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans})
    norm_exp = evaluator.normalize_text(str(expected))
    norm_c = evaluator.normalize_text(correct_ans)
    norm_i = evaluator.normalize_text(incorrect_ans)
    if norm_c == norm_exp and norm_i != norm_exp:
        return EvalResult(
            ground_truth="TRUTHFUL", method_used=evaluator.name,
            confidence=EVAL_CONFIDENCE_FULL, details="Correct choice matches (string)",
            meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans})
    if norm_i == norm_exp and norm_c != norm_exp:
        return EvalResult(
            ground_truth="UNTRUTHFUL", method_used=evaluator.name,
            confidence=EVAL_CONFIDENCE_FULL, details="Incorrect choice matches (string)",
            meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans})
    return EvalResult(
        ground_truth="UNKNOWN", method_used=evaluator.name,
        confidence=EVAL_CONFIDENCE_ZERO,
        details="Neither choice clearly matches expected",
        meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans,
              "expected": str(expected)})


class CNMOEvaluator(BaseEvaluator):
    """CNMO evaluator from LiveMathBench. Metric: G-Pass@k."""

    name = "cnmo"
    description = "CNMO evaluator with numerical extraction and tolerance comparison"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate model response against expected CNMO answer."""
        choices = kwargs.get("choices")
        if choices and len(choices) == EVAL_NUM_CONTRASTIVE_PAIR_SIZE:
            return _contrastive_numeric(self, choices, expected)
        model_raw = _extract_numerical_answer(response)
        if model_raw is None:
            return EvalResult(
                ground_truth="UNKNOWN", method_used=self.name,
                confidence=EVAL_CONFIDENCE_ZERO,
                details="Could not extract answer from response",
                meta={"response_preview": response[:DISPLAY_TRUNCATION_MEDIUM]
                      if response else None, "expected": expected})
        model_num = _parse_number(model_raw)
        expected_num = _parse_number(str(expected))
        if model_num is not None and expected_num is not None:
            is_correct = _numbers_close(model_num, expected_num)
            return EvalResult(
                ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
                method_used=self.name,
                confidence=EVAL_CONFIDENCE_FULL if is_correct else EVAL_CONFIDENCE_ZERO,
                details=f"Model: '{model_raw}' vs Expected: '{expected}'",
                meta={"model_answer": model_raw, "expected_answer": str(expected),
                      "is_equivalent": is_correct})
        norm_m = self.normalize_text(model_raw)
        norm_e = self.normalize_text(str(expected))
        is_correct = norm_m == norm_e
        return EvalResult(
            ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
            method_used=self.name,
            confidence=EVAL_CONFIDENCE_FULL if is_correct else EVAL_CONFIDENCE_ZERO,
            details=f"String: '{norm_m}' vs '{norm_e}'",
            meta={"model_answer": model_raw, "expected_answer": str(expected),
                  "is_equivalent": is_correct})


class OlympiadBenchEvaluator(BaseEvaluator):
    """OlympiadBench evaluator with SymPy equivalence. Metric: Micro-avg accuracy."""

    name = "olympiadbench"
    description = "OlympiadBench evaluator with SymPy equivalence"

    def _check_equivalence(self, candidate: str, expected_str: str) -> bool:
        """Check equivalence: numeric, is_equiv (LaTeX), SymPy, string."""
        cand_num = _parse_number(candidate)
        exp_num = _parse_number(expected_str)
        if cand_num is not None and exp_num is not None:
            if _numbers_close(cand_num, exp_num):
                return True
        if is_equiv(candidate, expected_str):
            return True
        sympy_result = _try_sympy_equiv(candidate, expected_str)
        if sympy_result is True:
            return True
        return self.normalize_text(candidate) == self.normalize_text(expected_str)

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate model response against expected OlympiadBench answer."""
        choices = kwargs.get("choices")
        if choices and len(choices) == EVAL_NUM_CONTRASTIVE_PAIR_SIZE:
            return self._evaluate_contrastive(choices, expected)
        model_raw = _extract_numerical_answer(response)
        if model_raw is None:
            return EvalResult(
                ground_truth="UNKNOWN", method_used=self.name,
                confidence=EVAL_CONFIDENCE_ZERO,
                details="Could not extract answer from response",
                meta={"response_preview": response[:DISPLAY_TRUNCATION_MEDIUM]
                      if response else None, "expected": expected})
        expected_str = _extract_numerical_answer(str(expected)) or str(expected).strip()
        is_correct = self._check_equivalence(model_raw, expected_str)
        return EvalResult(
            ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
            method_used=self.name,
            confidence=EVAL_CONFIDENCE_FULL if is_correct else EVAL_CONFIDENCE_ZERO,
            details=f"Model: '{model_raw}' vs Expected: '{expected_str}'",
            meta={"model_answer": model_raw, "expected_answer": expected_str,
                  "is_equivalent": is_correct})

    def _evaluate_contrastive(self, choices: list, expected: Any) -> EvalResult:
        """Contrastive evaluation: compare two choices against expected."""
        correct_ans = str(choices[EVAL_CONTRASTIVE_CORRECT_IDX])
        incorrect_ans = str(choices[EVAL_CONTRASTIVE_INCORRECT_IDX])
        expected_str = str(expected).strip()
        c_match = self._check_equivalence(correct_ans, expected_str)
        i_match = self._check_equivalence(incorrect_ans, expected_str)
        if c_match and not i_match:
            return EvalResult(
                ground_truth="TRUTHFUL", method_used=self.name,
                confidence=EVAL_CONFIDENCE_FULL,
                details="Correct choice is equivalent to expected",
                meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans,
                      "expected": expected_str})
        if i_match and not c_match:
            return EvalResult(
                ground_truth="UNTRUTHFUL", method_used=self.name,
                confidence=EVAL_CONFIDENCE_FULL,
                details="Incorrect choice is equivalent to expected",
                meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans,
                      "expected": expected_str})
        return EvalResult(
            ground_truth="UNKNOWN", method_used=self.name,
            confidence=EVAL_CONFIDENCE_ZERO,
            details="Neither choice clearly matches expected",
            meta={"correct_answer": correct_ans, "incorrect_answer": incorrect_ans,
                  "expected": expected_str})
