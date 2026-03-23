"""Multiple-choice evaluators for multilingual benchmarks.

All evaluators here use log-likelihood / MC comparison: compare response
to expected answer choice.  For contrastive evaluation (``choices`` kwarg),
compare which choice matches expected better.
"""

from typing import Any
import logging

from wisent.core.reading.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN,
    SCORE_RANGE_MAX,
    EVAL_PREFIX_MATCH_CONFIDENCE,
    EVAL_NUM_CONTRASTIVE_CHOICES,
    EVAL_SINGLE_CHAR_LENGTH,
    EVAL_CONTRASTIVE_CORRECT_IDX,
    EVAL_CONTRASTIVE_INCORRECT_IDX,
)

logger = logging.getLogger(__name__)


def _mc_evaluate(
    evaluator: BaseEvaluator,
    response: str,
    expected: Any,
    **kwargs,
) -> EvalResult:
    """Shared MC evaluation logic used by every evaluator in this module.

    arguments:
        evaluator:
            the evaluator instance (used for ``name`` and ``normalize_text``).
        response:
            model response string.
        expected:
            expected answer (str, or list of acceptable answers).
        **kwargs:
            choices -- optional ``[correct, incorrect]`` for contrastive eval.
    """
    choices = kwargs.get("choices")
    if choices and len(choices) == EVAL_NUM_CONTRASTIVE_CHOICES:
        return _evaluate_contrastive(evaluator, choices, expected)
    return _evaluate_direct(evaluator, response, expected)


def _evaluate_contrastive(
    evaluator: BaseEvaluator,
    choices: list,
    expected: Any,
) -> EvalResult:
    """Compare two contrastive choices against the expected answer."""
    correct_answer = str(choices[EVAL_CONTRASTIVE_CORRECT_IDX]).strip()
    incorrect_answer = str(choices[EVAL_CONTRASTIVE_INCORRECT_IDX]).strip()

    expected_list = expected if isinstance(expected, list) else [expected]

    correct_matches = _any_match(evaluator, correct_answer, expected_list)
    incorrect_matches = _any_match(evaluator, incorrect_answer, expected_list)

    meta = {
        "correct_answer": correct_answer,
        "incorrect_answer": incorrect_answer,
        "expected": expected,
    }

    if correct_matches and not incorrect_matches:
        return EvalResult(
            ground_truth="TRUTHFUL",
            method_used=evaluator.name,
            confidence=SCORE_RANGE_MAX,
            details=(
                f"Correct choice '{correct_answer}' matches "
                f"expected '{expected}'"
            ),
            meta=meta,
        )
    if incorrect_matches and not correct_matches:
        return EvalResult(
            ground_truth="UNTRUTHFUL",
            method_used=evaluator.name,
            confidence=SCORE_RANGE_MAX,
            details=(
                f"Incorrect choice '{incorrect_answer}' matches "
                f"expected '{expected}'"
            ),
            meta=meta,
        )
    return EvalResult(
        ground_truth="UNKNOWN",
        method_used=evaluator.name,
        confidence=SCORE_RANGE_MIN,
        details=(
            f"Ambiguous: correct_matches={correct_matches}, "
            f"incorrect_matches={incorrect_matches}"
        ),
        meta=meta,
    )


def _any_match(
    evaluator: BaseEvaluator,
    candidate: str,
    expected_list: list,
) -> bool:
    """Return True if *candidate* matches any entry in *expected_list*."""
    cand_norm = evaluator.normalize_text(candidate)
    for exp in expected_list:
        exp_norm = evaluator.normalize_text(str(exp).strip())
        if cand_norm == exp_norm:
            return True
        if (
            len(cand_norm) == EVAL_SINGLE_CHAR_LENGTH
            and len(exp_norm) == EVAL_SINGLE_CHAR_LENGTH
            and cand_norm == exp_norm
        ):
            return True
        if (
            len(exp_norm) == EVAL_SINGLE_CHAR_LENGTH
            and cand_norm.startswith(exp_norm)
        ):
            return True
        if (
            len(cand_norm) == EVAL_SINGLE_CHAR_LENGTH
            and exp_norm.startswith(cand_norm)
        ):
            return True
    return False


def _evaluate_direct(
    evaluator: BaseEvaluator,
    response: str,
    expected: Any,
) -> EvalResult:
    """Direct comparison of response to expected."""
    resp_norm = evaluator.normalize_text(str(response).strip())
    expected_list = expected if isinstance(expected, list) else [expected]

    for exp in expected_list:
        exp_norm = evaluator.normalize_text(str(exp).strip())
        if resp_norm == exp_norm:
            return EvalResult(
                ground_truth="TRUTHFUL",
                method_used=evaluator.name,
                confidence=SCORE_RANGE_MAX,
                details=f"Match: '{response}' == '{exp}'",
                meta={"matched_answer": exp},
            )
        if (
            len(resp_norm) == EVAL_SINGLE_CHAR_LENGTH
            and len(exp_norm) == EVAL_SINGLE_CHAR_LENGTH
            and resp_norm == exp_norm
        ):
            return EvalResult(
                ground_truth="TRUTHFUL",
                method_used=evaluator.name,
                confidence=SCORE_RANGE_MAX,
                details=f"Letter match: '{response}' == '{exp}'",
                meta={"matched_answer": exp},
            )
        if (
            len(exp_norm) == EVAL_SINGLE_CHAR_LENGTH
            and resp_norm.startswith(exp_norm)
        ):
            return EvalResult(
                ground_truth="TRUTHFUL",
                method_used=evaluator.name,
                confidence=EVAL_PREFIX_MATCH_CONFIDENCE,
                details=f"Prefix match: '{response}' starts with '{exp}'",
                meta={"matched_answer": exp},
            )
    return EvalResult(
        ground_truth="UNTRUTHFUL",
        method_used=evaluator.name,
        confidence=SCORE_RANGE_MIN,
        details=f"No match: '{response}' not in {expected_list}",
    )


# ---------------------------------------------------------------------------
# Evaluator classes
# ---------------------------------------------------------------------------


class CLUEWSCEvaluator(BaseEvaluator):
    """Binary classification (true/false) via log-likelihood.

    Used for CLUE-WSC, the Chinese Winograd Schema Challenge.
    """

    name = "cluewsc"
    description = (
        "CLUE-WSC binary classification evaluator "
        "(true/false log-likelihood)"
    )

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        return _mc_evaluate(self, response, expected, **kwargs)


class EusExamsEvaluator(BaseEvaluator):
    """MC evaluator for Basque public exam questions (A/B/C/D)."""

    name = "eus_exams"
    description = "EusExams Basque public exam MC evaluator"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        return _mc_evaluate(self, response, expected, **kwargs)


class InverseScalingEvaluator(BaseEvaluator):
    """Log-likelihood MC evaluator for the Inverse Scaling benchmark."""

    name = "inverse_scaling"
    description = "Inverse Scaling log-likelihood MC evaluator"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        return _mc_evaluate(self, response, expected, **kwargs)


class MMLUReduxEvaluator(BaseEvaluator):
    """Log-likelihood MC evaluator for MMLU-Redux (curated subset)."""

    name = "mmlu_redux"
    description = "MMLU-Redux log-likelihood MC evaluator"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        return _mc_evaluate(self, response, expected, **kwargs)


class OkapiHellaswagEvaluator(BaseEvaluator):
    """Log-likelihood MC evaluator for multilingual HellaSwag (Okapi)."""

    name = "okapi_hellaswag"
    description = "Okapi multilingual HellaSwag MC evaluator"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        return _mc_evaluate(self, response, expected, **kwargs)


class OkapiMMLUEvaluator(BaseEvaluator):
    """Log-likelihood MC evaluator for multilingual MMLU (Okapi)."""

    name = "okapi_mmlu"
    description = "Okapi multilingual MMLU MC evaluator"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        return _mc_evaluate(self, response, expected, **kwargs)


class PawsXEvaluator(BaseEvaluator):
    """Binary classification (paraphrase/not) via log-likelihood.

    Used for PAWS-X, cross-lingual paraphrase adversaries benchmark.
    """

    name = "paws_x"
    description = (
        "PAWS-X cross-lingual paraphrase binary classification evaluator"
    )

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        return _mc_evaluate(self, response, expected, **kwargs)
