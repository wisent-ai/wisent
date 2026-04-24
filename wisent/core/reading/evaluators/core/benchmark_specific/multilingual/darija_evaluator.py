"""DarijaBench evaluator using BLEU and ROUGE-L metrics.

Paper: Atlas-Chat / DarijaBench. Multi-task benchmark for Moroccan
Arabic (Darija) covering sentiment, translation, and summarization.
Evaluation uses BLEU, chrF, and ROUGE as specified in the paper.
"""

from __future__ import annotations

from typing import Any
import logging

import evaluate as hf_evaluate

from wisent.core.reading.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN,
    SCORE_RANGE_MAX,
    CHANCE_LEVEL_ACCURACY,
    EVAL_NUM_CONTRASTIVE_CHOICES,
    EVAL_CONTRASTIVE_CORRECT_IDX,
    EVAL_CONTRASTIVE_INCORRECT_IDX,
)

logger = logging.getLogger(__name__)

_BLEU_METRIC = None
_ROUGE_METRIC = None


def _get_bleu():
    """Lazy-load HuggingFace BLEU metric."""
    global _BLEU_METRIC
    if _BLEU_METRIC is None:
        _BLEU_METRIC = hf_evaluate.load("bleu")
    return _BLEU_METRIC


def _get_rouge():
    """Lazy-load HuggingFace ROUGE metric."""
    global _ROUGE_METRIC
    if _ROUGE_METRIC is None:
        _ROUGE_METRIC = hf_evaluate.load("rouge")
    return _ROUGE_METRIC


def _compute_darija_score(prediction: str, reference: str) -> float:
    """Compute max of BLEU and ROUGE-L for a prediction/reference pair.

    DarijaBench uses BLEU for translation tasks and ROUGE for
    summarization tasks. Since the evaluator does not know which
    sub-task is being evaluated, we compute both and take the max.
    """
    pred_stripped = prediction.strip()
    ref_stripped = reference.strip()
    if not pred_stripped or not ref_stripped:
        return SCORE_RANGE_MIN
    bleu_result = _get_bleu().compute(
        predictions=[pred_stripped],
        references=[[ref_stripped]],
    )
    bleu_score = bleu_result["bleu"]
    rouge_result = _get_rouge().compute(
        predictions=[pred_stripped],
        references=[ref_stripped],
    )
    rouge_l = rouge_result["rougeL"]
    return max(bleu_score, rouge_l)


class DarijaBenchEvaluator(BaseEvaluator):
    """DarijaBench multi-task evaluator using BLEU and ROUGE-L.

    Paper: Atlas-Chat / DarijaBench.
    Method: Multi-task (sentiment/translation/summarization),
    BLEU/chrF/ROUGE. Metric: BLEU, chrF, ROUGE.
    """

    name = "darija_bench"
    description = "DarijaBench multi-task evaluator (BLEU/ROUGE-L)"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        choices = kwargs.get("choices")
        if choices and len(choices) == EVAL_NUM_CONTRASTIVE_CHOICES:
            return self._eval_contrastive(choices, expected)
        return self._eval_direct(response, expected)

    def _eval_direct(self, response: str, expected: Any) -> EvalResult:
        if not response or not response.strip():
            return EvalResult(
                ground_truth="UNKNOWN", method_used=self.name,
                confidence=SCORE_RANGE_MIN, details="Empty response")
        expected_list = expected if isinstance(expected, list) else [expected]
        best_score = SCORE_RANGE_MIN
        best_ref = None
        for ref in expected_list:
            score = _compute_darija_score(response, str(ref))
            if score > best_score:
                best_score = score
                best_ref = ref
        if best_score >= CHANCE_LEVEL_ACCURACY:
            gt = "TRUTHFUL"
        elif best_score > SCORE_RANGE_MIN:
            gt = "UNKNOWN"
        else:
            gt = "UNTRUTHFUL"
        return EvalResult(
            ground_truth=gt, method_used=self.name,
            confidence=best_score,
            details=f"BLEU/ROUGE-L score: {best_score:.3f}",
            meta={"score": best_score, "best_reference": best_ref})

    def _eval_contrastive(self, choices: list, expected: Any) -> EvalResult:
        expected_str = str(expected)
        c_score = _compute_darija_score(
            str(choices[EVAL_CONTRASTIVE_CORRECT_IDX]), expected_str)
        i_score = _compute_darija_score(
            str(choices[EVAL_CONTRASTIVE_INCORRECT_IDX]), expected_str)
        meta = {"correct_score": c_score, "incorrect_score": i_score}
        if c_score > i_score:
            return EvalResult(
                ground_truth="TRUTHFUL", method_used=self.name,
                confidence=c_score,
                details=f"Correct: {c_score:.3f} > Incorrect: {i_score:.3f}",
                meta=meta)
        if i_score > c_score:
            return EvalResult(
                ground_truth="UNTRUTHFUL", method_used=self.name,
                confidence=i_score,
                details=f"Incorrect: {i_score:.3f} > Correct: {c_score:.3f}",
                meta=meta)
        return EvalResult(
            ground_truth="UNKNOWN", method_used=self.name,
            confidence=SCORE_RANGE_MIN,
            details=f"Tied: {c_score:.3f}", meta=meta)
