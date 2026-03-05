"""Generation-based evaluator for benchmarks that require text generation.

This evaluator handles tasks like GSM8K, DROP, TriviaQA, TruthfulQA where the model generates
free-form text that must be parsed and compared to reference answers.

Uses semantic similarity (NLI + embeddings) for robust text matching.
"""

import re
from typing import Any, Dict
import logging

from wisent.core.reading.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.utils.infra_tools.errors import NumericalExtractionError, TextExtractionError
from wisent.core.utils.config_tools.constants import (
    COMPARE_TOL,
)
from wisent.core.reading.evaluators.benchmark_specific._generation_evaluator_helpers import (
    GenerationEvaluatorHelpersMixin,
)

logger = logging.getLogger(__name__)

# Lazy-loaded models for semantic matching
_CE_MODEL = None
_EMB_MODEL = None
CE_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_cross_encoder():
    """Lazy load NLI cross-encoder model."""
    global _CE_MODEL
    if _CE_MODEL is None:
        from sentence_transformers import CrossEncoder
        _CE_MODEL = CrossEncoder(CE_MODEL_NAME)
    return _CE_MODEL


def _get_embedding_model():
    """Lazy load sentence embedding model."""
    global _EMB_MODEL
    if _EMB_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMB_MODEL = SentenceTransformer(EMB_MODEL_NAME)
    return _EMB_MODEL


class GenerationEvaluator(GenerationEvaluatorHelpersMixin, BaseEvaluator):
    """Evaluator for generation-based benchmarks.

    Handles tasks that require text generation followed by answer extraction:
    - GSM8K: Math word problems requiring numerical answers
    - DROP: Reading comprehension with discrete reasoning
    - TriviaQA: Trivia questions requiring factual answers

    Note: Task names are declared in extractors (via evaluator_name attribute),
    not in evaluators. This maintains proper separation of concerns.
    """

    name = "generation"
    description = "Generation-based evaluator for text generation tasks"

    def __init__(
        self,
        *,
        generation_embedding_weight: float,
        generation_nli_weight: float,
    ):
        """Initialize with semantic matching weights.

        Args:
            generation_embedding_weight: Weight for embedding similarity in reference comparison.
            generation_nli_weight: Weight for NLI entailment in reference comparison.
        """
        self._generation_embedding_weight = generation_embedding_weight
        self._generation_nli_weight = generation_nli_weight

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate generated response against expected answer.

        Args:
            response: Generated model response (can be empty if using choices)
            expected: Expected answer (str, int, float, or list of acceptable answers)
            **kwargs:
                task_name: Task name for task-specific logic
                answer_type: Type of answer ("numerical", "text", "exact")
                normalize: Whether to normalize strings before comparison
                choices: List of [correct_answer, incorrect_answer]
                model: WisentModel instance for generating responses

        Returns:
            EvalResult with TRUTHFUL/UNTRUTHFUL/UNKNOWN
        """
        task_name = kwargs.get('task_name', '')
        answer_type = kwargs.get('answer_type', 'text')
        normalize = kwargs.get('normalize', True)
        choices = kwargs.get('choices')

        # If we have choices, do direct comparison (for contrastive evaluation)
        if choices and len(choices) == 2:
            return self._evaluate_choices(
                choices, expected, task_name, answer_type, normalize
            )

        # If no choices but we have a response and references, do relative comparison
        correct_answers = kwargs.get('correct_answers', [])
        incorrect_answers = kwargs.get('incorrect_answers', [])

        if response and correct_answers and incorrect_answers:
            return self._compare_to_references(
                response, correct_answers, incorrect_answers, task_name, answer_type,
                embedding_weight=self._generation_embedding_weight,
                nli_weight=self._generation_nli_weight,
            )

        # If no choices, extract answer from generated response
        extracted_answer = self._extract_answer(response, task_name, answer_type)

        if extracted_answer is None:
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details="Could not extract answer from generated response",
            )

        expected_answers = expected if isinstance(expected, list) else [expected]

        is_correct, matched_answer, confidence = self._check_match(
            extracted_answer, expected_answers, answer_type, normalize
        )

        return EvalResult(
            ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
            method_used=self.name,
            confidence=confidence,
            details=f"Extracted: '{extracted_answer}', Expected: '{matched_answer or expected_answers[0]}'",
            meta={
                "extracted_answer": extracted_answer,
                "expected_answers": expected_answers,
                "matched_answer": matched_answer,
                "answer_type": answer_type,
            }
        )

    def _evaluate_choices(self, choices, expected, task_name, answer_type, normalize):
        """Evaluate contrastive choices against expected answer."""
        correct_answer = choices[0]
        incorrect_answer = choices[1]

        math_tasks = {
            "gsm8k", "asdiv", "arithmetic", "math", "math_500", "math500",
            "aime", "aime2024", "aime2025", "hmmt", "hmmt_feb_2025",
            "polymath", "polymath_en_medium", "polymath_zh_medium",
            "polymath_en_high", "polymath_zh_high", "livemathbench",
            "livemathbench_cnmo_en", "livemathbench_cnmo_zh", "hendrycks_math",
        }
        if task_name in math_tasks:
            answer_type = "numerical"

        extracted_correct = self._extract_answer(correct_answer, task_name, answer_type)
        extracted_incorrect = self._extract_answer(incorrect_answer, task_name, answer_type)
        extracted_expected = self._extract_answer(str(expected), task_name, answer_type)

        if extracted_expected is None:
            return EvalResult(
                ground_truth="UNKNOWN", method_used=self.name, confidence=0.0,
                details="Could not extract answer from expected response",
            )

        expected_answers = expected if isinstance(expected, list) else [expected]

        if answer_type == "numerical":
            correct_matches = (extracted_correct is not None and extracted_expected is not None
                             and abs(extracted_correct - extracted_expected) < COMPARE_TOL)
            incorrect_matches = (extracted_incorrect is not None and extracted_expected is not None
                               and abs(extracted_incorrect - extracted_expected) < COMPARE_TOL)
            conf_correct = 1.0 if correct_matches else 0.0
            conf_incorrect = 1.0 if incorrect_matches else 0.0
        else:
            correct_matches, _, conf_correct = self._check_match(
                extracted_correct, expected_answers, answer_type, normalize)
            incorrect_matches, _, conf_incorrect = self._check_match(
                extracted_incorrect, expected_answers, answer_type, normalize)

        meta = {"correct_answer": correct_answer, "incorrect_answer": incorrect_answer, "expected": expected}

        if correct_matches and not incorrect_matches:
            return EvalResult(ground_truth="TRUTHFUL", method_used=self.name, confidence=conf_correct,
                            details=f"Correct answer '{correct_answer}' matches expected '{expected}'", meta=meta)
        elif incorrect_matches and not correct_matches:
            return EvalResult(ground_truth="UNTRUTHFUL", method_used=self.name, confidence=conf_incorrect,
                            details=f"Incorrect answer '{incorrect_answer}' matches expected '{expected}'", meta=meta)
        else:
            return EvalResult(ground_truth="UNKNOWN", method_used=self.name, confidence=0.0,
                            details=f"Ambiguous: correct={correct_matches}, incorrect={incorrect_matches}", meta=meta)

    def _extract_answer(self, response: str, task_name: str, answer_type: str) -> Any:
        """Extract answer from generated response."""
        if answer_type == "numerical" or task_name == "gsm8k":
            return self._extract_numerical_answer(response)
        else:
            return self._extract_text_answer(response)

    def _extract_numerical_answer(self, response: str) -> float:
        """Extract numerical answer from response (for math problems)."""
        frac_pattern = r'\\frac\{([-+]?\d+\.?\d*)\}\{([-+]?\d+\.?\d*)\}'
        frac_match = re.search(frac_pattern, response)
        if frac_match:
            try:
                numerator = float(frac_match.group(1))
                denominator = float(frac_match.group(2))
                if denominator != 0:
                    return numerator / denominator
            except (ValueError, ZeroDivisionError):
                pass

        degree_pattern = r'([-+]?\d+\.?\d*)\^\{?\\circ\}?'
        degree_match = re.search(degree_pattern, response)
        if degree_match:
            try:
                return float(degree_match.group(1))
            except ValueError:
                pass

        patterns = [
            r'####\s*([-+]?\d*\.?\d+)',
            r'answer\s*is\s*:?\s*([-+]?\d*\.?\d+)',
            r'=\s*([-+]?\d*\.?\d+)\s*$',
            r'^\s*([-+]?\d*\.?\d+)\s*$',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        raise NumericalExtractionError(response=response)

    def _extract_text_answer(self, response: str) -> str:
        """Extract text answer from response."""
        patterns = [
            r'answer\s*is:?\s*(.+?)(?:\n|$)',
            r'final\s+answer:?\s*(.+?)(?:\n|$)',
            r'(?:^|\n)answer:?\s*(.+?)(?:\n|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        raise TextExtractionError(response=response)

    def _check_match(self, extracted: Any, expected_list: list, answer_type: str, normalize: bool) -> tuple:
        """Check if extracted answer matches any expected answer."""
        if answer_type == "numerical":
            return self._check_numerical_match(extracted, expected_list)
        else:
            return self._check_text_match(extracted, expected_list, normalize)

    def _check_numerical_match(self, extracted: float, expected_list: list) -> tuple:
        """Check numerical match with tolerance."""
        if extracted is None:
            return False, None, 0.0
        for expected in expected_list:
            try:
                expected_num = float(expected)
                if abs(extracted - expected_num) < COMPARE_TOL:
                    return True, expected, 1.0
            except (ValueError, TypeError):
                continue
        return False, None, 0.0

    def _check_text_match(self, extracted: str, expected_list: list, normalize: bool) -> tuple:
        """Check text match using semantic similarity (NLI + embeddings)."""
        if extracted is None:
            return False, None, 0.0
        extracted_norm = self.normalize_text(extracted) if normalize else extracted
        for expected in expected_list:
            expected_str = str(expected)
            expected_norm = self.normalize_text(expected_str) if normalize else expected_str
            if extracted_norm == expected_norm:
                return True, expected, 1.0
            if extracted_norm in expected_norm or expected_norm in extracted_norm:
                return True, expected, 0.9

        for expected in expected_list:
            expected_str = str(expected)
            nli_score = self._nli_entailment(extracted, expected_str)
            if nli_score is not None and nli_score >= 0.5:
                confidence = min(0.85, 0.6 + nli_score * 0.3)
                return True, expected, confidence
            emb_score = self._embedding_similarity(extracted, expected_str)
            if emb_score is not None and emb_score >= 0.6:
                confidence = min(0.8, 0.5 + emb_score * 0.3)
                return True, expected, confidence
        return False, None, 0.0
