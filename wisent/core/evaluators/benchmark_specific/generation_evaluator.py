"""Generation-based evaluator for benchmarks that require text generation.

This evaluator handles tasks like GSM8K, DROP, TriviaQA where the model generates
free-form text that must be parsed and compared to reference answers.
"""

import re
from typing import Any, Dict
import logging

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)


class GenerationEvaluator(BaseEvaluator):
    """Evaluator for generation-based benchmarks.

    Handles tasks that require text generation followed by answer extraction:
    - GSM8K: Math word problems requiring numerical answers
    - DROP: Reading comprehension with discrete reasoning
    - TriviaQA: Trivia questions requiring factual answers

    Methods:
    - Exact match: Direct string comparison
    - Normalized match: Lenient comparison after normalization
    - Numerical match: For math problems
    """

    name = "generation"
    description = "Generation-based evaluator for text generation tasks"
    task_names = ("gsm8k", "drop", "triviaqa")

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate generated response against expected answer.

        Args:
            response: Generated model response
            expected: Expected answer (str, int, float, or list of acceptable answers)
            **kwargs:
                task_name: Task name for task-specific logic
                answer_type: Type of answer ("numerical", "text", "exact")
                normalize: Whether to normalize strings before comparison

        Returns:
            EvalResult with TRUTHFUL/UNTRUTHFUL/UNKNOWN
        """
        task_name = kwargs.get('task_name', '')
        answer_type = kwargs.get('answer_type', 'text')
        normalize = kwargs.get('normalize', True)

        # Extract answer from generated response
        extracted_answer = self._extract_answer(response, task_name, answer_type)

        if extracted_answer is None:
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details="Could not extract answer from generated response",
            )

        # Normalize expected answer
        if isinstance(expected, list):
            expected_answers = expected
        else:
            expected_answers = [expected]

        # Check if extracted answer matches any expected answer
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

    def _extract_answer(self, response: str, task_name: str, answer_type: str) -> Any:
        """Extract answer from generated response."""
        if answer_type == "numerical" or task_name == "gsm8k":
            return self._extract_numerical_answer(response)
        else:
            return self._extract_text_answer(response)

    def _extract_numerical_answer(self, response: str) -> float:
        """Extract numerical answer from response (for math problems)."""
        # Look for common patterns
        patterns = [
            r'####\s*([-+]?\d*\.?\d+)',  # GSM8K format
            r'answer\s*is\s*([-+]?\d*\.?\d+)',
            r'=\s*([-+]?\d*\.?\d+)\s*$',
            r'\$?\s*([-+]?\d*\.?\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        # Fallback: find last number in response
        numbers = re.findall(r'[-+]?\d*\.?\d+', response)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass

        return None

    def _extract_text_answer(self, response: str) -> str:
        """Extract text answer from response."""
        # Look for explicit answer markers
        patterns = [
            r'answer\s*is:?\s*(.+?)(?:\n|$)',
            r'final\s+answer:?\s*(.+?)(?:\n|$)',
            r'(?:^|\n)answer:?\s*(.+?)(?:\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: use first sentence
        sentences = re.split(r'[.!?]\s+', response)
        if sentences:
            return sentences[0].strip()

        return response.strip()

    def _check_match(
        self, extracted: Any, expected_list: list, answer_type: str, normalize: bool
    ) -> tuple:
        """Check if extracted answer matches any expected answer.

        Returns:
            (is_correct, matched_answer, confidence)
        """
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
                # Check if close enough (tolerance for floating point)
                if abs(extracted - expected_num) < 1e-6:
                    return True, expected, 1.0
            except (ValueError, TypeError):
                continue

        return False, None, 0.0

    def _check_text_match(self, extracted: str, expected_list: list, normalize: bool) -> tuple:
        """Check text match with optional normalization."""
        if extracted is None:
            return False, None, 0.0

        if normalize:
            extracted_norm = self.normalize_text(extracted)
        else:
            extracted_norm = extracted

        for expected in expected_list:
            expected_str = str(expected)
            if normalize:
                expected_norm = self.normalize_text(expected_str)
            else:
                expected_norm = expected_str

            # Exact match
            if extracted_norm == expected_norm:
                return True, expected, 1.0

            # Substring match
            if extracted_norm in expected_norm or expected_norm in extracted_norm:
                return True, expected, 0.8

        return False, None, 0.0
