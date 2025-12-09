"""AIME evaluator for AIME competition math benchmarks.

This evaluator handles answer comparison for AIME (American Invitational Mathematics Examination)
benchmarks where answers are integers from 0-999 and may be in \\boxed{} format.
"""

import logging
from typing import Any

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.evaluators.benchmark_specific.math_parsing.extract_boxed import extract_boxed_answer

logger = logging.getLogger(__name__)


class AIMEEvaluator(BaseEvaluator):
    """Evaluator for AIME competition answer comparison.

    Designed for AIME benchmarks where:
    - Answers are integers from 0-999
    - Model outputs may contain answers in \\boxed{} format

    Uses the is_equiv function from math_equivalence package for robust comparison.
    """

    name = "aime"
    description = "AIME competition evaluator for integer answers (0-999)"

    @staticmethod
    def get_prompt(problem: str) -> str:
        """Create instruction prompt for LLM to solve AIME problem.

        Args:
            problem: The AIME problem statement

        Returns:
            Formatted prompt string
        """
        return f"""Solve the following AIME math problem step by step. AIME answers are always integers from 0 to 999. At the end, put your final answer inside \\boxed{{}}.

Problem: {problem}

Solution:"""

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate model response against expected AIME answer.

        Args:
            response: Model-generated response (may contain \\boxed{answer})
            expected: Expected answer - int 0-999

        Returns:
            EvalResult with TRUTHFUL/UNTRUTHFUL/UNKNOWN
        """

        # Try to extract answer from model response
        model_answer = extract_boxed_answer(response)

        if model_answer is None:
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details="Could not extract answer from model response",
                meta={
                    "response_preview": response[:200] if response else None,
                    "expected": expected,
                }
            )

        # AIME answers are integers 0-999, convert both to int for comparison
        try:
            model_int = int(model_answer)
            expected_int = int(expected)
            is_correct = model_int == expected_int
        except (ValueError, TypeError):
            # model_answer is not a valid integer
            is_correct = False

        return EvalResult(
            ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
            method_used=self.name,
            confidence=1.0 if is_correct else 0.0,
            details=f"Model: '{model_answer}' vs Expected: '{expected}'",
            meta={
                "model_answer": model_answer,
                "expected_answer": expected,
                "is_equivalent": is_correct,
            }
        )
