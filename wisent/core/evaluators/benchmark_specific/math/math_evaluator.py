"""Math evaluator for competition math benchmarks.

This evaluator handles mathematical answer comparison for benchmarks like
qwedsacf/competition_math (MATH dataset) where answers are in LaTeX format
and need to be compared for mathematical equivalence.

Uses the is_equiv function from math_equivalence package (hendrycks/math).
"""

import logging
from typing import Any

from wisent.core.evaluators.benchmark_specific.math_parsing.is_equiv import is_equiv

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.evaluators.benchmark_specific.math_parsing.extract_boxed import extract_boxed_answer

logger = logging.getLogger(__name__)


class MathEvaluator(BaseEvaluator):
    """Evaluator for mathematical answer comparison.

    Designed for benchmarks like qwedsacf/competition_math where:
    - Model outputs contain answers in \\boxed{} format
    - Ground truth solutions also contain \\boxed{} answers
    - Answers need mathematical equivalence checking (not just string matching)

    Uses the is_equiv function from math_equivalence package for robust comparison
    of LaTeX mathematical expressions.
    """

    name = "math"
    description = "Mathematical equivalence evaluator for competition math benchmarks"

    @staticmethod
    def get_prompt(problem: str) -> str:
        """Create instruction prompt for LLM to solve math problem.

        Args:
            problem: The math problem statement

        Returns:
            Formatted prompt string
        """
        return f"""Solve the following math problem step by step. At the end, put your final answer inside \\boxed{{}}.

Problem: {problem}

Solution:"""

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate model response against expected mathematical answer.

        Args:
            response: Model-generated response (should contain \\boxed{answer})
            expected: Expected answer - can be:
                - A string containing \\boxed{answer} (will extract)
                - A raw answer string (used directly)
            **kwargs:
                extract_from_expected: If True (default), extract \\boxed{} from expected
                verbose: If True, log debug information

        Returns:
            EvalResult with TRUTHFUL/UNTRUTHFUL/UNKNOWN
        """
        extract_from_expected = kwargs.get('extract_from_expected', True)
        verbose = kwargs.get('verbose', False)

        # Extract answer from model response
        model_answer = extract_boxed_answer(response)

        if model_answer is None:
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details="Could not extract \\boxed{} answer from model response",
                meta={
                    "response_preview": response[:200] if response else None,
                    "expected": expected,
                }
            )

        # Get expected answer
        if extract_from_expected and isinstance(expected, str):
            expected_answer = extract_boxed_answer(expected)
            if expected_answer is None:
                # If no boxed answer in expected, use expected directly
                expected_answer = expected
        else:
            expected_answer = str(expected) if expected is not None else None

        if expected_answer is None:
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details="Could not determine expected answer",
                meta={
                    "model_answer": model_answer,
                    "expected_raw": expected,
                }
            )

        # Compare using mathematical equivalence
        is_correct = is_equiv(model_answer, expected_answer, verbose=verbose)

        return EvalResult(
            ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
            method_used=self.name,
            confidence=1.0 if is_correct else 0.0,
            details=f"Model: '{model_answer}' vs Expected: '{expected_answer}'",
            meta={
                "model_answer": model_answer,
                "expected_answer": expected_answer,
                "is_equivalent": is_correct,
            }
        )
