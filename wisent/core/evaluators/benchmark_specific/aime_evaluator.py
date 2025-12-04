"""AIME evaluator for AIME competition math benchmarks.

This evaluator handles answer comparison for AIME (American Invitational Mathematics Examination)
benchmarks where answers are integers from 0-999 and may be in \\boxed{} format.

Uses the is_equiv function from math_equivalence package for robust comparison.
"""

import re
import logging
from typing import Any

from math_equivalence import is_equiv

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)


def extract_boxed_answer(text: str) -> str | None:
    """Extract the LAST \\boxed{} answer from text (final answer convention).

    Handles nested braces correctly (e.g., \\boxed{\\frac{1}{2}}).

    Args:
        text: The text containing \\boxed{answer}

    Returns:
        The extracted answer from the last \\boxed{} or None if not found
    """
    # Find all \boxed{ occurrences
    start_pattern = r'\\boxed\{'
    matches = list(re.finditer(start_pattern, text))

    if not matches:
        return None

    # Process the LAST match (final answer convention)
    last_match = matches[-1]

    # Start after \boxed{
    start_idx = last_match.end()
    brace_count = 1
    idx = start_idx

    # Find the matching closing brace
    while idx < len(text) and brace_count > 0:
        if text[idx] == '{':
            brace_count += 1
        elif text[idx] == '}':
            brace_count -= 1
        idx += 1

    if brace_count == 0:
        # Extract content between the braces
        return text[start_idx:idx-1].strip()
    
    #Add fallback if there is no \boxed in model's answer

    return None


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

        # Aime answers are 0-999 int, so we can direclty compare them, if model_answer is not int then it is incorrect
        is_correct = expected == model_answer

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
