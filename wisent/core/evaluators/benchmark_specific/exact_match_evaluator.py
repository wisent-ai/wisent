"""Exact match evaluator for benchmarks requiring exact string matching.

Used for tasks like GSM8K, TriviaQA where the answer must match exactly
(possibly after normalization).
"""

from typing import Any
import logging

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)


class ExactMatchEvaluator(BaseEvaluator):
    """Evaluator using exact match for answer comparison.

    Compatible with:
    - GSM8K: Math problems (numerical exact match)
    - TriviaQA: Factual questions (text exact match)
    - Any task requiring exact answer matching

    Note: Task names are declared in extractors (via evaluator_name attribute),
    not in evaluators. This maintains proper separation of concerns.
    """

    name = "exact_match"
    description = "Exact match evaluator for precise answer comparison"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate using exact match.

        Args:
            response: Generated or extracted answer
            expected: Expected answer (can be list of acceptable answers)
            **kwargs:
                normalize: Whether to normalize before comparison (default: True)
                case_sensitive: Whether match is case-sensitive (default: False)

        Returns:
            EvalResult with TRUTHFUL/UNTRUTHFUL
        """
        normalize = kwargs.get('normalize', True)
        case_sensitive = kwargs.get('case_sensitive', False)

        # Handle list of acceptable answers
        if isinstance(expected, list):
            expected_answers = expected
        else:
            expected_answers = [expected]

        # Prepare response
        response_clean = str(response).strip()
        if normalize:
            response_clean = self.normalize_text(response_clean)
        if not case_sensitive:
            response_clean = response_clean.lower()

        # Check each expected answer
        for exp in expected_answers:
            exp_clean = str(exp).strip()
            if normalize:
                exp_clean = self.normalize_text(exp_clean)
            if not case_sensitive:
                exp_clean = exp_clean.lower()

            if response_clean == exp_clean:
                return EvalResult(
                    ground_truth="TRUTHFUL",
                    method_used=self.name,
                    confidence=1.0,
                    details=f"Exact match: '{response}' == '{exp}'",
                    meta={"matched_answer": exp}
                )

        return EvalResult(
            ground_truth="UNTRUTHFUL",
            method_used=self.name,
            confidence=0.0,
            details=f"No match: '{response}' not in {expected_answers}",
        )
