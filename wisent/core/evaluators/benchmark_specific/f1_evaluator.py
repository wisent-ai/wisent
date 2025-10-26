"""F1 score evaluator for benchmarks requiring token-level comparison.

Used for reading comprehension tasks where partial credit is appropriate.
"""

from typing import Any, Set
import logging

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)


class F1Evaluator(BaseEvaluator):
    """Evaluator using F1 score for token-level comparison.

    Compatible with:
    - DROP: Reading comprehension with discrete reasoning
    - SQuAD: Question answering
    - Any task where partial token overlap should be rewarded
    """

    name = "f1"
    description = "F1 score evaluator for token-level comparison"
    task_names = ("drop", "squad")

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate using F1 score.

        Args:
            response: Generated answer
            expected: Expected answer (can be list of acceptable answers)
            **kwargs:
                normalize: Whether to normalize tokens (default: True)

        Returns:
            EvalResult with F1 score as confidence
        """
        normalize = kwargs.get('normalize', True)

        # Handle list of acceptable answers - use best F1
        if isinstance(expected, list):
            expected_answers = expected
        else:
            expected_answers = [expected]

        best_f1 = 0.0
        best_match = None

        for exp in expected_answers:
            f1 = self._compute_f1(response, str(exp), normalize)
            if f1 > best_f1:
                best_f1 = f1
                best_match = exp

        # Determine ground truth based on F1 threshold
        if best_f1 >= 0.8:
            ground_truth = "TRUTHFUL"
        elif best_f1 >= 0.5:
            ground_truth = "UNKNOWN"  # Partial match
        else:
            ground_truth = "UNTRUTHFUL"

        return EvalResult(
            ground_truth=ground_truth,
            method_used=self.name,
            confidence=best_f1,
            details=f"F1 score: {best_f1:.3f} (response vs '{best_match}')",
            meta={"f1_score": best_f1, "matched_answer": best_match}
        )

    def _compute_f1(self, response: str, expected: str, normalize: bool = True) -> float:
        """Compute F1 score between response and expected.

        Returns:
            F1 score in [0, 1]
        """
        # Tokenize
        response_tokens = self._tokenize(response, normalize)
        expected_tokens = self._tokenize(expected, normalize)

        if not response_tokens or not expected_tokens:
            return 0.0

        # Compute precision and recall
        common = response_tokens & expected_tokens
        if not common:
            return 0.0

        precision = len(common) / len(response_tokens)
        recall = len(common) / len(expected_tokens)

        # Compute F1
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def _tokenize(self, text: str, normalize: bool = True) -> Set[str]:
        """Tokenize text into set of tokens."""
        if normalize:
            text = self.normalize_text(text)
        return set(text.split())
