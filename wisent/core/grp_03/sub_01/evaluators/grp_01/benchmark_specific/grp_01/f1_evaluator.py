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

    Note: Task names are declared in extractors (via evaluator_name attribute),
    not in evaluators. This maintains proper separation of concerns.
    """

    name = "f1"
    description = "F1 score evaluator for token-level comparison"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate using F1 score.

        Args:
            response: Generated answer (can be empty if using choices)
            expected: Expected answer (can be list of acceptable answers)
            **kwargs:
                normalize: Whether to normalize tokens (default: True)
                choices: List of [correct_answer, incorrect_answer] for contrastive evaluation
                model: WisentModel instance for generating responses

        Returns:
            EvalResult with F1 score as confidence
        """
        normalize = kwargs.get('normalize', True)
        choices = kwargs.get('choices')

        # If we have choices, do contrastive comparison
        if choices and len(choices) == 2:
            correct_answer = choices[0]
            incorrect_answer = choices[1]

            # Handle list of acceptable answers
            if isinstance(expected, list):
                expected_answers = expected
            else:
                expected_answers = [expected]

            # Compute F1 scores for both choices against expected
            best_f1_correct = 0.0
            best_f1_incorrect = 0.0

            for exp in expected_answers:
                f1_correct = self._compute_f1(str(correct_answer), str(exp), normalize)
                f1_incorrect = self._compute_f1(str(incorrect_answer), str(exp), normalize)

                if f1_correct > best_f1_correct:
                    best_f1_correct = f1_correct
                if f1_incorrect > best_f1_incorrect:
                    best_f1_incorrect = f1_incorrect

            # If correct answer has better F1 with expected, return TRUTHFUL
            # If incorrect answer has better F1 with expected, return UNTRUTHFUL
            if best_f1_correct >= 0.8 and best_f1_incorrect < 0.8:
                return EvalResult(
                    ground_truth="TRUTHFUL",
                    method_used=self.name,
                    confidence=best_f1_correct,
                    details=f"Correct answer F1: {best_f1_correct:.3f}, Incorrect F1: {best_f1_incorrect:.3f}",
                    meta={
                        "correct_f1": best_f1_correct,
                        "incorrect_f1": best_f1_incorrect,
                        "correct_answer": correct_answer,
                        "incorrect_answer": incorrect_answer,
                    }
                )
            elif best_f1_incorrect >= 0.8 and best_f1_correct < 0.8:
                return EvalResult(
                    ground_truth="UNTRUTHFUL",
                    method_used=self.name,
                    confidence=best_f1_incorrect,
                    details=f"Incorrect answer F1: {best_f1_incorrect:.3f}, Correct F1: {best_f1_correct:.3f}",
                    meta={
                        "correct_f1": best_f1_correct,
                        "incorrect_f1": best_f1_incorrect,
                        "correct_answer": correct_answer,
                        "incorrect_answer": incorrect_answer,
                    }
                )
            else:
                # Ambiguous - both have high or both have low F1
                return EvalResult(
                    ground_truth="UNKNOWN",
                    method_used=self.name,
                    confidence=max(best_f1_correct, best_f1_incorrect),
                    details=f"Ambiguous: Correct F1={best_f1_correct:.3f}, Incorrect F1={best_f1_incorrect:.3f}",
                    meta={
                        "correct_f1": best_f1_correct,
                        "incorrect_f1": best_f1_incorrect,
                        "correct_answer": correct_answer,
                        "incorrect_answer": incorrect_answer,
                    }
                )

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
