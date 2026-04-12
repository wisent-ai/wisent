"""Generation-based evaluator for benchmarks that require text generation.

Uses semantic similarity (NLI + embeddings) for robust text matching.
"""

from typing import Any
import logging

from wisent.core.reading.evaluators.core.atoms import BaseEvaluator, EvalResult
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

    Compares responses directly using text matching and semantic similarity (NLI + embeddings).
    """

    name = "generation"
    description = "Generation-based evaluator for text generation tasks"

    def __init__(
        self,
        *,
        generation_embedding_weight: float,
        generation_nli_weight: float,
    ):
        self._generation_embedding_weight = generation_embedding_weight
        self._generation_nli_weight = generation_nli_weight

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate generated response against expected answer.

        Args:
            response: Generated model response.
            expected: Expected answer (str, or list of acceptable answers).
            **kwargs:
                task_name: Task name.
                normalize: Whether to normalize strings before comparison.
                choices: List of [correct_answer, incorrect_answer].
                correct_answers: List of correct reference answers.
                incorrect_answers: List of incorrect reference answers.

        Returns:
            EvalResult with TRUTHFUL/UNTRUTHFUL/UNKNOWN.
        """
        task_name = kwargs.get('task_name', '')
        normalize = kwargs.get('normalize', True)
        correct_answers = kwargs.get('correct_answers', [])
        incorrect_answers = kwargs.get('incorrect_answers', [])
        choices = kwargs.get('choices')

        # Only use _evaluate_choices when no correct/incorrect references
        # exist — otherwise _compare_to_references gives better signal by
        # actually comparing the generated response.
        if choices and len(choices) == 2 and not correct_answers and not incorrect_answers:
            return self._evaluate_choices(choices, expected, normalize)

        if response and correct_answers and incorrect_answers:
            return self._compare_to_references(
                response, correct_answers, incorrect_answers, task_name, "text",
                embedding_weight=self._generation_embedding_weight,
                nli_weight=self._generation_nli_weight,
            )

        expected_answers = expected if isinstance(expected, list) else [expected]
        is_correct, matched_answer, confidence = self._check_text_match(
            response, expected_answers, normalize,
        )

        return EvalResult(
            ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
            method_used=self.name,
            confidence=confidence,
            details=f"Response: '{response[:100]}', Expected: '{matched_answer or expected_answers[0]}'",
        )

    def _evaluate_choices(self, choices, expected, normalize):
        """Evaluate contrastive choices against expected answer."""
        correct_answer = choices[0]
        incorrect_answer = choices[1]
        expected_answers = expected if isinstance(expected, list) else [expected]

        correct_matches, _, conf_correct = self._check_text_match(
            correct_answer, expected_answers, normalize,
        )
        incorrect_matches, _, conf_incorrect = self._check_text_match(
            incorrect_answer, expected_answers, normalize,
        )

        meta = {"correct_answer": correct_answer, "incorrect_answer": incorrect_answer, "expected": expected}

        if correct_matches and not incorrect_matches:
            return EvalResult(ground_truth="TRUTHFUL", method_used=self.name, confidence=conf_correct,
                            details=f"Correct answer matches expected", meta=meta)
        elif incorrect_matches and not correct_matches:
            return EvalResult(ground_truth="UNTRUTHFUL", method_used=self.name, confidence=conf_incorrect,
                            details=f"Incorrect answer matches expected", meta=meta)
        else:
            return EvalResult(ground_truth="UNKNOWN", method_used=self.name, confidence=0.0,
                            details=f"Ambiguous: correct={correct_matches}, incorrect={incorrect_matches}", meta=meta)

    def _check_text_match(self, text: str, expected_list: list, normalize: bool) -> tuple:
        """Check text match using normalization and semantic similarity (NLI + embeddings)."""
        if text is None:
            return False, None, 0.0
        text_norm = self.normalize_text(text) if normalize else text
        for expected in expected_list:
            expected_str = str(expected)
            expected_norm = self.normalize_text(expected_str) if normalize else expected_str
            if text_norm == expected_norm:
                return True, expected, 1.0
            if text_norm in expected_norm or expected_norm in text_norm:
                return True, expected, 0.9

        for expected in expected_list:
            expected_str = str(expected)
            nli_score = self._nli_entailment(text, expected_str)
            if nli_score is not None and nli_score >= 0.5:
                confidence = min(0.85, 0.6 + nli_score * 0.3)
                return True, expected, confidence
            emb_score = self._embedding_similarity(text, expected_str)
            if emb_score is not None and emb_score >= 0.6:
                confidence = min(0.8, 0.5 + emb_score * 0.3)
                return True, expected, confidence
        return False, None, 0.0
