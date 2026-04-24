"""Helper methods for GenerationEvaluator: reference comparison and semantic matching."""

from __future__ import annotations
from typing import Any

from wisent.core.reading.evaluators.core.atoms import EvalResult
from wisent.core.utils.config_tools.constants import (
    EVAL_MIN_LEXICAL_DIVERSITY,
    ROUNDING_PRECISION,
)

__all__ = ["GenerationEvaluatorHelpersMixin"]


class GenerationEvaluatorHelpersMixin:
    """Mixin providing reference comparison and semantic matching for generation evaluator."""

    def _compare_to_references(
        self,
        response: str,
        correct_answers: list,
        incorrect_answers: list,
        task_name: str,
        answer_type: str,
        *,
        embedding_weight: float,
        nli_weight: float,
    ) -> EvalResult:
        """Compare model response to both correct and incorrect reference answers.

        Uses semantic similarity to determine which reference the response is closer to.
        Returns TRUTHFUL if closer to correct, UNTRUTHFUL if closer to incorrect.
        """
        # Reject repetitive garbage before running semantic comparison.
        # Pure repetition (diversity < 0.15): "within within within..."
        # Stop-word soup (content ratio < 0.5): "nor nor except other..."
        # Verbose coherent text has diversity 0.40+ and content ratio 0.50+.
        words = response.lower().split()
        if words:
            diversity = len(set(words)) / len(words)
            _STOP = frozenset({
                "nor", "not", "no", "the", "a", "an", "and", "or", "but",
                "is", "are", "was", "were", "be", "been", "being", "have",
                "has", "had", "do", "does", "did", "will", "would", "could",
                "should", "may", "might", "shall", "can", "to", "of", "in",
                "for", "on", "with", "at", "by", "from", "as", "into",
                "through", "during", "before", "after", "it", "its", "this",
                "that", "these", "those", "i", "you", "he", "she", "we",
                "they", "me", "him", "her", "us", "them", "my", "your",
                "his", "our", "their", "except", "other", "anything",
            })
            content_words = [w for w in words if w not in _STOP]
            content_ratio = len(content_words) / len(words) if words else 0
            is_pure_repetition = diversity < EVAL_MIN_LEXICAL_DIVERSITY / 3
            is_stopword_soup = content_ratio < EVAL_MIN_LEXICAL_DIVERSITY
            if is_pure_repetition or is_stopword_soup:
                return EvalResult(
                    ground_truth="UNTRUTHFUL",
                    method_used=self.name,
                    confidence=0.0,
                    details=f"Repetitive response (diversity={diversity:.3f}, content={content_ratio:.3f})",
                    meta={"lexical_diversity": round(diversity, ROUNDING_PRECISION)},
                )

        # Get max similarity to correct answers
        max_correct_sim = 0.0
        best_correct = None
        for correct in correct_answers:
            sim = self._embedding_similarity(response, str(correct))
            if sim is not None and sim > max_correct_sim:
                max_correct_sim = sim
                best_correct = correct

        # Get max similarity to incorrect answers
        max_incorrect_sim = 0.0
        best_incorrect = None
        for incorrect in incorrect_answers:
            sim = self._embedding_similarity(response, str(incorrect))
            if sim is not None and sim > max_incorrect_sim:
                max_incorrect_sim = sim
                best_incorrect = incorrect

        # Also check NLI entailment
        nli_correct = 0.0
        for correct in correct_answers:
            nli = self._nli_entailment(response, str(correct))
            if nli is not None and nli > nli_correct:
                nli_correct = nli

        nli_incorrect = 0.0
        for incorrect in incorrect_answers:
            nli = self._nli_entailment(response, str(incorrect))
            if nli is not None and nli > nli_incorrect:
                nli_incorrect = nli

        # Combine scores (weighted average)
        score_correct = embedding_weight * max_correct_sim + nli_weight * nli_correct
        score_incorrect = embedding_weight * max_incorrect_sim + nli_weight * nli_incorrect

        margin = score_correct - score_incorrect

        meta = {
            "embedding_sim_correct": round(max_correct_sim, ROUNDING_PRECISION),
            "embedding_sim_incorrect": round(max_incorrect_sim, ROUNDING_PRECISION),
            "nli_correct": round(nli_correct, ROUNDING_PRECISION),
            "nli_incorrect": round(nli_incorrect, ROUNDING_PRECISION),
            "score_correct": round(score_correct, ROUNDING_PRECISION),
            "score_incorrect": round(score_incorrect, ROUNDING_PRECISION),
            "margin": round(margin, ROUNDING_PRECISION),
            "best_correct_match": best_correct,
            "best_incorrect_match": best_incorrect,
        }

        if margin > 0.05:
            confidence = min(0.95, 0.6 + margin)
            return EvalResult(
                ground_truth="TRUTHFUL",
                method_used=self.name,
                confidence=confidence,
                details=f"Response closer to truthful (margin={margin:.3f})",
                meta=meta,
            )
        elif margin < -0.05:
            confidence = min(0.95, 0.6 + abs(margin))
            return EvalResult(
                ground_truth="UNTRUTHFUL",
                method_used=self.name,
                confidence=confidence,
                details=f"Response closer to false (margin={margin:.3f})",
                meta=meta,
            )
        else:
            return EvalResult(
                ground_truth="UNTRUTHFUL",
                method_used=self.name,
                confidence=0.0,
                details=f"Response ambiguous (margin={margin:.3f})",
                meta=meta,
            )

    def _nli_entailment(self, response: str, expected: str) -> float:
        """Check if response entails expected using NLI cross-encoder."""
        try:
            from wisent.core.reading.evaluators.benchmark_specific.generation_evaluator import _get_cross_encoder
            ce = _get_cross_encoder()
            import torch
            import torch.nn.functional as F

            # Check both directions
            pairs = [(response, expected), (expected, response)]
            logits = torch.tensor(ce.predict(pairs))
            probs = F.softmax(logits, dim=-1).tolist()

            # Return max entailment score from both directions
            ent_forward = probs[0][1]  # response entails expected
            ent_backward = probs[1][1]  # expected entails response
            return max(ent_forward, ent_backward)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"NLI entailment failed: {e}")
            return None

    def _embedding_similarity(self, response: str, expected: str) -> float:
        """Compute cosine similarity between response and expected using embeddings."""
        try:
            from wisent.core.reading.evaluators.benchmark_specific.generation_evaluator import _get_embedding_model
            emb = _get_embedding_model()
            import torch

            vecs = emb.encode([response, expected], convert_to_tensor=True, normalize_embeddings=True)
            similarity = torch.matmul(vecs[0], vecs[1]).item()
            return similarity
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Embedding similarity failed: {e}")
            return None
