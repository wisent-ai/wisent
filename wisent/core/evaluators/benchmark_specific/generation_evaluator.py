"""Generation-based evaluator for benchmarks that require text generation.

This evaluator handles tasks like GSM8K, DROP, TriviaQA, TruthfulQA where the model generates
free-form text that must be parsed and compared to reference answers.

Uses semantic similarity (NLI + embeddings) for robust text matching.
"""

import re
from typing import Any, Dict
import logging

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.errors import NumericalExtractionError, TextExtractionError

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

    Note: Task names are declared in extractors (via evaluator_name attribute),
    not in evaluators. This maintains proper separation of concerns.
    """

    name = "generation"
    description = "Generation-based evaluator for text generation tasks"

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate generated response against expected answer.

        Args:
            response: Generated model response (can be empty if using choices)
            expected: Expected answer (str, int, float, or list of acceptable answers)
            **kwargs:
                task_name: Task name for task-specific logic
                answer_type: Type of answer ("numerical", "text", "exact")
                normalize: Whether to normalize strings before comparison
                choices: List of [correct_answer, incorrect_answer] for contrastive evaluation
                model: WisentModel instance for generating responses

        Returns:
            EvalResult with TRUTHFUL/UNTRUTHFUL/UNKNOWN
        """
        task_name = kwargs.get('task_name', '')
        answer_type = kwargs.get('answer_type', 'text')
        normalize = kwargs.get('normalize', True)
        choices = kwargs.get('choices')
        model = kwargs.get('model')
        question = kwargs.get('question', '')

        # If we have choices, do direct comparison (for contrastive evaluation)
        if choices and len(choices) == 2:
            correct_answer = choices[0]
            incorrect_answer = choices[1]

            # Determine answer type based on task or if it looks numerical
            if task_name in ["gsm8k", "asdiv", "arithmetic", "math", "math_500", "math500", "aime", "aime2024", "aime2025", "hmmt", "hmmt_feb_2025", "polymath", "polymath_en_medium", "polymath_zh_medium", "polymath_en_high", "polymath_zh_high", "livemathbench", "livemathbench_cnmo_en", "livemathbench_cnmo_zh", "hendrycks_math"]:
                answer_type = "numerical"

            # For generation tasks, we compare the pre-generated answers directly
            # Expected is what we're comparing against
            # Extract answers from both choices
            extracted_correct = self._extract_answer(correct_answer, task_name, answer_type)
            extracted_incorrect = self._extract_answer(incorrect_answer, task_name, answer_type)
            extracted_expected = self._extract_answer(str(expected), task_name, answer_type)

            if extracted_expected is None:
                return EvalResult(
                    ground_truth="UNKNOWN",
                    method_used=self.name,
                    confidence=0.0,
                    details="Could not extract answer from expected response",
                )

            # Check which choice matches the expected answer
            if isinstance(expected, list):
                expected_answers = expected
            else:
                expected_answers = [expected]

            # For numerical answers, we've already extracted, so compare directly
            if answer_type == "numerical":
                # Compare extracted numbers directly
                correct_matches = (extracted_correct is not None and
                                 extracted_expected is not None and
                                 abs(extracted_correct - extracted_expected) < 1e-6)
                incorrect_matches = (extracted_incorrect is not None and
                                   extracted_expected is not None and
                                   abs(extracted_incorrect - extracted_expected) < 1e-6)
                conf_correct = 1.0 if correct_matches else 0.0
                conf_incorrect = 1.0 if incorrect_matches else 0.0
            else:
                # For text answers, check match against the list
                correct_matches, _, conf_correct = self._check_match(
                    extracted_correct, expected_answers, answer_type, normalize
                )
                incorrect_matches, _, conf_incorrect = self._check_match(
                    extracted_incorrect, expected_answers, answer_type, normalize
                )

            # If correct matches expected, return TRUTHFUL
            # If incorrect matches expected, return UNTRUTHFUL
            if correct_matches and not incorrect_matches:
                return EvalResult(
                    ground_truth="TRUTHFUL",
                    method_used=self.name,
                    confidence=conf_correct,
                    details=f"Correct answer '{correct_answer}' matches expected '{expected}'",
                    meta={
                        "correct_answer": correct_answer,
                        "incorrect_answer": incorrect_answer,
                        "expected": expected,
                    }
                )
            elif incorrect_matches and not correct_matches:
                return EvalResult(
                    ground_truth="UNTRUTHFUL",
                    method_used=self.name,
                    confidence=conf_incorrect,
                    details=f"Incorrect answer '{incorrect_answer}' matches expected '{expected}'",
                    meta={
                        "correct_answer": correct_answer,
                        "incorrect_answer": incorrect_answer,
                        "expected": expected,
                    }
                )
            else:
                # Both match or neither matches - unclear
                return EvalResult(
                    ground_truth="UNKNOWN",
                    method_used=self.name,
                    confidence=0.0,
                    details=f"Ambiguous: correct={correct_matches}, incorrect={incorrect_matches}",
                    meta={
                        "correct_answer": correct_answer,
                        "incorrect_answer": incorrect_answer,
                        "expected": expected,
                    }
                )

        # If no choices but we have a response and expected, do relative comparison
        # Check which reference answer the model output is closer to
        correct_answers = kwargs.get('correct_answers', [])
        incorrect_answers = kwargs.get('incorrect_answers', [])

        if response and correct_answers and incorrect_answers:
            return self._compare_to_references(
                response, correct_answers, incorrect_answers, task_name, answer_type
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
        # Handle LaTeX fractions: \frac{a}{b}
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

        # Handle LaTeX degree notation: number^{\circ} or number^\circ
        degree_pattern = r'([-+]?\d+\.?\d*)\^\{?\\circ\}?'
        degree_match = re.search(degree_pattern, response)
        if degree_match:
            try:
                return float(degree_match.group(1))
            except ValueError:
                pass

        # Look for common patterns (specific markers only)
        patterns = [
            r'####\s*([-+]?\d*\.?\d+)',  # GSM8K format: #### 42
            r'answer\s*is\s*:?\s*([-+]?\d*\.?\d+)',  # "answer is 42" or "answer is: 42"
            r'=\s*([-+]?\d*\.?\d+)\s*$',  # Ends with "= 42"
            r'^\s*([-+]?\d*\.?\d+)\s*$',  # Just a number by itself (entire response)
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        # No fallback - raise error if numerical answer cannot be extracted
        raise NumericalExtractionError(response=response)

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

        # No fallback - raise error if text answer cannot be extracted
        raise TextExtractionError(response=response)

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
        """Check text match using semantic similarity (NLI + embeddings).

        Uses a multi-stage approach:
        1. Exact/substring match (high confidence)
        2. NLI entailment check (medium-high confidence)
        3. Embedding similarity (medium confidence)
        """
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

            # Stage 1: Exact match
            if extracted_norm == expected_norm:
                return True, expected, 1.0

            # Stage 2: Substring match
            if extracted_norm in expected_norm or expected_norm in extracted_norm:
                return True, expected, 0.9

        # Stage 3: Semantic similarity using NLI + embeddings
        for expected in expected_list:
            expected_str = str(expected)

            # Try NLI entailment
            nli_score = self._nli_entailment(extracted, expected_str)
            if nli_score is not None and nli_score >= 0.5:
                confidence = min(0.85, 0.6 + nli_score * 0.3)
                return True, expected, confidence

            # Try embedding similarity
            emb_score = self._embedding_similarity(extracted, expected_str)
            if emb_score is not None and emb_score >= 0.6:
                confidence = min(0.8, 0.5 + emb_score * 0.3)
                return True, expected, confidence

        return False, None, 0.0

    def _nli_entailment(self, response: str, expected: str) -> float:
        """Check if response entails expected using NLI cross-encoder."""
        try:
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
            logger.debug(f"NLI entailment failed: {e}")
            return None

    def _embedding_similarity(self, response: str, expected: str) -> float:
        """Compute cosine similarity between response and expected using embeddings."""
        try:
            emb = _get_embedding_model()
            import torch

            vecs = emb.encode([response, expected], convert_to_tensor=True, normalize_embeddings=True)
            similarity = torch.matmul(vecs[0], vecs[1]).item()
            return similarity
        except Exception as e:
            logger.debug(f"Embedding similarity failed: {e}")
            return None

    def _compare_to_references(
        self,
        response: str,
        correct_answers: list,
        incorrect_answers: list,
        task_name: str,
        answer_type: str
    ) -> EvalResult:
        """Compare model response to both correct and incorrect reference answers.

        Uses semantic similarity to determine which reference the response is closer to.
        Returns TRUTHFUL if closer to correct, UNTRUTHFUL if closer to incorrect.
        """
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
        score_correct = 0.4 * max_correct_sim + 0.6 * nli_correct
        score_incorrect = 0.4 * max_incorrect_sim + 0.6 * nli_incorrect

        margin = score_correct - score_incorrect

        meta = {
            "embedding_sim_correct": round(max_correct_sim, 3),
            "embedding_sim_incorrect": round(max_incorrect_sim, 3),
            "nli_correct": round(nli_correct, 3),
            "nli_incorrect": round(nli_incorrect, 3),
            "score_correct": round(score_correct, 3),
            "score_incorrect": round(score_incorrect, 3),
            "margin": round(margin, 3),
            "best_correct_match": best_correct,
            "best_incorrect_match": best_incorrect,
        }

        # Require minimum margin for confident decision
        MIN_MARGIN = 0.05

        if margin > MIN_MARGIN:
            confidence = min(0.95, 0.6 + margin)
            return EvalResult(
                ground_truth="TRUTHFUL",
                method_used=self.name,
                confidence=confidence,
                details=f"Response closer to truthful (margin={margin:.3f})",
                meta=meta,
            )
        elif margin < -MIN_MARGIN:
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
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details=f"Response ambiguous (margin={margin:.3f})",
                meta=meta,
            )
