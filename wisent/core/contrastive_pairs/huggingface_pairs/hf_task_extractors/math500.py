from __future__ import annotations

from typing import Any
import logging

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

from latex2sympy2_extended import latex2sympy
from sympy import latex
from wisent.core.evaluators.benchmark_specific.math_parsing.scripts import strip_string

__all__ = ["MATH500Extractor"]

log = logging.getLogger(__name__)

task_names = ("math500",)

class MATH500Extractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for MATH dataset (competition mathematics problems).

    MATH schema (HuggingFaceH4/MATH-500):
        - problem: str (math problem statement)
        - solution: str (detailed solution with LaTeX)
        - answer: str (final answer)
        - subject: str (problem type: Algebra, Geometry, etc.)
        - level: str (difficulty level 1-5)
        - unique_id: str (unique identifier)
    """


    evaluator_name = "math"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from MATH examples.

        For math tasks, we create pairs where:
        - Positive: Correct numerical answer
        - Negative: Incorrect numerical answer (off by 1 or modified)

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load MATH dataset
        docs = self.load_dataset(
            dataset_name="HuggingFaceH4/MATH-500",
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} MATH examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid MATH pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single MATH doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            problem = doc.get("problem", "").strip()
            answer = doc.get("answer", "").strip()
            level = doc.get("level", "")
            subject = doc.get("subject", "")

            if not problem or not answer:
                log.debug("Skipping: missing problem or answer")
                return None

            # Strip the answer
            correct_answer = strip_string(answer)
            if not correct_answer:
                correct_answer = answer

            # Create incorrect answer (add 1 or modify)
            incorrect_answer = self._create_incorrect_answer(correct_answer)

            # Format the question
            question = f"Question: {problem}\n\nWhat is the answer?"

            metadata = {
                "label": "math",
                "level": level,
                "subject": subject,
            }

            return self._build_pair(
                question=question,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _create_incorrect_answer(self, correct: str) -> str:
        """Create an incorrect answer by modifying the correct one (input is already stripped)."""
        try:
            parsed_correct = latex2sympy(correct)
            incorrect = latex(parsed_correct + 1)
            return str(incorrect)
        except Exception:
            return f"{correct} + 1"

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        """Build a ContrastivePair from question and responses."""
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None,
            metadata=metadata,
        )
