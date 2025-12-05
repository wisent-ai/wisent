from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

from latex2sympy2_extended import latex2sympy
from sympy import latex
from wisent.core.evaluators.benchmark_specific.math_parsing.scripts import strip_string
import re

__all__ = ["MATHExtractor"]

log = setup_logger(__name__)

task_names = ("math",)

class MATHExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for MATH dataset (competition mathematics problems).

    MATH schema (qwedsacf/competition_math):
        - problem: str (math problem statement)
        - level str (level of difficulty of the problem, Level 1 - Level 5)
        - type str (problem type: Algebra, Geometry, etc.)
        - solution: str (detailed solution with LaTeX)
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
            dataset_name="qwedsacf/competition_math",
            split="train",
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
            solution = doc.get("solution", "").strip()

            if not problem or not solution:
                log.debug("Skipping: missing problem or solution")
                return None

            # Extract and strip the answer
            raw_answer = self.extract_boxed_answer(solution)
            if not raw_answer:
                log.debug("Skipping: no boxed answer found")
                return None

            correct_answer = strip_string(raw_answer)
            if not correct_answer:
                correct_answer = raw_answer

            # Create incorrect answer (add 1 or modify)
            incorrect_answer = self._create_incorrect_answer(correct_answer)

            # Format the question
            question = f"Question: {problem}\n\nWhat is the answer?"

            metadata = {
                "label": "math",
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

    def extract_boxed_answer(self, text: str) -> str | None:
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

        return None

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
