from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

from latex2sympy2_extended import latex2sympy
from sympy import latex
from wisent.core.evaluators.benchmark_specific.math_parsing.scripts import strip_string

__all__ = ["HMMTExtractor"]

log = setup_logger(__name__)

task_names = ("hmmt", "hmmt_feb_2025")

class HMMTExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for HMMT dataset (Harvard-MIT Math Tournament).

    HMMT schema (MathArena/hmmt_feb_2025):
        - problem: str (math problem statement)
        - answer: str or int (final answer)
    """


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from HMMT examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load HMMT dataset
        docs = self.load_dataset(
            dataset_name="MathArena/hmmt_feb_2025",
            split="train",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} HMMT examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid HMMT pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single HMMT doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            problem = doc.get("problem", "").strip()
            answer = doc.get("answer", "")

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
                "label": "hmmt",
                "source": "MathArena/hmmt_feb_2025",
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

