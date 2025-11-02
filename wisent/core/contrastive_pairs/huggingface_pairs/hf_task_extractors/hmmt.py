from __future__ import annotations

from typing import Any
import logging

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["HMMTExtractor"]

log = logging.getLogger(__name__)


class HMMTExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for HMMT dataset (Harvard-MIT Math Tournament).

    HMMT schema (MathArena/hmmt_feb_2025):
        - problem: str (math problem statement)
        - answer: str or int (final answer)
    """

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

            # Convert answer to string
            correct_answer = str(answer).strip()

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
        """Create an incorrect answer by modifying the correct one."""
        try:
            clean = correct.replace('$', '').replace(',', '').strip()
            num = int(clean)
            return str(num + 1)
        except ValueError:
            try:
                num = float(clean)
                return str(num + 1)
            except ValueError:
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
