from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["AIMEExtractor"]

log = setup_logger(__name__)

task_names = ("aime",)

class AIMEExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for AIME dataset (American Invitational Mathematics Examination).

    AIME schema:
        - Question: str (math problem statement)
        - Answer: int 0-999
    """


    evaluator_name = "aime"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from AIME examples.

        For AIME tasks, we create pairs where:
        - Positive: Correct numerical answer
        - Negative: Incorrect numerical answer (off by 1 or modified)

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load AIME dataset
        docs = self.load_dataset(
            dataset_name="gneubig/aime-1983-2024",
            split="train",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} AIME examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid AIME pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single AIME doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            question = doc.get("Question", "").strip()
            correct = doc.get("Answer", "").strip()

            if not question or not correct:
                log.debug("Skipping: missing problem or answer")
                return None

            incorrect = str(int(correct) + 1)

            question = f"Question: {question}\n\nWhat is the answer?"

            metadata = {
                "label": "aime",
            }

            return self._build_pair(
                question=question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
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
