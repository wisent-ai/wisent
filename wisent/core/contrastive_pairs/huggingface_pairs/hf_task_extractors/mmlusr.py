"""Extractor for MMLU-SR (Stress-Testing) benchmarks from HuggingFace."""

from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

__all__ = ["MMLUSRExtractor"]
_LOG = setup_logger(__name__)


class MMLUSRExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for MMLU-SR (Stress-Testing) benchmarks."""

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from MMLU-SR dataset.

        Args:
            limit: Maximum number of pairs to extract.

        Returns:
            List of contrastive pairs.
        """
        # All three variants (answer_only, question_only, question_and_answer) have the same format
        # So we load from 'answer_only' config
        docs = self.load_dataset(
            dataset_name="NiniCat/MMLU-SR",
            dataset_config="answer_only",
            split="test",
            limit=limit,
        )

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single MMLU-SR doc into a ContrastivePair."""
        log = bind(_LOG, doc_id=doc.get("__index__", "unknown"))

        try:
            # MMLU-SR format: column_0 (question), column_1-4 (choices), column_5 (answer letter)
            question = str(doc.get("column_0", "")).strip()
            choices = [
                str(doc.get("column_1", "")).strip(),
                str(doc.get("column_2", "")).strip(),
                str(doc.get("column_3", "")).strip(),
                str(doc.get("column_4", "")).strip(),
            ]
            answer_letter = str(doc.get("column_5", "")).strip().upper()

            if not question or not all(choices) or not answer_letter:
                log.debug("Skipping: missing question, choices, or answer")
                return None

            # Convert answer letter to index
            if answer_letter not in ['A', 'B', 'C', 'D']:
                log.debug(f"Skipping: invalid answer letter '{answer_letter}'")
                return None

            answer_index = ord(answer_letter) - ord('A')

            if not (0 <= answer_index < len(choices)):
                log.debug(f"Skipping: answer index {answer_index} out of range")
                return None

            correct_answer = choices[answer_index]

            # Get an incorrect answer (any other option)
            incorrect_index = (answer_index + 1) % len(choices)
            incorrect_answer = choices[incorrect_index]

            # Format question with options
            formatted_question = f"Question: {question}\nOptions:\n"
            for i, choice in enumerate(choices):
                formatted_question += f"{chr(ord('A') + i)}. {choice}\n"
            formatted_question += "Answer:"

            metadata = {
                "label": "mmlusr",
            }

            positive_response = PositiveResponse(model_response=correct_answer)
            negative_response = NegativeResponse(model_response=incorrect_answer)

            return ContrastivePair(
                prompt=formatted_question.strip(),
                positive_response=positive_response,
                negative_response=negative_response,
                label=metadata.get("label"),
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None
