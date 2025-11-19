"""Extractor for Meddialog benchmark."""
from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["MeddialogExtractor"]


class MeddialogExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Meddialog benchmark.

    Meddialog evaluates question extraction from medical dialogues.
    The task is to extract and summarize the main medical question from
    a patient-doctor dialogue.
    """

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from Meddialog dataset.

        Args:
            limit: Optional maximum number of pairs to extract

        Returns:
            List of ContrastivePair objects
        """
        # Load dataset from HuggingFace
        docs = self.load_dataset(
            dataset_name="lighteval/med_dialog",
            dataset_config="icliniq",
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
        """Convert a single Meddialog doc into a ContrastivePair.

        Args:
            doc: Document with 'src' (dialogue), 'tgt' (extracted question), 'id'

        Returns:
            ContrastivePair or None if doc is malformed
        """
        src = str(doc.get("src", "")).strip()
        tgt = str(doc.get("tgt", "")).strip()

        if not src or not tgt:
            return None

        # Task: extract and summarize the medical question from dialogue
        # Positive: the actual extracted question
        # Negative: a generic non-specific summary
        prompt = f"Extract and summarize the medical question from the following dialogue:\n\n{src}\n\nQuestion:"

        positive_response = PositiveResponse(model_response=tgt)
        negative_response = NegativeResponse(model_response="What is the patient's medical concern?")

        return ContrastivePair(
            prompt=prompt,
            positive_response=positive_response,
            negative_response=negative_response,
            label="meddialog",
            metadata={"id": doc.get("id")},
        )
