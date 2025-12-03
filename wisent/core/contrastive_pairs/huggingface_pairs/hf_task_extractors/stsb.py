from __future__ import annotations

from typing import Any
import logging

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor


__all__ = ["StsbExtractor"]
log = logging.getLogger(__name__)

task_names = ("stsb",)

class StsbExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Stsb benchmark."""

    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        

        max_items = self._normalize_limit(limit)
        docs = self.load_dataset(
            dataset_name="sentence-transformers/stsb",
            split="test",
            limit=max_items,
        )
        
        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid stsb pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        try:
            sentence1 = doc.get("sentence1", "").strip()
            sentence2 = doc.get("sentence2", "").strip()
            score = doc.get("score")

            if not sentence1 or not sentence2 or score is None:
                log.debug("Skipping: missing sentence1 or sentence2 or score")
                return None

            correct = str(score)
            incorrect = str(1 - score) if score != 0.5 else str(0.1)

            formatted_question = (
                f"Rate the semantic similarity between the following two sentences "
                f"on a scale from 0 to 1, where 0 means completely different and 1 means identical in meaning.\n\n"
                f"Sentence 1: {sentence1}\n"
                f"Sentence 2: {sentence2}"
            )

            metadata = {"label": "stsb"}

            return self._build_pair(
                question=formatted_question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(prompt=question, positive_response=positive_response, negative_response=negative_response, label=metadata.get("label"))
