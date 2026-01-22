"""SQuAD v2 and DROP benchmark extractors."""
from __future__ import annotations
from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

log = setup_logger(__name__)

__all__ = ["SQuADv2Extractor", "DROPExtractor"]


class SQuADv2Extractor(HuggingFaceBenchmarkExtractor):
    """Extract contrastive pairs from SQuAD v2 benchmark."""

    def __init__(self):
        super().__init__()
        self.name = "squadv2"

    def extract_contrastive_pairs(self, limit: int | None = 500) -> list[ContrastivePair]:
        pairs: list[ContrastivePair] = []

        try:
            docs = self.load_dataset(
                dataset_name="rajpurkar/squad_v2",
                split="validation",
                limit=limit,
            )
            log.info(f"Loaded {len(docs)} examples from SQuAD v2")
        except Exception as e:
            log.error(f"Failed to load SQuAD v2: {e}")
            return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair:
                pairs.append(pair)

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            context = doc.get("context", "").strip()
            question = doc.get("question", "").strip()
            answers = doc.get("answers", {})
            
            answer_texts = answers.get("text", [])
            if not answer_texts:
                # Unanswerable question
                correct = "The question cannot be answered based on the given context."
                incorrect = "Based on the context, the answer is [incorrect guess]."
            else:
                correct = answer_texts[0]
                incorrect = "I don't know the answer."

            if not context or not question:
                return None

            task_prompt = f"""Read the following context and answer the question.

Context: {context[:1500]}

Question: {question}

Answer:"""

            positive_response = PositiveResponse(model_response=correct)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt=task_prompt,
                positive_response=positive_response,
                negative_response=negative_response,
                label="squadv2",
                metadata={"source": "rajpurkar/squad_v2"},
            )

        except Exception as e:
            log.debug(f"Failed to extract pair: {e}")
            return None


class DROPExtractor(HuggingFaceBenchmarkExtractor):
    """Extract contrastive pairs from DROP benchmark."""

    def __init__(self):
        super().__init__()
        self.name = "drop"

    def extract_contrastive_pairs(self, limit: int | None = 500) -> list[ContrastivePair]:
        pairs: list[ContrastivePair] = []

        try:
            docs = self.load_dataset(
                dataset_name="ucinlp/drop",
                split="validation",
                limit=limit,
            )
            log.info(f"Loaded {len(docs)} examples from DROP")
        except Exception as e:
            log.error(f"Failed to load DROP: {e}")
            return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair:
                pairs.append(pair)

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            passage = doc.get("passage", "").strip()
            question = doc.get("question", "").strip()
            answers = doc.get("answers_spans", {})
            
            answer_spans = answers.get("spans", [])
            if answer_spans:
                correct = answer_spans[0]
            else:
                return None

            if not passage or not question:
                return None

            task_prompt = f"""Read the passage and answer the question (may require reasoning or calculation).

Passage: {passage[:1500]}

Question: {question}

Answer:"""

            incorrect = "I cannot determine the answer."

            positive_response = PositiveResponse(model_response=correct)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt=task_prompt,
                positive_response=positive_response,
                negative_response=negative_response,
                label="drop",
                metadata={"source": "ucinlp/drop"},
            )

        except Exception as e:
            log.debug(f"Failed to extract pair: {e}")
            return None
