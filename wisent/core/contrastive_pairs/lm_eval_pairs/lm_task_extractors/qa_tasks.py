from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["BoolQExtractor", "RecordExtractor"]
_LOG = setup_logger(__name__)


class BoolQExtractor(LMEvalBenchmarkExtractor):
    """Extractor for BoolQ benchmark (binary yes/no questions)."""

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from BoolQ."""
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items)

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid BoolQ pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a BoolQ doc into a ContrastivePair."""
        log = bind(_LOG, doc_id=doc.get("idx", "unknown"))

        try:
            question = str(doc.get("question", "")).strip()
            passage = str(doc.get("passage", "")).strip()
            label = doc.get("label")

            if not question or label is None:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            # label: 1 = True/Yes, 0 = False/No
            correct = "Yes" if label == 1 else "No"
            incorrect = "No" if label == 1 else "Yes"

            # Format with passage and question
            formatted_question = f"Passage: {passage}\n\nQuestion: {question}\nAnswer with Yes or No:"

            metadata = {"label": "boolq"}

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
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None
        )


class RecordExtractor(LMEvalBenchmarkExtractor):
    """Extractor for ReCoRD benchmark (reading comprehension)."""

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from ReCoRD."""
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items)

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            extracted_pairs = self._extract_pairs_from_doc(doc)
            for pair in extracted_pairs:
                if pair is not None:
                    pairs.append(pair)
                    if max_items is not None and len(pairs) >= max_items:
                        break
            if max_items is not None and len(pairs) >= max_items:
                break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid ReCoRD pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pairs_from_doc(self, doc: dict[str, Any]) -> list[ContrastivePair]:
        """
        Convert a ReCoRD doc into ContrastivePairs.
        ReCoRD can have multiple questions per passage.
        """
        log = bind(_LOG, doc_id=doc.get("idx", "unknown"))
        pairs = []

        try:
            passage = str(doc.get("passage", "")).strip()
            query = str(doc.get("query", "")).strip()
            entities = doc.get("entities", [])
            answers = doc.get("answers", [])

            if not passage or not query or not entities or not answers:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return pairs

            # Get correct and incorrect entities
            correct_entities = [str(ans).strip() for ans in answers if ans]
            incorrect_entities = [str(ent).strip() for ent in entities if str(ent).strip() not in correct_entities]

            if not correct_entities or not incorrect_entities:
                log.debug("Skipping doc: no correct or incorrect entities", extra={"doc": doc})
                return pairs

            # Create pairs using first correct and first incorrect
            correct = correct_entities[0]
            incorrect = incorrect_entities[0]

            formatted_question = f"Passage: {passage}\n\nQuestion: {query}\nAnswer:"

            metadata = {"label": "record"}

            pair = self._build_pair(
                question=formatted_question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )
            pairs.append(pair)

        except Exception as exc:
            log.error("Error extracting pairs from doc", exc_info=exc, extra={"doc": doc})

        return pairs

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None
        )
