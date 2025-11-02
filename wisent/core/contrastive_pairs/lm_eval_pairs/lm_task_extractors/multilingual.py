from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MultilingualExtractor"]
_LOG = setup_logger(__name__)


class MultilingualExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Multilingual tasks."""

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Build contrastive pairs from docs."""
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items)

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc, lm_eval_task_data)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(
        self, doc: dict[str, Any], task_data: Any = None
    ) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            # Generic fallback - use task_data methods
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = str(doc)

            # Try multiple possible answer fields
            correct_answer = None
            for field in ["answer", "label", "target", "gold", "correct_answer"]:
                if field in doc:
                    correct_answer = doc[field]
                    break

            if not formatted_question or correct_answer is None:
                _LOG.debug("Skipping: missing question or answer")
                return None

            # Generic incorrect answer
            incorrect_answer = "incorrect"

            metadata = {
                "label": "multilingual",
                "source": getattr(task_data, "NAME", "multilingual"),
            }

            return self._build_pair(
                question=formatted_question,
                correct=str(correct_answer),
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            _LOG.error(f"Error extracting pair from doc: {exc}", exc_info=True)
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
