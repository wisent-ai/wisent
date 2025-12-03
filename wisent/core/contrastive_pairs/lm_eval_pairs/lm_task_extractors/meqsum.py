from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MeqsumExtractor"]
_LOG = setup_logger(__name__)

task_names = ("meqsum",)

class MeqsumExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Meqsum benchmark."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Meqsum docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Meqsum.
            limit: Optional maximum number of pairs to produce.
            preferred_doc: Optional preferred document source.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc)

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
            log.warning("No valid Meqsum pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Meqsum doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        Meqsum format: CHQ (Consumer Health Question) -> Summary
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # meqsum format: CHQ (consumer health question) + Summary
            if "CHQ" in doc and "Summary" in doc:
                chq = str(doc.get("CHQ", "")).strip()
                summary = str(doc.get("Summary", "")).strip()

                if not chq or not summary:
                    log.debug("Skipping doc with empty CHQ or Summary", extra={"doc": doc})
                    return None

                # Extract the MESSAGE portion from CHQ
                text = chq
                idx = text.find("MESSAGE")
                if idx != -1:
                    text = text[idx + 9:]  # Skip "MESSAGE: " (9 chars)

                # Create synthetic negative - generic non-answer for summarization
                incorrect = "Unable to provide a summary. The question is unclear."

                metadata = {"label": "meqsum"}
                return self._build_pair(
                    question=f"Extract and summarize the following medical question:\n{text}",
                    correct=summary,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            log.debug("Skipping doc without CHQ/Summary fields", extra={"doc": doc})
            return None

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
