from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MtsDialogExtractor"]
_LOG = setup_logger(__name__)

task_names = ("mts_dialog",)

class MtsDialogExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the MTS Dialog benchmark (medical dialog summarization)."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Mts Dialog docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Mts Dialog.
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
            log.warning("No valid Mts Dialog pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single MTS Dialog doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        MTS Dialog format:
        - dialogue: medical dialog between doctor and patient
        - section_text: clinical note summary (target)
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # MTS Dialog format
            if "dialogue" in doc and "section_text" in doc:
                dialogue = str(doc.get("dialogue", "")).strip()
                section_text = str(doc.get("section_text", "")).strip()

                if not dialogue or not section_text:
                    log.debug("Skipping doc with missing dialogue/section_text", extra={"doc": doc})
                    return None

                # Prompt is just the dialogue (as in lm-eval)
                prompt = dialogue

                # Positive: the actual clinical note summary
                correct = section_text

                # Negative: generic non-summary response
                incorrect = "I cannot provide a summary of this medical dialog."

                metadata = {"label": "mts_dialog"}

                return self._build_pair(
                    question=prompt,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            log.debug("Skipping doc without dialogue/section_text fields", extra={"doc": doc})
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
