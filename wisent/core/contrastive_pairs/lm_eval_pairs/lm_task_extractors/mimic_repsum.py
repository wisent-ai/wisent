from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MimicRepsumExtractor"]
_LOG = setup_logger(__name__)

task_names = ("mimic_repsum",)

class MimicRepsumExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Mimic Repsum benchmark."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Mimic Repsum docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Mimic Repsum.
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
            log.warning("No valid Mimic Repsum pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Mimic Repsum doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        Mimic Repsum format: extractive_notes_summ (FINDINGS + IMPRESSION sections)
        Task: Summarize the findings into diagnostic statements (IMPRESSION)
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            import re

            # mimic_repsum format: extractive_notes_summ with FINDINGS and IMPRESSION
            if "extractive_notes_summ" in doc:
                text = str(doc.get("extractive_notes_summ", "")).strip()

                if not text or len(text) < 5:
                    log.debug("Skipping doc with empty or too short extractive_notes_summ", extra={"doc": doc})
                    return None

                # Extract FINDINGS and IMPRESSION sections (following utils.py logic)
                a = re.search("IMPRESSION", text, re.IGNORECASE)
                if a is not None:
                    a = a.start()
                else:
                    a = -1
                b = re.search("FINDING", text, re.IGNORECASE)
                if b is not None:
                    b = b.start()
                else:
                    b = -1

                if a < b:
                    impressions = text[a:b].split("     ")[0]
                    findings = text[b:].split("     ")[0]
                else:
                    impressions = text[a:].split("     ")[0]
                    findings = text[b:a].split("     ")[0]

                if len(findings) < 5 < len(impressions):
                    findings = text[:a]

                # Skip if findings or impressions are too short
                if len(findings) < 5 or len(impressions) < 5:
                    log.debug("Skipping doc with too short findings or impressions", extra={"doc": doc})
                    return None

                # Create synthetic negative - generic non-answer for summarization
                incorrect = "Unable to summarize the findings. Additional information is required."

                prompt = f"Given the findings: {findings}.\nSummarize the findings."

                metadata = {"label": "mimic_repsum"}
                return self._build_pair(
                    question=prompt,
                    correct=impressions,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            log.debug("Skipping doc without extractive_notes_summ field", extra={"doc": doc})
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
