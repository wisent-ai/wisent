from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["AtisExtractor"]
_LOG = setup_logger(__name__)

task_names = ("atis",)

class AtisExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Atis benchmark - NER task."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
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
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Extract contrastive pair from ATIS NER doc.
        Schema: {'source': str (prompt), 'target': str (correct extraction), 'task_data': dict}
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            prompt = doc.get("source", "").strip()
            correct = doc.get("target", "").strip()

            if not prompt or not correct:
                log.debug("Skipping doc due to missing/invalid fields", extra={"doc": doc})
                return None

            # Generate an incorrect response by corrupting the target
            # Strategy: reverse the extraction order
            parts = correct.split(", ")
            if len(parts) > 1:
                # Reverse the order of extractions
                incorrect = ", ".join(reversed(parts))
            else:
                # If only one part, add "none" as incorrect
                incorrect = "none"

            # Ensure incorrect is actually different
            if incorrect == correct:
                incorrect = "none"

            metadata = {"label": "atis"}

            return self._build_pair(
                question=prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

