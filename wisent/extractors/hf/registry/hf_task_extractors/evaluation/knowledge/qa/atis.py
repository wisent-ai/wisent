from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind
from wisent.core.utils.config_tools.constants import EVAL_SINGLE_CHAR_LENGTH

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
        lm_eval_task_data: ConfigurableTask | None = None,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="atis")
        max_items = self._normalize_limit(limit)

        if lm_eval_task_data is not None:
            docs = self.load_docs(
                lm_eval_task_data, max_items,
                preferred_doc=preferred_doc)
        else:
            docs = self._load_atis_from_hf(max_items)

        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid pairs extracted", extra={"task": "atis"})

        return pairs

    def _load_atis_from_hf(self, max_items):
        """Load ATIS dataset directly from HuggingFace across all splits."""
        docs = self.load_all_splits(dataset_name="tuetschek/atis")
        if max_items:
            docs = docs[:max_items]
        return docs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Extract contrastive pair from ATIS doc.

        Supports two schemas:
        - lm-eval: {'source': str, 'target': str}
        - tuetschek/atis HF: {'text': str, 'intent': str, 'slots': str}
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            prompt = (
                doc.get("source", "")
                or doc.get("text", "")
            ).strip()
            correct = (
                doc.get("target", "")
                or doc.get("slots", "")
            ).strip()

            if not prompt or not correct:
                log.debug("Skipping: missing fields")
                return None

            incorrect = self._corrupt_target(correct)
            metadata = {"label": "atis"}

            return self._build_pair(
                question=prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair", exc_info=exc)
            return None

    @staticmethod
    def _corrupt_target(correct: str) -> str:
        """Generate incorrect response by reversing parts."""
        parts = correct.split(", ")
        if len(parts) > EVAL_SINGLE_CHAR_LENGTH:
            incorrect = ", ".join(reversed(parts))
        else:
            incorrect = "none"
        if incorrect == correct:
            incorrect = "none"
        return incorrect

