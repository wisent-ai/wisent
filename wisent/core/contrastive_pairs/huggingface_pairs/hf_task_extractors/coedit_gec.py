from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["CoeditGecExtractor"]
_LOG = setup_logger(__name__)

task_names = ("coedit_gec",)

class CoeditGecExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for CoEdit GEC - grammar error correction."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="coedit_gec")
        max_items = self._normalize_limit(limit)

        from datasets import load_dataset
        try:
            dataset = load_dataset("grammarly/coedit", split="test")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load coedit_gec dataset: {e}")
            return []

        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(dataset)})

        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid pairs extracted", extra={"task": "coedit_gec"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            src = doc.get("src", doc.get("source", "")).strip()
            tgt = doc.get("tgt", doc.get("target", "")).strip()

            if not src or not tgt:
                log.debug("Skipping doc due to missing src or tgt", extra={"doc": doc})
                return None

            # src is text with errors, tgt is corrected text
            correct = tgt
            incorrect = src

            question = f"Correct the grammar in the following text:\n\n{src}"
            metadata = {"label": "coedit_gec"}

            return self._build_pair(
                question=question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

