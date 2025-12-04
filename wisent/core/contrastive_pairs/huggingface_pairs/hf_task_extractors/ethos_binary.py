from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["EthosBinaryExtractor"]
_LOG = setup_logger(__name__)

task_names = ("ethos_binary",)

class EthosBinaryExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Ethos Binary - hate speech classification."""


    evaluator_name = "exact_match"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="ethos_binary")
        max_items = self._normalize_limit(limit)

        from datasets import load_dataset
        try:
            dataset = load_dataset("ethos", "binary", split="train")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load ethos_binary dataset: {e}")
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
            log.warning("No valid pairs extracted", extra={"task": "ethos_binary"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            text = doc.get("text", "").strip()
            label = doc.get("label")

            if not text or label is None:
                log.debug("Skipping doc due to missing text or label", extra={"doc": doc})
                return None

            if label == 1:
                correct = "hate speech"
                incorrect = "not hate speech"
            else:
                correct = "not hate speech"
                incorrect = "hate speech"

            question = f"Is the following text hate speech?\n\nText: {text}\nAnswer:"
            metadata = {"label": "ethos_binary"}

            return self._build_pair(
                question=question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

