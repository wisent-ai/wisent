from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["EpecKorefBinExtractor"]
_LOG = setup_logger(__name__)

task_names = ("epec_koref_bin",)

class EpecKorefBinExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for EPEC Koref Bin - coreference resolution binary classification."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="epec_koref_bin")
        max_items = self._normalize_limit(limit)

        from datasets import load_dataset
        try:
            dataset = load_dataset("classla/EPEC-coref", split="test")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load epec_koref_bin dataset: {e}")
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
            log.warning("No valid pairs extracted", extra={"task": "epec_koref_bin"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            text = doc.get("text", "").strip()
            span1 = doc.get("span1_text", doc.get("span1", "")).strip()
            span2 = doc.get("span2_text", doc.get("span2", "")).strip()
            label = doc.get("label")

            if not text or not span1 or not span2 or label is None:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            prompt = f"Text: {text}\n\nDo '{span1}' and '{span2}' refer to the same entity?\nAnswer:"

            if label == 1:
                correct = "Yes"
                incorrect = "No"
            else:
                correct = "No"
                incorrect = "Yes"

            metadata = {"label": "epec_koref_bin"}

            return self._build_pair(
                question=prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

