from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["UnfairTosExtractor"]
_LOG = setup_logger(__name__)

task_names = ("unfair_tos",)

class UnfairTosExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Unfair ToS - unfair terms of service classification."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="unfair_tos")
        max_items = self._normalize_limit(limit)

        from datasets import load_dataset
        try:
            dataset = load_dataset("lex_glue", "unfair_tos", split="test")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load unfair_tos dataset: {e}")
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
            log.warning("No valid pairs extracted", extra={"task": "unfair_tos"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            text = doc.get("text", "").strip()
            labels = doc.get("labels", [])

            if not text or not labels:
                log.debug("Skipping doc due to missing text or labels", extra={"doc": doc})
                return None

            # Unfair ToS labels: binary classification for whether clause is unfair
            if 1 in labels or any(l > 0 for l in labels):
                correct = "unfair"
                incorrect = "fair"
            else:
                correct = "fair"
                incorrect = "unfair"

            prompt = f"Is the following clause in terms of service fair or unfair?\n\n{text}\n\nAnswer:"
            metadata = {"label": "unfair_tos"}

            return self._build_pair(
                question=prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

