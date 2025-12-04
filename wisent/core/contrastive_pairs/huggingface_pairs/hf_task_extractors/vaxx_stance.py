from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["VaxxStanceExtractor"]
_LOG = setup_logger(__name__)

task_names = ("vaxx_stance",)

class VaxxStanceExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Vaxx Stance - vaccination stance classification."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="vaxx_stance")
        max_items = self._normalize_limit(limit)

        from datasets import load_dataset
        try:
            dataset = load_dataset("tweet_eval", "stance_climate", split="test")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load vaxx_stance dataset: {e}")
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
            log.warning("No valid pairs extracted", extra={"task": "vaxx_stance"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            text = doc.get("text", "").strip()
            label = doc.get("label")

            if not text or label is None:
                log.debug("Skipping doc due to missing text or label", extra={"doc": doc})
                return None

            # Stance labels: 0=none, 1=against, 2=favor
            label_map = {0: "none", 1: "against", 2: "favor"}

            if label not in label_map:
                log.debug("Invalid label", extra={"doc": doc})
                return None

            correct = label_map[label]
            incorrect_idx = (label + 1) % 3
            incorrect = label_map[incorrect_idx]

            prompt = f"What is the stance of this tweet?\n\n{text}\n\nA. {incorrect}\nB. {correct}"
            metadata = {"label": "vaxx_stance"}

            return self._build_pair(
                question=prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

