from __future__ import annotations

import random
from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["EvalitaSpSumTaskFpSmallP1Extractor"]
_LOG = setup_logger(__name__)

task_names = ("evalita-sp_sum_task_fp-small_p1",)

class EvalitaSpSumTaskFpSmallP1Extractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Evalita Sp Sum Task FP Small P1 - Italian summarization."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="evalita-sp_sum_task_fp-small_p1")
        max_items = self._normalize_limit(limit)

        from datasets import load_dataset
        try:
            dataset = load_dataset("RiTA-nlp/EVALITA_2023_EST_FSP", "small", split="train")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load evalita-sp_sum_task_fp-small_p1 dataset: {e}")
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
            log.warning("No valid pairs extracted", extra={"task": "evalita-sp_sum_task_fp-small_p1"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            source = doc.get("source", doc.get("text", "")).strip()
            target = doc.get("target", doc.get("summary", "")).strip()

            if not source or not target:
                log.debug("Skipping doc due to missing source or target", extra={"doc": doc})
                return None

            prompt = f"Summarize the following text:\n\n{source}"

            correct = target

            # Create negative by shuffling words
            words = target.split()
            if len(words) < 5:
                log.debug("Summary too short to shuffle", extra={"doc": doc})
                return None

            shuffled_words = words.copy()
            random.shuffle(shuffled_words)
            incorrect = ' '.join(shuffled_words)

            metadata = {"label": "evalita-sp_sum_task_fp-small_p1"}

            return self._build_pair(
                question=prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

