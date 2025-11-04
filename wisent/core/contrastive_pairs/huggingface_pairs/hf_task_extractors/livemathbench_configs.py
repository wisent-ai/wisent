"""Configuration-specific LiveMathBench extractors."""
from __future__ import annotations

from typing import Any
import logging

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.livemathbench import LiveMathBenchExtractor

__all__ = [
    "LiveMathBenchCnmoEnExtractor",
    "LiveMathBenchCnmoZhExtractor",
]

log = logging.getLogger(__name__)


class LiveMathBenchCnmoEnExtractor(LiveMathBenchExtractor):
    """Extractor for LiveMathBench CNMO English."""

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        max_items = self._normalize_limit(limit)
        docs = self.load_dataset(
            dataset_name="opencompass/LiveMathBench",
            dataset_config="v202412_CNMO_en",
            split="test",
            limit=max_items,
        )
        pairs: list[ContrastivePair] = []
        log.info(f"Extracting contrastive pairs from {len(docs)} LiveMathBench (cnmo_en) examples")
        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break
        if not pairs:
            log.warning("No valid LiveMathBench (cnmo_en) pairs extracted")
        return pairs


class LiveMathBenchCnmoZhExtractor(LiveMathBenchExtractor):
    """Extractor for LiveMathBench CNMO Chinese."""

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        max_items = self._normalize_limit(limit)
        docs = self.load_dataset(
            dataset_name="opencompass/LiveMathBench",
            dataset_config="v202412_CNMO_cn",
            split="test",
            limit=max_items,
        )
        pairs: list[ContrastivePair] = []
        log.info(f"Extracting contrastive pairs from {len(docs)} LiveMathBench (cnmo_zh) examples")
        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break
        if not pairs:
            log.warning("No valid LiveMathBench (cnmo_zh) pairs extracted")
        return pairs
