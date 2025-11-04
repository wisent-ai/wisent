"""Configuration-specific PolyMath extractors."""
from __future__ import annotations

from typing import Any
import logging

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.polymath import PolyMathExtractor

__all__ = [
    "PolyMathEnMediumExtractor",
    "PolyMathZhMediumExtractor",
    "PolyMathEnHighExtractor",
    "PolyMathZhHighExtractor",
]

log = logging.getLogger(__name__)


class PolyMathEnMediumExtractor(PolyMathExtractor):
    """Extractor for PolyMath English Medium difficulty."""

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        max_items = self._normalize_limit(limit)
        docs = self.load_dataset(
            dataset_name="Qwen/PolyMath",
            dataset_config="en",
            split="medium",
            limit=max_items,
        )
        pairs: list[ContrastivePair] = []
        log.info(f"Extracting contrastive pairs from {len(docs)} PolyMath (en_medium) examples")
        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break
        if not pairs:
            log.warning("No valid PolyMath (en_medium) pairs extracted")
        return pairs


class PolyMathZhMediumExtractor(PolyMathExtractor):
    """Extractor for PolyMath Chinese Medium difficulty."""

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        max_items = self._normalize_limit(limit)
        docs = self.load_dataset(
            dataset_name="Qwen/PolyMath",
            dataset_config="zh",
            split="medium",
            limit=max_items,
        )
        pairs: list[ContrastivePair] = []
        log.info(f"Extracting contrastive pairs from {len(docs)} PolyMath (zh_medium) examples")
        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break
        if not pairs:
            log.warning("No valid PolyMath (zh_medium) pairs extracted")
        return pairs


class PolyMathEnHighExtractor(PolyMathExtractor):
    """Extractor for PolyMath English High difficulty."""

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        max_items = self._normalize_limit(limit)
        docs = self.load_dataset(
            dataset_name="Qwen/PolyMath",
            dataset_config="en",
            split="high",
            limit=max_items,
        )
        pairs: list[ContrastivePair] = []
        log.info(f"Extracting contrastive pairs from {len(docs)} PolyMath (en_high) examples")
        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break
        if not pairs:
            log.warning("No valid PolyMath (en_high) pairs extracted")
        return pairs


class PolyMathZhHighExtractor(PolyMathExtractor):
    """Extractor for PolyMath Chinese High difficulty."""

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        max_items = self._normalize_limit(limit)
        docs = self.load_dataset(
            dataset_name="Qwen/PolyMath",
            dataset_config="zh",
            split="high",
            limit=max_items,
        )
        pairs: list[ContrastivePair] = []
        log.info(f"Extracting contrastive pairs from {len(docs)} PolyMath (zh_high) examples")
        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break
        if not pairs:
            log.warning("No valid PolyMath (zh_high) pairs extracted")
        return pairs
