from __future__ import annotations

import random
from typing import Any, Optional
from wisent.core.utils.cli.cli_logger import setup_logger

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

log = setup_logger(__name__)

class WMT16Extractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for WMT16 translation benchmark.

    Dataset: wmt16 on HuggingFace

    WMT16 (Workshop on Machine Translation 2016) contains parallel corpora
    for various language pairs including English-German, English-Romanian, etc.
    """

    evaluator_name = "generation"

    def __init__(self, lang_pair: Optional[str] = None):
        """
        Initialize WMT16 extractor.

        Args:
            lang_pair: Language pair (e.g., 'de-en', 'en-de', 'ro-en', 'en-ro')
        """
        super().__init__()
        self.lang_pair = lang_pair if lang_pair is not None else "de-en"
        parts = self.lang_pair.split("-")
        self.source_lang = parts[0] if len(parts) > 0 else "de"
        self.target_lang = parts[1] if len(parts) > 1 else "en"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from WMT16 dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        try:
            config = f"{self.source_lang}-{self.target_lang}"
            docs = self.load_dataset(
                dataset_name="wmt16",
                dataset_config=config,
                split="test",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from WMT16 ({config})")
        except Exception as e:
            try:
                config = f"{self.target_lang}-{self.source_lang}"
                docs = self.load_dataset(
                    dataset_name="wmt16",
                    dataset_config=config,
                    split="test",
                    limit=max_items,
                )
                log.info(f"Loaded {len(docs)} examples from WMT16 ({config})")
            except Exception as e2:
                log.error(f"Failed to load WMT16: {e2}")
                return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            translation = doc.get("translation", {})
            source_text = translation.get(self.source_lang, "").strip()
            target_text = translation.get(self.target_lang, "").strip()

            if not source_text or not target_text:
                return None

            lang_names = {
                "en": "English",
                "de": "German",
                "ro": "Romanian",
                "cs": "Czech",
                "fi": "Finnish",
                "ru": "Russian",
                "tr": "Turkish",
            }

            source_name = lang_names.get(self.source_lang, self.source_lang.upper())
            target_name = lang_names.get(self.target_lang, self.target_lang.upper())

            prompt = f"Translate the following from {source_name} to {target_name}:\n{source_text}"

            correct_translation = target_text

            words = target_text.split()
            if len(words) < 2:
                incorrect_translation = "[incorrect translation]"
            else:
                shuffled_words = words.copy()
                random.shuffle(shuffled_words)
                incorrect_translation = " ".join(shuffled_words)

            metadata = {
                "label": f"wmt16_{self.source_lang}_{self.target_lang}",
                "source": "wmt16",
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
            }

            return self._build_pair(
                question=prompt,
                correct=correct_translation,
                incorrect=incorrect_translation,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting WMT16 pair: {exc}", exc_info=True)
            return None
