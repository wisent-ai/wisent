from __future__ import annotations

import random
from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["Wmt14FrEnExtractor"]
_LOG = setup_logger(__name__)

task_names = ("wmt14-fr-en",)


class Wmt14FrEnExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for WMT14 French-to-English translation task."""

    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="wmt14-fr-en")
        max_items = self._normalize_limit(limit)

        # Load dataset from HuggingFace
        docs = self.load_dataset(
            dataset_name="wmt14",
            dataset_config="fr-en",
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # WMT14 format: {'translation': {'fr': '...', 'en': '...'}}
            if "translation" not in doc:
                log.debug("Skipping doc due to missing translation field", extra={"doc": doc})
                return None

            translation = doc.get("translation", {})
            source_text = translation.get("fr", "").strip()
            target_text = translation.get("en", "").strip()

            if not source_text or not target_text:
                log.debug("Skipping doc due to empty text", extra={"doc": doc})
                return None

            # Create translation prompt
            prompt = f"Translate the following from French to English:\n{source_text}"

            # Positive: correct translation
            correct_translation = target_text

            # Negative: shuffled words for synthetic incorrect translation
            words = target_text.split()
            if len(words) < 2:
                incorrect_translation = "[incorrect translation]"
            else:
                shuffled_words = words.copy()
                random.shuffle(shuffled_words)
                incorrect_translation = ' '.join(shuffled_words)

            metadata = {"label": "wmt14_fr_en"}

            return self._build_pair(
                question=prompt,
                correct=correct_translation,
                incorrect=incorrect_translation,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

