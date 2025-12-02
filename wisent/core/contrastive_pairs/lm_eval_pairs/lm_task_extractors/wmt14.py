from __future__ import annotations

import random
import logging
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["Wmt14Extractor"]
log = logging.getLogger(__name__)

# Group extractor for wmt14 translation tasks
task_names = ("wmt14-en-fr", "wmt14-fr-en")


class Wmt14Extractor(LMEvalBenchmarkExtractor):
    """Extractor for WMT14 English-French translation benchmark."""

    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        # Determine source/target language from task name
        task_name = getattr(lm_eval_task_data, "NAME", "wmt14-en-fr")
        if "fr-en" in task_name:
            source_lang, target_lang = "fr", "en"
        else:
            source_lang, target_lang = "en", "fr"

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc)
        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc, source_lang, target_lang)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any], source_lang: str, target_lang: str) -> ContrastivePair | None:
        try:
            translation = doc.get("translation", {})
            source_text = translation.get(source_lang, "").strip()
            target_text = translation.get(target_lang, "").strip()

            if not source_text or not target_text:
                log.debug("Skipping doc due to missing source or target text")
                return None

            correct = target_text

            # Create incorrect by shuffling words in target
            words = target_text.split()
            if len(words) > 2:
                shuffled = words.copy()
                random.shuffle(shuffled)
                incorrect = " ".join(shuffled)
            else:
                # For short sentences, reverse the words
                incorrect = " ".join(reversed(words))

            lang_names = {"en": "English", "fr": "French"}
            source_name = lang_names.get(source_lang, source_lang)
            target_name = lang_names.get(target_lang, target_lang)

            formatted_question = (
                f"Translate the following {source_name} text to {target_name}.\n\n"
                f"{source_name}: {source_text}\n"
                f"{target_name}:"
            )
            metadata = {"label": "wmt14"}

            return self._build_pair(
                question=formatted_question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(prompt=question, positive_response=positive_response, negative_response=negative_response, label=metadata.get("label"))
