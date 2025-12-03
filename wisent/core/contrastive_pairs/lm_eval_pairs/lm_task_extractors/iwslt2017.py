from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["IWSLT2017Extractor"]
_LOG = setup_logger(__name__)

task_names = (
    "iwslt2017-ar-en",
    "iwslt2017-en-ar",
)

class IWSLT2017Extractor(LMEvalBenchmarkExtractor):
    """Extractor for IWSLT 2017 benchmark - machine translation tasks."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))
        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc)
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
            # IWSLT2017 format: {'translation': {'ar': '...', 'en': '...'}}
            if "translation" in doc:
                translation = doc.get("translation", {})

                # Get source and target language keys
                lang_keys = list(translation.keys())
                if len(lang_keys) < 2:
                    log.debug("Skipping doc due to insufficient languages", extra={"doc": doc})
                    return None

                source_lang = lang_keys[0]
                target_lang = lang_keys[1]

                source_text = translation.get(source_lang, "").strip()
                target_text = translation.get(target_lang, "").strip()

                if not source_text or not target_text:
                    log.debug("Skipping doc due to empty text", extra={"doc": doc})
                    return None

                # Create translation prompt
                prompt = f"Translate the following from {source_lang} to {target_lang}:\n{source_text}"

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

                metadata = {"label": "iwslt2017"}

                return self._build_pair(
                    question=prompt,
                    correct=correct_translation,
                    incorrect=incorrect_translation,
                    metadata=metadata,
                )

            log.debug("Skipping doc due to unrecognized format", extra={"doc": doc})
            return None

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
