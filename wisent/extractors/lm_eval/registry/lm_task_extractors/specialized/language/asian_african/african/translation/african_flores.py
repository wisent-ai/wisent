from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["AfricanFloresExtractor"]
_LOG = setup_logger(__name__)


class AfricanFloresExtractor(LMEvalBenchmarkExtractor):
    """Extractor for african_flores and african_flores_tasks group benchmarks.

    These are lm-eval group tasks whose leaf subtasks expose flores-format
    documents with ``sentence_{lang}_{script}`` fields.  Contrastive pairs are
    built by treating the first sentence field as the source and the second as
    the correct translation; the negative is a word-shuffled version of the
    target sentence.
    """

    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: "ConfigurableTask",
        limit: int | None = None,
        preferred_doc: str | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))
        max_items = self._normalize_limit(limit)

        docs = self.load_docs(
            lm_eval_task_data,
            max_items,
            preferred_doc=preferred_doc,
            train_ratio=train_ratio,
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
            log.warning("No valid african_flores pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single flores doc into a ContrastivePair.

        Flores lm-eval docs expose sentence fields in one of two forms:

        1. ``sentence_{lang}_{script}``  (e.g. ``sentence_afr_Latn``)
        2. A nested ``translation`` dict  (e.g. ``{"en": "...", "fr": "..."}``)

        For form 1 the first two ``sentence_*`` keys are used as source /
        target.  For form 2 the same logic as TranslationExtractor applies.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # --- Form 1: sentence_* fields ---
            sentence_fields = [k for k in doc if k.startswith("sentence_")]
            if len(sentence_fields) >= 2:
                source_field = sentence_fields[0]
                target_field = sentence_fields[1]

                source_text = str(doc.get(source_field, "")).strip()
                target_text = str(doc.get(target_field, "")).strip()

                if not source_text or not target_text:
                    log.debug("Skipping doc: empty sentence fields", extra={"doc": doc})
                    return None

                source_lang = source_field.replace("sentence_", "")
                target_lang = target_field.replace("sentence_", "")
                prompt = (
                    f"Translate the following from {source_lang} to {target_lang}:\n"
                    f"{source_text}"
                )
                return self._make_pair(prompt, target_text)

            # --- Form 2: translation dict ---
            if "translation" in doc and isinstance(doc["translation"], dict):
                translation = doc["translation"]
                lang_keys = list(translation.keys())
                if len(lang_keys) < 2:
                    return None
                source_lang, target_lang = lang_keys[0], lang_keys[1]
                source_text = str(translation[source_lang]).strip()
                target_text = str(translation[target_lang]).strip()
                if not source_text or not target_text:
                    return None
                prompt = (
                    f"Translate from {source_lang} to {target_lang}:\n"
                    f"{source_text}"
                )
                return self._make_pair(prompt, target_text)

            log.debug("Skipping doc: unrecognised format", extra={"doc": doc})
            return None

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

    @staticmethod
    def _make_pair(prompt: str, target_text: str) -> ContrastivePair:
        """Build a ContrastivePair from a prompt and the correct target sentence."""
        words = target_text.split()
        if len(words) < 2:
            incorrect = "[incorrect translation]"
        else:
            shuffled = words.copy()
            random.shuffle(shuffled)
            for _ in range(5):
                if shuffled != words:
                    break
                random.shuffle(shuffled)
            incorrect = " ".join(shuffled)

        return ContrastivePair(
            prompt=prompt,
            positive_response=PositiveResponse(model_response=target_text),
            negative_response=NegativeResponse(model_response=incorrect),
            label="african_flores",
        )
