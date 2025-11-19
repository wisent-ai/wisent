from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["Wmt16EnDeExtractor"]
_LOG = setup_logger(__name__)

task_names = ("wmt16-en-de",)

evaluator_name = "generation"


class Wmt16EnDeExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Wmt16 En De benchmark."""

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
            # WMT16 translation format: translation dict with 'en' (source) and 'de' (target)
            translation = doc.get("translation", {})
            source_text = translation.get("en", "").strip()
            target_text = translation.get("de", "").strip()

            if not source_text or not target_text:
                log.debug("Skipping doc due to missing source or target", extra={"doc": doc})
                return None

            # Use English text as prompt and German translation as correct response
            # For incorrect response, use a truncated/corrupted version of the German translation
            words = target_text.split()
            if len(words) > 3:
                # Use first half as truncated translation
                incorrect = " ".join(words[:len(words)//2])
            else:
                # If target is very short, use a single word or empty
                incorrect = words[0] if words else "No translation"

            # Create prompt with instruction to translate from English to German
            prompt = f"Translate from English to German:\n{source_text}"

            metadata = {"label": "wmt16_en_de"}

            return self._build_pair(
                question=prompt,
                correct=target_text,
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
