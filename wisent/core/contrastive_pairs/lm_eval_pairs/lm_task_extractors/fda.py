from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["FdaExtractor"]
_LOG = setup_logger(__name__)

task_names = ("fda",)

class FdaExtractor(LMEvalBenchmarkExtractor):
    """Extractor for FDA benchmark - extracting key-value pairs from FDA documents.

    This is an information extraction task where models extract values for given keys from documents.
    Format: text (document chunk) + key (field to extract) → value (extracted text)
    """


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
        log = bind(_LOG, doc_id=doc.get("doc_id", "unknown"))

        try:
            # FDA format: text (document chunk) + key (field name) → value (extracted text)
            text = doc.get("text", "").strip()
            key = doc.get("key", "").strip()
            value = doc.get("value", "").strip()

            if not text or not key or not value:
                log.debug("Skipping doc due to missing text, key, or value", extra={"doc": doc})
                return None

            # Create prompt: {text} \n {key}:
            prompt = f"{text}\n{key}:"

            # Positive: correct value
            correct_value = value

            # Negative: shuffled version of the value to create synthetic incorrect extraction
            words = value.split()
            if len(words) < 2:
                # If value is too short, use a generic wrong answer
                incorrect_value = "Not found"
            else:
                shuffled_words = words.copy()
                random.shuffle(shuffled_words)
                incorrect_value = ' '.join(shuffled_words)

            metadata = {"label": "fda"}

            return self._build_pair(
                question=prompt,
                correct=correct_value,
                incorrect=incorrect_value,
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
