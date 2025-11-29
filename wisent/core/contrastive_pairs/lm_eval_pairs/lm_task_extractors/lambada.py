from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["LambadaExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "lambada",
    "lambada_cloze",
    "lambada_multilingual",
)

class LambadaExtractor(LMEvalBenchmarkExtractor):
    """Extractor for LAMBADA benchmarks (word prediction).

    Works for: lambada_openai, lambada_standard
    """


    evaluator_name = "exact_match"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from LAMBADA task docs.

        Schema:
            - text: str (passage ending with a word to predict)

        Args:
            lm_eval_task_data: lm-eval task instance.
            limit: Optional maximum number of pairs to produce.
            preferred_doc: Preferred document source ("validation", "test", "training", "fewshot")

        Returns:
            A list of ContrastivePair objects.
        """
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
        """
        Convert a single LAMBADA doc into a ContrastivePair.

        Strategy:
        1. Take the text and extract the last word as the target
        2. Positive: correct last word
        3. Negative: random incorrect word
        """
        log = bind(_LOG)

        try:
            # Get the text
            text = str(doc.get("text", "")).strip()

            if not text:
                log.debug("Skipping doc due to missing text", extra={"doc": doc})
                return None

            # Split into words
            words = text.split()

            if len(words) < 5:
                log.debug("Not enough words to create pair", extra={"doc": doc})
                return None

            # Take last word as target, rest as context
            context_words = words[:-1]
            target_word = words[-1]

            context = ' '.join(context_words)

            # Create a random incorrect word (not the target)
            # Use simple common words as alternatives
            common_words = ["the", "and", "but", "or", "not", "very", "some", "all", "any", "more",
                           "said", "went", "came", "took", "made", "found", "gave", "told", "asked"]
            incorrect_words = [w for w in common_words if w.lower() != target_word.lower()]

            if incorrect_words:
                incorrect_word = random.choice(incorrect_words)
            else:
                # Fallback: shuffle letters of target word
                incorrect_word = ''.join(random.sample(target_word, len(target_word)))

            metadata = {
                "label": "lambada",
            }

            return self._build_pair(
                question=context,
                correct=target_word,
                incorrect=incorrect_word,
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
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None
        )
