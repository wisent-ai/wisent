from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["WikitextExtractor"]
_LOG = setup_logger(__name__)

task_names = ("wikitext",)

class WikitextExtractor(LMEvalBenchmarkExtractor):
    """Extractor for WikiText and similar perplexity benchmarks.

    Works for: wikitext, wikitext103, ptb, penn_treebank
    """


    evaluator_name = "perplexity"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from perplexity task docs.

        Schema:
            - page: str (long text passage)

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
        Convert a single doc into a ContrastivePair for perplexity tasks.

        Strategy:
        1. Take a text passage and split it into context and continuation
        2. Positive: correct continuation
        3. Negative: shuffled or corrupted continuation
        """
        log = bind(_LOG)

        try:
            # Get the page content
            page = str(doc.get("page", "")).strip()

            if not page or len(page) < 100:
                log.debug("Skipping doc due to short/missing text", extra={"doc": doc})
                return None

            # Split the page into sentences (simple approach)
            sentences = [s.strip() for s in page.split('.') if s.strip()]

            if len(sentences) < 4:
                log.debug("Not enough sentences to create pair", extra={"doc": doc})
                return None

            # Take first 2-3 sentences as context, next 2-3 as continuation
            mid_point = len(sentences) // 2
            context_sentences = sentences[:mid_point]
            continuation_sentences = sentences[mid_point:mid_point+3]

            if not continuation_sentences:
                return None

            context = '. '.join(context_sentences) + '.'
            correct_continuation = ' '.join(continuation_sentences) + '.'

            # Create negative by shuffling words in the continuation
            words = correct_continuation.split()
            shuffled_words = words.copy()
            random.shuffle(shuffled_words)
            incorrect_continuation = ' '.join(shuffled_words)

            # Truncate context if too long
            if len(context) > 500:
                context = context[:500] + "..."

            metadata = {
                "label": "perplexity",
            }

            return self._build_pair(
                question=context,
                correct=correct_continuation,
                incorrect=incorrect_continuation,
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
