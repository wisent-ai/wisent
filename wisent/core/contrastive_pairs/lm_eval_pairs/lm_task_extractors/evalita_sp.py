from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["EvalitaSpExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "evalita-sp_sum_task_fp-small_p1",
    "evalita-sp_sum_task_fp-small_p2",
    "evalita-sp_sum_task_fp_p1",
    "evalita-sp_sum_task_fp_p2",
)

class EvalitaSpExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Evalita Sp benchmark - Italian summarization task.

    This is a summarization task where models must generate summaries.
    Format: source (article text) + target (reference summary)
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
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Evalita-sp format: source (article) + target (reference summary)
            source = doc.get("source", "").strip()
            target = doc.get("target", "").strip()

            if not source or not target:
                log.debug("Skipping doc due to missing source or target", extra={"doc": doc})
                return None

            # Create prompt for summarization
            prompt = f"Summarize the following text:\n\n{source}"

            # Positive: correct reference summary
            correct_summary = target

            # Negative: shuffled version of the summary to create synthetic incorrect
            words = target.split()
            if len(words) < 5:
                log.debug("Summary too short to shuffle", extra={"doc": doc})
                return None

            shuffled_words = words.copy()
            random.shuffle(shuffled_words)
            incorrect_summary = ' '.join(shuffled_words)

            metadata = {"label": "evalita-sp"}

            return self._build_pair(
                question=prompt,
                correct=correct_summary,
                incorrect=incorrect_summary,
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
