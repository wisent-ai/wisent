from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["YahooExtractor"]
_LOG = setup_logger(__name__)

task_names = ("yahoo_answers_topics",)

class YahooExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Yahoo benchmark."""


    evaluator_name = "log_likelihoods"
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

        print("running yahoo")

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
            # Yahoo answers topics format: source (classification prompt) and target (category)
            source = doc.get("source", "").strip()
            target = doc.get("target", "").strip()

            if not source or not target:
                log.debug("Skipping doc due to missing source or target", extra={"doc": doc})
                return None

            # Use the source as prompt and target as correct response
            # For incorrect response, use a different category
            # Common Yahoo Answers categories to use as alternatives
            categories = [
                "Society & Culture",
                "Science & Mathematics",
                "Health",
                "Education & Reference",
                "Computers & Internet",
                "Sports",
                "Business & Finance",
                "Entertainment & Music",
                "Family & Relationships",
                "Politics & Government"
            ]

            # Find an incorrect category (different from target)
            incorrect = "Society & Culture"  # default
            for cat in categories:
                if cat != target:
                    incorrect = cat
                    break

            metadata = {"label": "yahoo"}

            return self._build_pair(
                question=source,
                correct=target,
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
