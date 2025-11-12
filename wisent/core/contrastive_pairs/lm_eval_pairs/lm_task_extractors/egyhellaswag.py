from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["EgyHellaSwagExtractor"]
_LOG = setup_logger(__name__)


class EgyHellaSwagExtractor(LMEvalBenchmarkExtractor):
    """Extractor for EgyHellaSwag benchmark - Egyptian Arabic commonsense reasoning.

    This is a HellaSwag-style multiple choice task where models complete activity descriptions.
    Format: query (activity_label + context) + choices (list of endings) + gold (correct index)
    """

    task_names = ("egyhellaswag",)
    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Egyhellaswag docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Egyhellaswag.
            limit: Optional maximum number of pairs to produce.
            preferred_doc: Optional preferred document source.

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
            log.warning("No valid Egyhellaswag pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single EgyHellaSwag doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("ind", "unknown"))

        try:
            # Extract fields - EgyHellaSwag format
            query = doc.get("query", "").strip()
            choices = doc.get("choices", [])
            gold = doc.get("gold")

            if not query or not choices or gold is None:
                log.debug(
                    "Skipping doc due to missing fields",
                    extra={"has_query": bool(query), "has_choices": bool(choices), "has_gold": gold is not None},
                )
                return None

            # Validate gold index
            if not isinstance(gold, int) or not (0 <= gold < len(choices)):
                log.debug(
                    "Skipping doc due to invalid gold index",
                    extra={"gold": gold, "num_choices": len(choices)},
                )
                return None

            # Get correct and incorrect answers
            correct = choices[gold]
            # Use the next choice as incorrect (wrapping around)
            incorrect_idx = (gold + 1) % len(choices)
            incorrect = choices[incorrect_idx]

            metadata = {
                "label": "egyhellaswag",
            }

            return self._build_pair(
                question=query,
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
