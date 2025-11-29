from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

import json

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["UnitxtExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "20_newsgroups", "ag_news", "argument_topic", "banking77", "claim_stance_topic",
    "cnn_dailymail", "dbpedia_14", "ethos_binary",
    "financial_tweets", "law_stack_exchange", "ledgar", "medical_abstracts",
    "unfair_tos", "xsum", "yahoo_answers_topics",
)

class UnitxtExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Unitxt benchmark."""

    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Unitxt docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Unitxt.
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
            log.warning("No valid Unitxt pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Unitxt doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            
            task_data_str = doc.get("task_data", "{}")
            task_data = json.loads(task_data_str) if isinstance(task_data_str, str) else task_data_str

            classes = task_data.get("classes")
            summaries = task_data.get("summaries")

            source = doc.get("source", "").strip()
            target = doc.get("target", "").strip()

            if not (classes or summaries) or not target or not source:
                log.debug("Skipping doc due to missing text or (classes or summaries) or target or source", extra={"doc": doc})
                return None


            correct = target
            if classes:
                if correct in classes:
                    correct_idx = classes.index(correct)
                    incorrect = classes[(correct_idx + 1) % len(classes)]
                else:
                    incorrect = classes[0]
            elif summaries:
                incorrect = "This is an incorrect summary."
            else:
                return None

            formatted_question = f"{source}"
            
            metadata = {
                "label": "unitxt",
            }

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
