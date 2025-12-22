from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["SycophancyExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "sycophancy",
    "sycophancy_on_nlp_survey",
    "sycophancy_on_philpapers2020",
    "sycophancy_on_political_typology_quiz",
)

class SycophancyExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Sycophancy benchmark.

    Tests whether models exhibit sycophantic behavior by agreeing with
    users' stated beliefs even when they should provide objective answers.
    """


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
        Convert a single sycophancy doc into a ContrastivePair.

        Sycophancy format: {
            question: "...",
            answer_matching_behavior: " (A)",
            answer_not_matching_behavior: " (B)"
        }

        For contrastive pairs:
        - Positive (correct): answer_not_matching_behavior (non-sycophantic)
        - Negative (incorrect): answer_matching_behavior (sycophantic)
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            if "question" not in doc:
                log.debug("Skipping doc due to missing question", extra={"doc": doc})
                return None

            question = str(doc["question"]).strip()

            # Sycophancy has specific fields for matching/not matching behavior
            answer_matching = doc.get("answer_matching_behavior", "").strip()
            answer_not_matching = doc.get("answer_not_matching_behavior", "").strip()

            if not question or not answer_matching or not answer_not_matching:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            # The non-sycophantic answer is the correct (positive) response
            correct = answer_not_matching
            # The sycophantic answer is the incorrect (negative) response
            incorrect = answer_matching
            metadata = {"label": "sycophancy"}

            return self._build_pair(
                question=question,
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
