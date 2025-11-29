from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MediqaQa2019Extractor"]
_LOG = setup_logger(__name__)

task_names = ("mediqa_qa2019",)

class MediqaQa2019Extractor(LMEvalBenchmarkExtractor):
    """Extractor for the Mediqa Qa2019 benchmark."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Mediqa Qa2019 docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Mediqa Qa2019.
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
            log.warning("No valid Mediqa Qa2019 pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Mediqa Qa2019 doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        Format: doc["QUESTION"]["QuestionText"] and doc["QUESTION"]["AnswerList"][0]["Answer"]["AnswerText"]
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Mediqa QA 2019 format: doc["QUESTION"] contains question and answer
            if "QUESTION" in doc:
                question_obj = doc["QUESTION"]
                question = str(question_obj.get("QuestionText", "")).strip()

                answer_list = question_obj.get("AnswerList", [])
                if not answer_list or not isinstance(answer_list, list):
                    log.debug("No answer list found", extra={"doc": doc})
                    return None

                # Get the first answer as correct
                first_answer = answer_list[0]
                if not isinstance(first_answer, dict) or "Answer" not in first_answer:
                    log.debug("Invalid answer format", extra={"doc": doc})
                    return None

                correct_answer = str(first_answer["Answer"].get("AnswerText", "")).strip()

                if not question or not correct_answer:
                    log.debug("Missing question or answer", extra={"doc": doc})
                    return None

                # Create synthetic negative - use generic non-answer
                # This represents a model that refuses to provide medical information
                incorrect_answer = "I'm not able to provide specific medical information. Please consult with a healthcare professional."

                metadata = {"label": "mediqa_qa2019"}
                return self._build_pair(
                    question=f"Question: {question}",
                    correct=correct_answer,
                    incorrect=incorrect_answer,
                    metadata=metadata,
                )

            return None

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
