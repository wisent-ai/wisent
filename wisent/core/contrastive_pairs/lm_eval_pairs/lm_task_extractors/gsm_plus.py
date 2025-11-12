from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["GSMPlusExtractor"]
_LOG = setup_logger(__name__)

task_names = ("gsm_plus",)

evaluator_name = "exact_match"


class GSMPlusExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the GSM-Plus benchmark - perturbed grade school math word problems."""

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from GSM-Plus docs.

        GSM-Plus schema:
            - question: str
            - answer: str
            - solution: str (optional)

        Args:
            lm_eval_task_data: lm-eval task instance for GSM-Plus.
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
            log.warning("No valid GSM-Plus pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single GSM-Plus doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = str(doc.get("question", "")).strip()
            answer = str(doc.get("answer", "")).strip()

            if not question or not answer:
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            # Extract numerical answer
            # GSM-Plus format: answer is just the number (e.g., "27")
            correct = answer

            # Create synthetic negative by adding 1 to the correct answer
            try:
                incorrect_val = float(answer.replace(',', '')) + 1
                incorrect = str(int(incorrect_val)) if incorrect_val == int(incorrect_val) else str(incorrect_val)
            except ValueError:
                # If answer is not numeric, use a generic wrong answer
                incorrect = "0"

            formatted_question = f"Question: {question}"

            metadata = {
                "label": "gsm_plus",
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
