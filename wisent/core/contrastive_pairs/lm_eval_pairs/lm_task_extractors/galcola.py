from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["GalcolaExtractor"]
_LOG = setup_logger(__name__)

task_names = ("galcola",)

class GalcolaExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Galcola benchmark."""


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
        Convert a single Galcola doc into a ContrastivePair.

        Galcola format:
        - sentence: the sentence to judge
        - label: 1 if acceptable, 0 if not acceptable

        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            sentence = str(doc.get("sentence", "")).strip()
            label = doc.get("label")

            if not sentence or label is None:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            # Convert label to int
            if isinstance(label, str):
                label = int(label)
            elif not isinstance(label, int):
                log.debug("Skipping doc due to invalid label type", extra={"doc": doc})
                return None

            # Create prompt asking about acceptability
            prompt = f"Is the following sentence grammatically acceptable?\n{sentence}"

            # Determine correct and incorrect responses based on label
            if label == 1:
                correct = "Yes, it is acceptable."
                incorrect = "No, it is not acceptable."
            else:
                correct = "No, it is not acceptable."
                incorrect = "Yes, it is acceptable."

            metadata = {"label": "galcola"}

            return self._build_pair(
                question=prompt,
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
