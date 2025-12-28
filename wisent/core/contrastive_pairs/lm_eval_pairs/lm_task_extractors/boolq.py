from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["BoolQExtractor"]
_LOG = setup_logger(__name__)

task_names = ("boolq",)
    
class BoolQExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the BoolQ benchmark."""

    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from BoolQ docs.

        BoolQ schema:
            - passage: str
            - question: str
            - label: 0 or 1

        Args:
            lm_eval_task_data: lm-eval task instance for BoolQ.
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
            log.warning("No valid BoolQ pairs extracted", extra={"task": task_name})

        return pairs
    
    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single BoolQ doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            passage = str(doc.get("passage", "")).strip()
            question = str(doc.get("question", "")).strip()
            label = doc.get("label")

            if not passage or not question or label not in {0, 1}:
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            correct = "yes" if label == 1 else "no"
            incorrect = "no" if label == 1 else "yes"

            prompt = f"{passage}\nQuestion: {question}?\nAnswer:"

            metadata = {
                "label": "boolq",
            }

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

    @staticmethod
    def extract_choices_and_answer(task, doc: dict[str, Any]) -> tuple[list[str], str]:
        """
        Extract choices and expected answer from a BoolQ document.

        Args:
            task: lm-eval task instance (has doc_to_choice, doc_to_target methods)
            doc: BoolQ document

        Returns:
            Tuple of (choices, expected_answer)
        """
        choices = task.doc_to_choice(doc)
        target_idx = task.doc_to_target(doc)
        expected = choices[target_idx]
        return choices, expected