from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["TruthfulQAMC1Extractor"]
_LOG = setup_logger(__name__)


class TruthfulQAMC1Extractor(LMEvalBenchmarkExtractor):
    """Extractor for the TruthfulQA MC1 benchmark."""

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from TruthfulQA MC1 docs.

        TruthfulQA MC1 schema:
            - question: str
            - mc1_targets: dict with 'choices' (list of answers) and 'labels' (list of 0/1)

        Args:
            lm_eval_task_data: lm-eval task instance for TruthfulQA MC1.
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items)

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
            log.warning("No valid TruthfulQA MC1 pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single TruthfulQA MC1 doc into a ContrastivePair.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("idx", "unknown"))

        try:
            question = str(doc.get("question", "")).strip()
            mc1_targets = doc.get("mc1_targets", {})

            if not question or not mc1_targets:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            choices = mc1_targets.get("choices", [])
            labels = mc1_targets.get("labels", [])

            if not choices or not labels or len(choices) != len(labels):
                log.debug("Skipping doc due to invalid targets", extra={"doc": doc})
                return None

            # Find correct and incorrect answers
            correct_answers = [choices[i] for i, label in enumerate(labels) if label == 1]
            incorrect_answers = [choices[i] for i, label in enumerate(labels) if label == 0]

            if not correct_answers or not incorrect_answers:
                log.debug("Skipping doc: no correct or incorrect answers", extra={"doc": doc})
                return None

            # Use first correct and first incorrect
            correct = correct_answers[0]
            incorrect = incorrect_answers[0]

            # Format question with choices
            formatted_question = f"Question: {question}\nChoices:\n"
            for i, choice in enumerate(choices):
                formatted_question += f"{chr(65+i)}. {choice}\n"
            formatted_question += "Answer:"

            metadata = {
                "label": "truthfulqa_mc1",
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
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None
        )
