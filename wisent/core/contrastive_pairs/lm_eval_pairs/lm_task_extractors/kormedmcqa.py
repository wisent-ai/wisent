from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["KormedmcqaExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "kormedmcqa",
    "kormedmcqa_dentist",
    "kormedmcqa_doctor",
    "kormedmcqa_nurse",
    "kormedmcqa_pharm",
)
class KormedmcqaExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Kormedmcqa benchmark."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Kormedmcqa docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Kormedmcqa.
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
            log.warning("No valid Kormedmcqa pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Kormedmcqa doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        Expected doc structure:
        {
            'question': str,
            'A': str,
            'B': str,
            'C': str,
            'D': str,
            'E': str,
            'answer': int (1-5, where 1=A, 2=B, etc.)
        }
        """
        log = bind(_LOG, doc_id=doc.get("q_number", "unknown"))

        try:
            question = str(doc.get("question", "")).strip()
            choices = [
                str(doc.get("A", "")).strip(),
                str(doc.get("B", "")).strip(),
                str(doc.get("C", "")).strip(),
                str(doc.get("D", "")).strip(),
                str(doc.get("E", "")).strip(),
            ]

            # answer is 1-indexed (1=A, 2=B, etc.)
            answer = doc.get("answer")

            if not question or not all(choices) or answer is None:
                log.debug(
                    "Skipping doc due to missing fields",
                    extra={"doc": doc},
                )
                return None

            # Convert 1-indexed answer to 0-indexed
            answer_idx = int(answer) - 1

            if not (0 <= answer_idx < len(choices)):
                log.debug(
                    "Skipping doc due to invalid answer index",
                    extra={"answer": answer, "answer_idx": answer_idx},
                )
                return None

            correct = choices[answer_idx]
            # Select a different choice as incorrect
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = choices[incorrect_idx]

            # Raw prompt without MC formatting
            prompt = question

            metadata = {
                "label": "kormedmcqa",
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
