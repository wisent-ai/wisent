from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MgsmExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "mgsm",
    "mgsm_direct", "mgsm_direct_en", "mgsm_direct_es", "mgsm_direct_fr", "mgsm_direct_de", "mgsm_direct_ru", "mgsm_direct_zh", "mgsm_direct_ja", "mgsm_direct_th", "mgsm_direct_sw", "mgsm_direct_bn", "mgsm_direct_te",
    "mgsm_cot_native", "mgsm_cot_native_bn", "mgsm_cot_native_de", "mgsm_cot_native_en", "mgsm_cot_native_es", "mgsm_cot_native_fr", "mgsm_cot_native_ja", "mgsm_cot_native_ru", "mgsm_cot_native_sw", "mgsm_cot_native_te", "mgsm_cot_native_th", "mgsm_cot_native_zh",
    "mgsm_native_cot_bn", "mgsm_native_cot_de", "mgsm_native_cot_en", "mgsm_native_cot_es", "mgsm_native_cot_fr", "mgsm_native_cot_ja", "mgsm_native_cot_ru", "mgsm_native_cot_sw", "mgsm_native_cot_te", "mgsm_native_cot_th", "mgsm_native_cot_zh"
)

class MgsmExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Mgsm benchmark."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Mgsm docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Mgsm.
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
            log.warning("No valid Mgsm pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Mgsm doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # mgsm format: question + answer_number
            if "question" in doc and "answer_number" in doc:
                question = str(doc.get("question", "")).strip()
                answer_number = doc.get("answer_number")

                if not question or answer_number is None:
                    log.debug("Skipping doc with missing question or answer_number", extra={"doc": doc})
                    return None

                # Convert answer_number to string for the correct response
                correct = str(answer_number)

                # Create an incorrect answer by modifying the number
                # For small numbers (0-2), add 2; otherwise subtract 1
                if answer_number < 3:
                    incorrect_number = answer_number + 2
                else:
                    incorrect_number = answer_number - 1
                incorrect = str(incorrect_number)

                formatted_question = f"Question: {question}\nAnswer:"
                metadata = {"label": "mgsm"}

                return self._build_pair(
                    question=formatted_question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            log.debug("Skipping doc without question/answer_number fields", extra={"doc": doc})
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
