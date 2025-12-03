from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MlqaExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "mlqa",
    # Arabic context
    "mlqa_ar_ar", "mlqa_ar_de", "mlqa_ar_vi", "mlqa_ar_zh", "mlqa_ar_en", "mlqa_ar_es", "mlqa_ar_hi",
    # German context
    "mlqa_de_ar", "mlqa_de_de", "mlqa_de_vi", "mlqa_de_zh", "mlqa_de_en", "mlqa_de_es", "mlqa_de_hi",
    # Vietnamese context
    "mlqa_vi_ar", "mlqa_vi_de", "mlqa_vi_vi", "mlqa_vi_zh", "mlqa_vi_en", "mlqa_vi_es", "mlqa_vi_hi",
    # Chinese context
    "mlqa_zh_ar", "mlqa_zh_de", "mlqa_zh_vi", "mlqa_zh_zh", "mlqa_zh_en", "mlqa_zh_es", "mlqa_zh_hi",
    # English context
    "mlqa_en_ar", "mlqa_en_de", "mlqa_en_vi", "mlqa_en_zh", "mlqa_en_en", "mlqa_en_es", "mlqa_en_hi",
    # Spanish context
    "mlqa_es_ar", "mlqa_es_de", "mlqa_es_vi", "mlqa_es_zh", "mlqa_es_en", "mlqa_es_es", "mlqa_es_hi",
    # Hindi context
    "mlqa_hi_ar", "mlqa_hi_de", "mlqa_hi_vi", "mlqa_hi_zh", "mlqa_hi_en", "mlqa_hi_es", "mlqa_hi_hi"
)

class MlqaExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Mlqa benchmark."""


    evaluator_name = "exact_match"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Mlqa docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Mlqa.
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
            log.warning("No valid Mlqa pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Mlqa doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        Mlqa format: context + question + answers (QA task like SQuAD)
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # mlqa format: context + question + answers (list of answer texts)
            if "context" in doc and "question" in doc and "answers" in doc:
                context = str(doc.get("context", "")).strip()
                question = str(doc.get("question", "")).strip()
                answers = doc.get("answers")

                # answers can be a dict with "text" key or a list
                if isinstance(answers, dict):
                    answer_texts = answers.get("text", [])
                elif isinstance(answers, list):
                    answer_texts = answers
                else:
                    answer_texts = []

                if not context or not question or not answer_texts:
                    log.debug("Skipping doc with missing context, question, or answers", extra={"doc": doc})
                    return None

                # Use the first answer as the correct answer
                correct_answer = str(answer_texts[0]).strip() if answer_texts else ""

                if not correct_answer:
                    log.debug("Skipping doc with empty answer", extra={"doc": doc})
                    return None

                # Create synthetic negative - generic non-answer for QA
                incorrect_answer = "I don't know."

                prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

                metadata = {"label": "mlqa"}
                return self._build_pair(
                    question=prompt,
                    correct=correct_answer,
                    incorrect=incorrect_answer,
                    metadata=metadata,
                )

            log.debug("Skipping doc without context/question/answers fields", extra={"doc": doc})
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
