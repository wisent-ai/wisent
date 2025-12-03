from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["LibraExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "libra",
    "librusec_history",
    "librusec_mhqa",
    "long_context_multiq",
    "matreshka_names",
    "matreshka_yes_no",
    "passkey",
    "passkey_with_librusec",
    "ru_2wikimultihopqa",
    "ru_babilong_qa1",
    "ru_babilong_qa2",
    "ru_babilong_qa3",
    "ru_babilong_qa4",
    "ru_babilong_qa5",
    "ru_gsm100",
    "ru_qasper",
    "ru_quality",
    "ru_sci_abstract_retrieval",
    "ru_sci_passage_count",
)
class LibraExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Libra benchmark."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Libra docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Libra.
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
            log.warning("No valid Libra pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Libra doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        Expected doc structure:
        {
            'context': str,
            'input': str,
            'positive_outputs': list[str],
            'negative_outputs': list[str] (usually empty)
        }
        """
        metadata_dict = doc.get("metadata", {})
        doc_id = metadata_dict.get("id", "unknown") if isinstance(metadata_dict, dict) else "unknown"
        log = bind(_LOG, doc_id=doc_id)

        try:
            context = str(doc.get("context", "")).strip()
            input_text = str(doc.get("input", "")).strip()
            positive_outputs = doc.get("positive_outputs", [])

            if not context or not input_text or not positive_outputs:
                log.debug(
                    "Skipping doc due to missing fields",
                    extra={"has_context": bool(context), "has_input": bool(input_text), "has_positive": bool(positive_outputs)},
                )
                return None

            # The prompt is the combined context and input
            # Libra tasks use a Russian prompt format
            prompt = f"Тебе предоставляется длинный текст, в котором содержится ключ доступа. Запомни только ключ доступа.\n\n{context}\n\nВ ответе нужно указать только ключ доступа.\n\nВопрос:{input_text}\n\nОтвет:"

            # Use the first positive output as correct answer
            correct_answer = str(positive_outputs[0]).strip()

            # Create a synthetic negative by corrupting the correct answer
            # For numeric answers, change one digit
            # For text answers, truncate or modify
            if correct_answer.isdigit():
                # Change one random digit
                incorrect_answer = self._corrupt_numeric_answer(correct_answer)
            else:
                # For text, truncate or add wrong text
                incorrect_answer = correct_answer[:len(correct_answer)//2] if len(correct_answer) > 1 else "неправильный ответ"

            metadata = {
                "label": "libra",
            }

            return self._build_pair(
                question=prompt,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

    @staticmethod
    def _corrupt_numeric_answer(answer: str) -> str:
        """Corrupt a numeric answer by changing one digit."""
        if not answer or not answer.isdigit():
            return "0"

        # Change one random digit
        idx = random.randint(0, len(answer) - 1)
        digit = int(answer[idx])
        new_digit = (digit + random.randint(1, 9)) % 10
        return answer[:idx] + str(new_digit) + answer[idx+1:]

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
