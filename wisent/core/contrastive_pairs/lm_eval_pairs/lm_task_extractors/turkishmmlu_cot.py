from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["TurkishmmluCotExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "turkishmmlu_cot_biology",
    "turkishmmlu_cot_chemistry",
    "turkishmmlu_cot_geography",
    "turkishmmlu_cot_history",
    "turkishmmlu_cot_mathematics",
    "turkishmmlu_cot_philosophy",
    "turkishmmlu_cot_physics",
    "turkishmmlu_cot_religion_and_ethics",
    "turkishmmlu_cot_turkish_language_and_literature",
)
class TurkishmmluCotExtractor(LMEvalBenchmarkExtractor):
    """Extractor for TurkishMMLU CoT (chain-of-thought) benchmarks."""


    evaluator_name = "generation"
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
            log.warning("No valid TurkishMMLU CoT pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = doc.get("question", "").strip()
            choices = doc.get("choices", [])
            answer = doc.get("answer", "").strip()

            if not question or not choices or not answer:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            if len(choices) != 5:
                log.debug("Skipping doc due to invalid number of choices", extra={"doc": doc})
                return None

            # Build the question with choices
            choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
            formatted_question = f"Soru: {question}\n{choices_text}\nÇözüm: Adım adım düşünelim."

            # Correct answer is the letter (A-E)
            correct = answer.upper()

            # Pick a wrong answer
            wrong_choices = [chr(65+i) for i in range(5) if chr(65+i) != correct]
            if not wrong_choices:
                log.debug("No incorrect choices available", extra={"doc": doc})
                return None

            incorrect = wrong_choices[0]

            positive_response = PositiveResponse(model_response=correct)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt=formatted_question,
                positive_response=positive_response,
                negative_response=negative_response,
                label="turkishmmlu_cot",
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None
