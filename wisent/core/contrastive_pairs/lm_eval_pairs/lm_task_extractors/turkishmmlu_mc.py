from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["TurkishmmluMultipleChoiceExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "turkishmmlu_biology",
    "turkishmmlu_chemistry",
    "turkishmmlu_geography",
    "turkishmmlu_history",
    "turkishmmlu_mathematics",
    "turkishmmlu_philosophy",
    "turkishmmlu_physics",
    "turkishmmlu_religion_and_ethics",
    "turkishmmlu_turkish_language_and_literature",
)
class TurkishmmluMultipleChoiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for TurkishMMLU multiple-choice benchmarks."""


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
            log.warning("No valid TurkishMMLU MC pairs extracted", extra={"task": task_name})

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

            # Find correct answer index
            answer_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
            answer_idx = answer_map.get(answer.upper())

            if answer_idx is None:
                log.debug("Skipping doc due to invalid answer", extra={"doc": doc, "answer": answer})
                return None

            correct = str(choices[answer_idx]).strip()
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = str(choices[incorrect_idx]).strip()

            positive_response = PositiveResponse(model_response=correct)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt=formatted_question,
                positive_response=positive_response,
                negative_response=negative_response,
                label="turkishmmlu_mc",
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None
