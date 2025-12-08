from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

from latex2sympy2_extended import latex2sympy
from sympy import latex
from wisent.core.evaluators.benchmark_specific.math_parsing.scripts import strip_string

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["GsmExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "gsm_plus",
    "gsm_plus_mini",
)


class GsmExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Gsm benchmark - math word problems."""


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
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:

            question = doc.get("question", "").strip()
            answer = doc.get("answer", "").strip()

            if not question or not answer:
                log.debug("Skipping doc due to missing/invalid fields", extra={"doc": doc})
                return None
            
            correct = answer
            if answer == "None":
                incorrect = "42"
            else:
                correct = strip_string(correct)
                incorrect = self._create_incorrect_answer(correct)

            formatted_question = f"Question: {question}"
            metadata = {"label": "gsm"}

            return self._build_pair(
                question=formatted_question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None
        
    def _create_incorrect_answer(self, correct: str) -> str:
        """Create an incorrect answer by modifying the correct one (input is already stripped)."""
        try:
            parsed_correct = latex2sympy(correct)
            incorrect = latex(parsed_correct + 1)
            return str(incorrect)
        except Exception:
            return f"{correct} + 1"

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
