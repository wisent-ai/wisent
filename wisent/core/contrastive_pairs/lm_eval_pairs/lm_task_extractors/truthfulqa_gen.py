from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["TruthfulQAGenExtractor"]
_LOG = setup_logger(__name__)

task_names = ("truthfulqa_gen", "truthfulqa_generation")

# Use semantic similarity evaluator for generation tasks
evaluator_name = "truthfulqa_gen"


class TruthfulQAGenExtractor(LMEvalBenchmarkExtractor):
    """
    Extractor for the TruthfulQA Generation benchmark (lm-eval version).

    TruthfulQA_gen schema:
        - question: str
        - best_answer: str (truthful answer)
        - correct_answers: list[str] (all truthful answers)
        - incorrect_answers: list[str] (false/misleading answers)
        - category: str
        - type: str
    """

    evaluator_name: str = "truthfulqa_gen"

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from TruthfulQA_gen docs.

        Args:
            lm_eval_task_data: lm-eval task instance for TruthfulQA_gen.
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
            log.warning("No valid TruthfulQA_gen pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single TruthfulQA_gen doc into a ContrastivePair.
        """
        log = bind(_LOG, doc_id=doc.get("question", "unknown")[:50])

        try:
            question = str(doc.get("question", "")).strip()
            best_answer = str(doc.get("best_answer", "")).strip()
            correct_answers = doc.get("correct_answers", [])
            incorrect_answers = doc.get("incorrect_answers", [])
            category = doc.get("category", "")

            if not question:
                log.debug("Skipping doc: missing question")
                return None

            if not best_answer and not correct_answers:
                log.debug("Skipping doc: no correct answers")
                return None

            if not incorrect_answers:
                log.debug("Skipping doc: no incorrect answers")
                return None

            # Use best_answer as the correct response, or first correct answer
            correct = best_answer if best_answer else correct_answers[0]

            # Pick a random incorrect answer
            incorrect = random.choice(incorrect_answers)

            # Format prompt like the original task
            formatted_question = f"Q: {question}\nA:"

            metadata = {
                "label": "truthfulqa_gen",
                "category": category,
                "num_correct": len(correct_answers),
                "num_incorrect": len(incorrect_answers),
                "correct_answers": correct_answers,
                "incorrect_answers": incorrect_answers,
            }

            return self._build_pair(
                question=formatted_question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc)
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
            label=metadata.get("label") if metadata else None,
            metadata=metadata,
        )
