from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["TruthfulQAGenerationExtractor"]
_LOG = setup_logger(__name__)

# This task uses truthfulqa_gen from lm-eval-harness as the underlying task
# The name "truthfulqa_generation" is an alias for open-ended generation evaluation
task_names = ("truthfulqa_generation",)

# Maps to the underlying lm-eval task
lm_eval_task_name = "truthfulqa_gen"

evaluator_name = "generation"


class TruthfulQAGenerationExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the TruthfulQA Generation benchmark.

    This extractor formats prompts for open-ended generation evaluation,
    where models generate responses from scratch rather than selecting
    from multiple choice options.

    TruthfulQA_Gen schema:
        - question: str
        - best_answer: str (truthful answer)
        - correct_answers: list (all truthful answers)
        - incorrect_answers: list (false/misleading answers)
    """

    # Override base class default to use generation evaluator
    evaluator_name: str = "generation"

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from TruthfulQA_Gen docs for generation evaluation.

        Args:
            lm_eval_task_data: lm-eval task instance for TruthfulQA_Gen.
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items)

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs for generation", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid TruthfulQA generation pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single TruthfulQA_Gen doc into a ContrastivePair for generation.

        The prompt is formatted as an open-ended question without multiple choice options,
        suitable for evaluating generated completions.

        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = str(doc.get("question", "")).strip()
            best_answer = str(doc.get("best_answer", "")).strip()
            incorrect_answers = doc.get("incorrect_answers", [])

            if not question or not best_answer or not incorrect_answers:
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            # For generation, we use the truthful answer as positive
            # and a randomly selected incorrect answer as negative
            correct = best_answer
            incorrect = random.choice(incorrect_answers) if len(incorrect_answers) > 1 else incorrect_answers[0]

            # Format as open-ended question for generation evaluation
            # No multiple choice options - the model should generate the answer
            formatted_question = f"Q: {question}\nA:"

            metadata = {
                "label": "truthfulqa_generation",
                "category": doc.get("category", ""),
                "type": doc.get("type", ""),
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
