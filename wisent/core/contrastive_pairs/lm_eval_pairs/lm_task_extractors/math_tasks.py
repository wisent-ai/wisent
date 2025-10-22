from __future__ import annotations

from typing import Any, TYPE_CHECKING
import random

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["GenericMathExtractor", "GSM8KExtractor"]
_LOG = setup_logger(__name__)


class GenericMathExtractor(LMEvalBenchmarkExtractor):
    """
    Generic extractor for math tasks.

    For math tasks, we create contrastive pairs by:
    - Positive: correct answer
    - Negative: random incorrect number (modified answer)
    """

    def __init__(
        self,
        task_label: str,
        question_field: str = "question",
        answer_field: str = "answer",
    ):
        """
        Initialize generic math extractor.

        Args:
            task_label: Label to use in metadata (e.g., "gsm8k")
            question_field: Field containing the question
            answer_field: Field containing the answer
        """
        self.task_label = task_label
        self.question_field = question_field
        self.answer_field = answer_field

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from math task."""
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
            log.warning(f"No valid {self.task_label} pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a math doc into a ContrastivePair."""
        log = bind(_LOG, doc_id=doc.get("idx", doc.get("id", "unknown")))

        try:
            question = str(doc.get(self.question_field, "")).strip()
            answer = doc.get(self.answer_field, "")

            if not question or not answer:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            # Extract numerical answer (handle formats like "#### 42" in GSM8K)
            answer_str = str(answer).strip()
            if "####" in answer_str:
                # GSM8K format: explanation #### numerical_answer
                numerical_answer = answer_str.split("####")[-1].strip()
            else:
                numerical_answer = answer_str

            # Create incorrect answer (add/subtract/multiply by factor)
            try:
                correct_num = float(numerical_answer.replace(",", ""))
                # Generate incorrect answer
                if correct_num == 0:
                    incorrect_num = random.choice([1, -1, 2, -2])
                else:
                    # Random modification: +/- 10-50%, or add/subtract 1-10
                    modification = random.choice([
                        correct_num * random.uniform(1.1, 1.5),  # 10-50% more
                        correct_num * random.uniform(0.5, 0.9),  # 10-50% less
                        correct_num + random.randint(1, 10),     # add 1-10
                        correct_num - random.randint(1, 10),     # subtract 1-10
                    ])
                    incorrect_num = modification

                # Format as integer if original was integer
                if "." not in numerical_answer and "," not in numerical_answer:
                    incorrect_answer = str(int(incorrect_num))
                else:
                    incorrect_answer = f"{incorrect_num:.2f}"

            except (ValueError, TypeError):
                # If not a number, just append "wrong"
                incorrect_answer = f"{numerical_answer} (wrong)"

            formatted_question = f"Question: {question}\nProvide the numerical answer:"

            metadata = {"label": self.task_label}

            return self._build_pair(
                question=formatted_question,
                correct=numerical_answer,
                incorrect=incorrect_answer,
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


# Pre-configured extractors for common math tasks
class GSM8KExtractor(GenericMathExtractor):
    """Extractor for GSM8K benchmark."""
    def __init__(self):
        super().__init__(
            task_label="gsm8k",
            question_field="question",
            answer_field="answer",
        )


class MATHExtractor(GenericMathExtractor):
    """Extractor for MATH benchmark."""
    def __init__(self):
        super().__init__(
            task_label="math",
            question_field="problem",
            answer_field="solution",
        )


class MATH500Extractor(GenericMathExtractor):
    """Extractor for MATH-500 benchmark."""
    def __init__(self):
        super().__init__(
            task_label="math500",
            question_field="problem",
            answer_field="solution",
        )
