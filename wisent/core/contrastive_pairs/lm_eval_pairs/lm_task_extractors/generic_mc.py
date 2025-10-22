from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["GenericMultipleChoiceExtractor"]
_LOG = setup_logger(__name__)


class GenericMultipleChoiceExtractor(LMEvalBenchmarkExtractor):
    """
    Generic extractor for multiple choice tasks.

    Handles tasks with structure:
    - Some question/context field
    - choices/endings field (list of options)
    - label/gold field (correct answer index)
    """

    def __init__(
        self,
        task_label: str,
        question_field: str = "query",
        choices_field: str = "choices",
        label_field: str = "gold",
        context_fields: list[str] | None = None,
    ):
        """
        Initialize generic MC extractor.

        Args:
            task_label: Label to use in metadata (e.g., "hellaswag")
            question_field: Field containing the question/query
            choices_field: Field containing list of choices
            label_field: Field containing the correct answer index
            context_fields: Optional list of fields to concatenate for context
        """
        self.task_label = task_label
        self.question_field = question_field
        self.choices_field = choices_field
        self.label_field = label_field
        self.context_fields = context_fields or []

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from multiple choice task."""
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
        """Convert a doc into a ContrastivePair."""
        log = bind(_LOG, doc_id=doc.get("idx", doc.get("id", "unknown")))

        try:
            # Get question/query
            question = str(doc.get(self.question_field, "")).strip()

            # Build context from multiple fields if specified
            context_parts = []
            for field in self.context_fields:
                if field in doc and doc[field]:
                    context_parts.append(str(doc[field]).strip())

            if context_parts:
                question = " ".join(context_parts) + " " + question

            # Get choices
            choices = doc.get(self.choices_field, [])
            if not isinstance(choices, list):
                choices = list(choices) if choices else []

            # Get correct label
            label = doc.get(self.label_field)
            if label is None:
                log.debug("Skipping doc due to missing label", extra={"doc": doc})
                return None

            # Convert label to int
            try:
                correct_idx = int(label)
            except (ValueError, TypeError):
                log.debug("Skipping doc due to invalid label", extra={"label": label})
                return None

            if not question or not choices or correct_idx < 0 or correct_idx >= len(choices):
                log.debug("Skipping doc due to missing/invalid fields", extra={"doc": doc})
                return None

            # Get correct and incorrect answers
            correct = choices[correct_idx]

            # Get first incorrect answer (any index != correct_idx)
            incorrect_options = [c for i, c in enumerate(choices) if i != correct_idx]
            if not incorrect_options:
                log.debug("Skipping doc: no incorrect answers", extra={"doc": doc})
                return None

            incorrect = incorrect_options[0]

            # Format question with choices
            formatted_question = f"{question}\nChoices:\n"
            for i, choice in enumerate(choices):
                formatted_question += f"{chr(65+i)}. {choice}\n"
            formatted_question += "Answer:"

            metadata = {
                "label": self.task_label,
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


# Pre-configured extractors for common tasks
class HellaSwagExtractor(GenericMultipleChoiceExtractor):
    """Extractor for HellaSwag benchmark."""
    def __init__(self):
        super().__init__(
            task_label="hellaswag",
            question_field="query",
            choices_field="choices",
            label_field="gold",
        )


class COPAExtractor(GenericMultipleChoiceExtractor):
    """Extractor for COPA benchmark."""
    def __init__(self):
        super().__init__(
            task_label="copa",
            question_field="premise",
            choices_field="choices",
            label_field="label",
        )

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        # COPA specific: choices are in choice1, choice2 fields
        if "choice1" in doc and "choice2" in doc:
            doc = dict(doc)  # Copy to avoid modifying original
            doc["choices"] = [doc["choice1"], doc["choice2"]]
        return super()._extract_pair_from_doc(doc)


class PIQAExtractor(GenericMultipleChoiceExtractor):
    """Extractor for PIQA benchmark."""
    def __init__(self):
        super().__init__(
            task_label="piqa",
            question_field="goal",
            choices_field="choices",
            label_field="label",
        )


class OpenBookQAExtractor(GenericMultipleChoiceExtractor):
    """Extractor for OpenBookQA benchmark."""
    def __init__(self):
        super().__init__(
            task_label="openbookqa",
            question_field="question_stem",
            choices_field="choices",
            label_field="answerKey",
        )

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        # OpenBookQA specific: answerKey is letter (A, B, C, D)
        if "answerKey" in doc:
            doc = dict(doc)  # Copy
            answer_key = doc["answerKey"]
            # Convert letter to index
            if isinstance(answer_key, str) and answer_key.isalpha():
                doc["answerKey"] = ord(answer_key.upper()) - ord('A')

        # Choices are in dict format {'text': 'answer', 'label': 'A'}
        if "choices" in doc and isinstance(doc["choices"], dict):
            doc = dict(doc)
            choices_dict = doc["choices"]
            if "text" in choices_dict:
                doc["choices"] = choices_dict["text"]

        return super()._extract_pair_from_doc(doc)


class ARCExtractor(GenericMultipleChoiceExtractor):
    """Extractor for ARC (easy/challenge) benchmarks."""
    def __init__(self, task_label: str = "arc"):
        super().__init__(
            task_label=task_label,
            question_field="question",
            choices_field="choices",
            label_field="answerKey",
        )

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        # ARC specific: answerKey is letter or number
        if "answerKey" in doc:
            doc = dict(doc)  # Copy
            answer_key = doc["answerKey"]
            # Convert to index
            if isinstance(answer_key, str):
                if answer_key.isalpha():
                    doc["answerKey"] = ord(answer_key.upper()) - ord('A')
                elif answer_key.isdigit():
                    doc["answerKey"] = int(answer_key) - 1

        # Choices might be in dict format
        if "choices" in doc and isinstance(doc["choices"], dict):
            doc = dict(doc)
            choices_dict = doc["choices"]
            if "text" in choices_dict:
                doc["choices"] = choices_dict["text"]

        return super()._extract_pair_from_doc(doc)


class SWAGExtractor(GenericMultipleChoiceExtractor):
    """Extractor for SWAG benchmark."""
    def __init__(self):
        super().__init__(
            task_label="swag",
            question_field="sent2",
            choices_field="endings",
            label_field="label",
            context_fields=["sent1"],
        )
