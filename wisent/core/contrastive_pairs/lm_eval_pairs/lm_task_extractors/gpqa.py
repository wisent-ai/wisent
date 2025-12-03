from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["GPQAExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "gpqa",
    "gpqa_diamond_cot_n_shot",
    "gpqa_diamond_cot_zeroshot",
    "gpqa_diamond_generative_n_shot",
    "gpqa_diamond_n_shot",
    "gpqa_diamond_zeroshot",
    "gpqa_extended_cot_n_shot",
    "gpqa_extended_cot_zeroshot",
    "gpqa_extended_generative_n_shot",
    "gpqa_extended_n_shot",
    "gpqa_extended_zeroshot",
    "gpqa_main_cot_n_shot",
    "gpqa_main_cot_zeroshot",
    "gpqa_main_generative_n_shot",
    "gpqa_main_n_shot",
    "gpqa_main_zeroshot",
)

class GPQAExtractor(LMEvalBenchmarkExtractor):
    """Extractor for GPQA (Graduate-Level Google-Proof Q&A) benchmark."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from GPQA docs.

        GPQA schema:
            - Question: str
            - Correct Answer: str
            - Incorrect Answer 1-3: str

        Args:
            lm_eval_task_data: lm-eval task instance for GPQA.
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
            pair = self._extract_pair_from_doc(doc, lm_eval_task_data)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid GPQA pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(
        self, doc: dict[str, Any], task_data: Any = None
    ) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            # Handle both raw format (from Idavidrein/gpqa dataset) and processed format
            if not all(key in doc for key in ["Question", "Correct Answer", "Incorrect Answer 1"]):
                _LOG.debug("Skipping: missing required fields")
                return None

            question = doc["Question"].strip()
            correct_answer = doc["Correct Answer"].strip()
            incorrect_answers = [
                doc.get("Incorrect Answer 1", "").strip(),
                doc.get("Incorrect Answer 2", "").strip(),
                doc.get("Incorrect Answer 3", "").strip(),
            ]
            # Filter out empty answers
            incorrect_answers = [ans for ans in incorrect_answers if ans]

            if not question or not correct_answer or not incorrect_answers:
                _LOG.debug("Skipping: empty question, correct answer, or no incorrect answers")
                return None

            # Create choices list with correct answer first
            choices = [correct_answer] + incorrect_answers

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                choice_letters = ["A", "B", "C", "D"]
                formatted_choices = []
                for i, choice in enumerate(choices[:4]):
                    if choice:
                        formatted_choices.append(f"({choice_letters[i]}) {choice}")
                formatted_question = f"{question}\nChoices:\n" + "\n".join(formatted_choices)

            # Randomly select one incorrect answer
            incorrect_answer = random.choice(incorrect_answers)

            metadata = {
                "label": "gpqa",
                "source": getattr(task_data, "NAME", "gpqa"),
            }

            return self._build_pair(
                question=formatted_question,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            _LOG.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        """Build a ContrastivePair from question and responses."""
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None,
            metadata=metadata,
        )
