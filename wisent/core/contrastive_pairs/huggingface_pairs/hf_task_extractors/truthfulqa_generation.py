from __future__ import annotations

import random
from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["TruthfulQAGenerationExtractor"]

log = setup_logger(__name__)

task_names = ("truthfulqa_generation", "truthfulqa_gen")

# Use semantic similarity evaluator that compares response to correct vs incorrect answers
evaluator_name = "truthfulqa_gen"


class TruthfulQAGenerationExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for the TruthfulQA Generation benchmark.

    This extractor formats prompts for open-ended generation evaluation,
    where models generate responses from scratch rather than selecting
    from multiple choice options.

    Evaluation uses semantic similarity to compare the model's response
    against correct_answers (should be similar) and incorrect_answers
    (should not be similar).

    TruthfulQA schema (truthfulqa/truthful_qa, generation config):
        - question: str
        - best_answer: str (truthful answer)
        - correct_answers: list (all truthful answers)
        - incorrect_answers: list (false/misleading answers)
        - category: str
        - type: str
    """

    # Use semantic similarity evaluator for generation tasks
    evaluator_name: str = "truthfulqa_gen"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from TruthfulQA for generation evaluation.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load TruthfulQA dataset from HuggingFace
        docs = self.load_dataset(
            dataset_name="truthfulqa/truthful_qa",
            dataset_config="generation",
            split="validation",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} TruthfulQA examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid TruthfulQA generation pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single TruthfulQA doc into a ContrastivePair for generation.

        The prompt is formatted as an open-ended question without multiple choice options,
        suitable for evaluating generated completions.

        Returns None when required fields are missing or malformed.
        """
        try:
            question = str(doc.get("question", "")).strip()
            best_answer = str(doc.get("best_answer", "")).strip()
            correct_answers = doc.get("correct_answers", [])
            incorrect_answers = doc.get("incorrect_answers", [])

            if not question or not best_answer or not incorrect_answers:
                log.debug(f"Skipping doc due to missing/invalid fields: {doc}")
                return None

            # For generation, we use the truthful answer as positive
            # and a randomly selected incorrect answer as negative
            correct = best_answer
            incorrect = random.choice(incorrect_answers) if len(incorrect_answers) > 1 else incorrect_answers[0]

            # Format as open-ended question for generation evaluation
            # No multiple choice options - the model should generate the answer
            formatted_question = f"Q: {question}\nA:"

            metadata = {
                "label": "truthfulqa_gen",
                "category": doc.get("category", ""),
                "type": doc.get("type", ""),
                "correct_answers": correct_answers if correct_answers else [best_answer],
                "incorrect_answers": incorrect_answers,
            }

            return self._build_pair(
                question=formatted_question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

