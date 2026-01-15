from __future__ import annotations

import random
from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["TruthfulQACustomExtractor"]
_LOG = setup_logger(__name__)

task_names = ("truthfulqa_custom",)

evaluator_name = "truthfulqa_gen"


class TruthfulQACustomExtractor(HuggingFaceBenchmarkExtractor):
    """
    Custom extractor for TruthfulQA that loads directly from HuggingFace datasets.
    
    This provides more control over the data loading and pair generation compared
    to the standard lm-eval based extractor.
    
    Dataset: truthfulqa/truthful_qa (generation config)
    Schema:
        - question: str
        - best_answer: str (truthful answer)
        - correct_answers: list[str] (all truthful answers)
        - incorrect_answers: list[str] (false/misleading answers)
        - category: str
        - type: str
        - source: str
    """

    evaluator_name: str = "truthfulqa_gen"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs by loading directly from HuggingFace.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        from datasets import load_dataset
        
        log = bind(_LOG, task="truthfulqa_custom")
        log.info("Loading TruthfulQA from HuggingFace")

        dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        
        max_items = None if (limit is None or limit <= 0) else int(limit)
        if max_items is not None:
            dataset = dataset.select(range(min(max_items * 2, len(dataset))))

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(dataset)})

        for doc in dataset:
            pair = self._extract_pair_from_doc(dict(doc))
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid TruthfulQA custom pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single TruthfulQA doc into a ContrastivePair.
        """
        log = bind(_LOG, doc_id=doc.get("question", "unknown")[:50])

        try:
            question = str(doc.get("question", "")).strip()
            best_answer = str(doc.get("best_answer", "")).strip()
            correct_answers = doc.get("correct_answers", [])
            incorrect_answers = doc.get("incorrect_answers", [])
            category = doc.get("category", "")
            source = doc.get("source", "")

            if not question:
                log.debug("Skipping doc: missing question")
                return None

            if not best_answer and not correct_answers:
                log.debug("Skipping doc: no correct answers")
                return None

            if not incorrect_answers:
                log.debug("Skipping doc: no incorrect answers")
                return None

            correct = best_answer if best_answer else correct_answers[0]
            incorrect = random.choice(incorrect_answers)

            formatted_question = f"Q: {question}\nA:"

            metadata = {
                "label": "truthfulqa_custom",
                "category": category,
                "source": source,
                "num_correct": len(correct_answers),
                "num_incorrect": len(incorrect_answers),
                "correct_answers": correct_answers,
                "incorrect_answers": incorrect_answers,
                "best_answer": best_answer,
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
