from __future__ import annotations

from typing import Any
from wisent.core.utils.cli.cli_logger import setup_logger

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

log = setup_logger(__name__)

class OkapiTruthfulQAExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Okapi TruthfulQA - Multilingual TruthfulQA benchmark.

    Dataset: jon-tow/okapi_truthfulqa on HuggingFace

    Multilingual translation of TruthfulQA benchmark measuring truthfulness
    across 26 languages.
    """

    evaluator_name = "okapi_truthfulqa"

    def __init__(self, language: str | None = None):
        """
        Initialize Okapi TruthfulQA extractor.

        Args:
            language: Optional language filter
        """
        super().__init__()
        self.language = language

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from Okapi TruthfulQA dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        try:
            # Original okapi_truthfulqa uses deprecated loading script
            # Use alternative Spanish version as fallback
            docs = self.load_dataset(
                dataset_name="alvarobartt/truthfulqa-okapi-eval-es",
                split="validation",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from TruthfulQA Okapi (es)")
            self._dataset_format = "mc_targets"  # Uses mc1_targets format
        except Exception as e:
            log.error(f"Failed to load Okapi TruthfulQA: {e}")
            return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair.

        Supports two formats:
        1. Original: question, best_answer, incorrect_answers
        2. MC format: question, mc1_targets (with choices and labels)
        """
        try:
            question = doc.get("question", "").strip()

            if not question:
                return None

            # Handle mc1_targets format
            mc1_targets = doc.get("mc1_targets", {})
            if mc1_targets and "choices" in mc1_targets:
                choices = mc1_targets.get("choices", [])
                labels = mc1_targets.get("labels", [])

                if not choices:
                    return None

                # Find correct answer (label=1) and incorrect answer (label=0)
                correct = None
                incorrect = None
                for i, (choice, label) in enumerate(zip(choices, labels)):
                    if label == 1 and correct is None:
                        correct = choice.strip()
                    elif label == 0 and incorrect is None:
                        incorrect = choice.strip()

                if not correct:
                    correct = choices[0].strip() if choices else "Unknown"
                if not incorrect:
                    incorrect = choices[-1].strip() if len(choices) > 1 else "I don't know"
            else:
                # Original format
                best_answer = doc.get("best_answer", "").strip()
                incorrect_answers = doc.get("incorrect_answers", [])

                if not best_answer:
                    return None

                correct = best_answer
                if incorrect_answers and isinstance(incorrect_answers, list):
                    incorrect = incorrect_answers[0] if incorrect_answers else "I don't know."
                else:
                    incorrect = "I don't know."

            task_prompt = f"""Question: {question}

Provide a truthful answer:"""

            metadata = {
                "label": "okapi_truthfulqa",
                "source": "jon-tow/okapi_truthfulqa",
                "language": self.language or "multilingual",
                "is_multilingual_benchmark": True,
                "is_truthfulness_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting Okapi TruthfulQA pair: {exc}", exc_info=True)
            return None
