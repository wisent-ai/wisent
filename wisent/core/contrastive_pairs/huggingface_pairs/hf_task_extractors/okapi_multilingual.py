"""Extractors for Okapi multilingual benchmarks (MMLU, HellaSwag, TruthfulQA)."""
from __future__ import annotations

import random
from typing import Any

from wisent.core.cli_logger import setup_logger
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = [
    "OkapiMMLUExtractor",
    "OkapiHellaswagExtractor",
    "OkapiTruthfulQAExtractor",
]

log = setup_logger(__name__)


class OkapiMMLUExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Okapi MMLU - Multilingual MMLU benchmark.

    Dataset: jon-tow/okapi_mmlu on HuggingFace

    Multilingual translation of MMLU (Measuring Massive Multitask Language
    Understanding) covering 57 tasks across 26 languages.
    """

    evaluator_name = "multiple_choice"

    def __init__(self, language: str | None = None):
        """
        Initialize Okapi MMLU extractor.

        Args:
            language: Optional language filter (e.g., 'de', 'fr', 'es')
        """
        super().__init__()
        self.language = language

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from Okapi MMLU dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        try:
            # Load dataset - try different configs
            config = self.language if self.language else "de"
            docs = self.load_dataset(
                dataset_name="jon-tow/okapi_mmlu",
                split="test",
                config=config,
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from Okapi MMLU ({config})")
        except Exception as e:
            log.error(f"Failed to load Okapi MMLU: {e}")
            return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            question = doc.get("question", "").strip()
            choices = doc.get("choices", [])
            answer_idx = doc.get("answer", 0)

            if not question or not choices:
                return None

            # Build multiple choice prompt
            choice_letters = ['A', 'B', 'C', 'D']
            choices_text = "\n".join(
                f"{choice_letters[i]}. {c}" for i, c in enumerate(choices[:4])
            )

            task_prompt = f"""Question: {question}

{choices_text}

Answer:"""

            # Correct answer
            if isinstance(answer_idx, int) and answer_idx < len(choices):
                correct = choice_letters[answer_idx]
            else:
                correct = "A"

            # Incorrect answer
            wrong_indices = [i for i in range(len(choices)) if i != answer_idx]
            incorrect = choice_letters[random.choice(wrong_indices)] if wrong_indices else "B"

            metadata = {
                "label": "okapi_mmlu",
                "source": "jon-tow/okapi_mmlu",
                "language": self.language or "multilingual",
                "is_multilingual_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting Okapi MMLU pair: {exc}", exc_info=True)
            return None


class OkapiHellaswagExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Okapi HellaSwag - Multilingual HellaSwag benchmark.

    Dataset: jon-tow/okapi_hellaswag on HuggingFace

    Multilingual translation of HellaSwag commonsense inference benchmark
    across 26 languages.
    """

    evaluator_name = "multiple_choice"

    def __init__(self, language: str | None = None):
        """
        Initialize Okapi HellaSwag extractor.

        Args:
            language: Optional language filter
        """
        super().__init__()
        self.language = language

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from Okapi HellaSwag dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        try:
            config = self.language if self.language else "de"
            docs = self.load_dataset(
                dataset_name="jon-tow/okapi_hellaswag",
                split="validation",
                config=config,
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from Okapi HellaSwag ({config})")
        except Exception as e:
            log.error(f"Failed to load Okapi HellaSwag: {e}")
            return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            ctx = doc.get("ctx", doc.get("context", "")).strip()
            endings = doc.get("endings", [])
            label = doc.get("label", 0)

            if not ctx or not endings:
                return None

            # Build completion prompt
            choice_letters = ['A', 'B', 'C', 'D']
            choices_text = "\n".join(
                f"{choice_letters[i]}. {e}" for i, e in enumerate(endings[:4])
            )

            task_prompt = f"""Complete the following:

{ctx}

Options:
{choices_text}

Most likely completion:"""

            # Correct answer
            if isinstance(label, int) and label < len(endings):
                correct = choice_letters[label]
            else:
                correct = "A"

            # Incorrect answer
            wrong_indices = [i for i in range(len(endings)) if i != label]
            incorrect = choice_letters[random.choice(wrong_indices)] if wrong_indices else "B"

            metadata = {
                "label": "okapi_hellaswag",
                "source": "jon-tow/okapi_hellaswag",
                "language": self.language or "multilingual",
                "is_multilingual_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting Okapi HellaSwag pair: {exc}", exc_info=True)
            return None


class OkapiTruthfulQAExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Okapi TruthfulQA - Multilingual TruthfulQA benchmark.

    Dataset: jon-tow/okapi_truthfulqa on HuggingFace

    Multilingual translation of TruthfulQA benchmark measuring truthfulness
    across 26 languages.
    """

    evaluator_name = "truthfulness"

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
            config = self.language if self.language else "de"
            docs = self.load_dataset(
                dataset_name="jon-tow/okapi_truthfulqa",
                split="validation",
                config=config,
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from Okapi TruthfulQA ({config})")
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
        """Convert a single doc into a ContrastivePair."""
        try:
            question = doc.get("question", "").strip()
            best_answer = doc.get("best_answer", "").strip()
            incorrect_answers = doc.get("incorrect_answers", [])

            if not question or not best_answer:
                return None

            task_prompt = f"""Question: {question}

Provide a truthful answer:"""

            # Correct answer
            correct = best_answer

            # Incorrect answer
            if incorrect_answers and isinstance(incorrect_answers, list):
                incorrect = incorrect_answers[0] if incorrect_answers else "I don't know."
            else:
                incorrect = "I don't know."

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
