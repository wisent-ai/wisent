"""Extractors for multilingual NLP benchmarks."""
from __future__ import annotations

import random
from typing import Any

from wisent.core.utils.cli.cli_logger import setup_logger
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

__all__ = [
    "PawsXExtractor",
    "MLQAExtractor",
    "DarijaBenchExtractor",
    "EusExamsExtractor",
    "LambadaMultilingualExtractor",
]

log = setup_logger(__name__)


class PawsXExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for PAWS-X - Paraphrase Adversaries from Word Scrambling (Cross-lingual).

    Dataset: google-research-datasets/paws-x on HuggingFace

    PAWS-X is a challenging paraphrase identification dataset that contains
    pairs of sentences with high lexical overlap. Available in multiple languages.
    """

    evaluator_name = "paws_x"

    def __init__(self, language: str | None = None):
        """
        Initialize PAWS-X extractor.

        Args:
            language: Optional language filter (e.g., 'de', 'fr', 'es', 'zh', 'ja', 'ko')
        """
        super().__init__()
        self.language = language or "en"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from PAWS-X dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        try:
            docs = self.load_dataset(
                dataset_name="google-research-datasets/paws-x",
                dataset_config=self.language,
                split="test",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from PAWS-X ({self.language})")
        except Exception as e:
            log.error(f"Failed to load PAWS-X: {e}")
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
            sentence1 = doc.get("sentence1", "").strip()
            sentence2 = doc.get("sentence2", "").strip()
            label = doc.get("label", 0)

            if not sentence1 or not sentence2:
                return None

            task_prompt = f"""Determine if the following two sentences are paraphrases of each other.

Sentence 1: {sentence1}
Sentence 2: {sentence2}

Are these sentences paraphrases? Answer Yes or No:"""

            # label=1 means paraphrase, label=0 means not paraphrase
            if label == 1:
                correct = "Yes"
                incorrect = "No"
            else:
                correct = "No"
                incorrect = "Yes"

            metadata = {
                "label": "paws_x",
                "source": "google-research-datasets/paws-x",
                "language": self.language,
                "is_multilingual_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting PAWS-X pair: {exc}", exc_info=True)
            return None


class MLQAExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for MLQA - Multilingual Question Answering.

    Dataset: facebook/mlqa on HuggingFace

    MLQA is a benchmark for evaluating cross-lingual extractive QA performance.
    It contains QA instances in 7 languages.
    """

    evaluator_name = "mlqa"

    def __init__(self, language: str | None = None):
        """
        Initialize MLQA extractor.

        Args:
            language: Optional language filter (e.g., 'en', 'de', 'es', 'ar', 'hi', 'vi', 'zh')
        """
        super().__init__()
        self.language = language or "en"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from MLQA dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []
        all_answers: list[str] = []

        try:
            # MLQA config is like "mlqa.en.en" or "mlqa-translate-train.en"
            config = f"mlqa.{self.language}.{self.language}"
            docs = self.load_dataset(
                dataset_name="facebook/mlqa",
                dataset_config=config,
                split="test",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from MLQA ({self.language})")
        except Exception as e:
            log.error(f"Failed to load MLQA: {e}")
            return []

        # First pass: collect all answers for negative sampling
        for doc in docs:
            answers = doc.get("answers", {}).get("text", [])
            if answers and isinstance(answers, list) and len(answers) > 0:
                all_answers.append(answers[0])

        for doc in docs:
            pair = self._extract_pair_from_doc(doc, all_answers)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        return pairs

    def _extract_pair_from_doc(
        self, doc: dict[str, Any], all_answers: list[str]
    ) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            context = doc.get("context", "").strip()
            question = doc.get("question", "").strip()
            answers = doc.get("answers", {}).get("text", [])

            if not context or not question or not answers:
                return None

            correct_answer = answers[0] if answers else ""
            if not correct_answer:
                return None

            task_prompt = f"""Context: {context}

Question: {question}

Answer:"""

            # Get an incorrect answer from other examples
            negative_candidates = [a for a in all_answers if a != correct_answer]
            if negative_candidates:
                incorrect = random.choice(negative_candidates)
            else:
                incorrect = "I don't know."

            metadata = {
                "label": "mlqa",
                "source": "facebook/mlqa",
                "language": self.language,
                "is_multilingual_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct_answer,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting MLQA pair: {exc}", exc_info=True)
            return None



# Re-exports from split modules
from wisent.extractors.hf.hf_task_extractors.darija_eusexams import (
    DarijaBenchExtractor,
    EusExamsExtractor,
)
from wisent.extractors.hf.hf_task_extractors.lambada_multilingual import (
    LambadaMultilingualExtractor,
)
