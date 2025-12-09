from __future__ import annotations

import random
import re
from typing import Any

from wisent.core.cli_logger import setup_logger
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["ConalaExtractor"]

log = setup_logger(__name__)


class ConalaExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for conala dataset (NL-to-code generation).

    Schema (neulab/conala):
        - intent: str (question/prompt)
        - snippet: str (answer/solution)
    
    Note: No test cases available. Uses generation evaluator (text similarity).
    """

    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from conala examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load dataset
        docs = self.load_dataset(
            dataset_name="neulab/conala",
            split="train",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} conala examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid conala pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            question = doc.get("intent", "").strip()
            answer = doc.get("snippet", "")

            if not question or not answer:
                log.debug("Skipping: missing question or answer")
                return None

            # Convert answer to string
            correct_answer = str(answer).strip()

            # Create incorrect answer (modify or corrupt)
            incorrect_answer = self._create_incorrect_answer(correct_answer)

            # Format the question
            formatted_question = f"Question: {question}\n\nWhat is the answer?"

            metadata = {
                "label": "conala",
                "source": "neulab/conala",
            }

            return self._build_pair(
                question=formatted_question,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _create_incorrect_answer(self, correct: str) -> str:
        """Create an incorrect answer by shuffling letters in words."""
        def shuffle_word(word: str) -> str:
            """Shuffle all letters in a word."""
            if len(word) <= 2:
                return word
            letters = list(word)
            random.shuffle(letters)
            # Make sure it's actually different
            shuffled = ''.join(letters)
            if shuffled == word:
                return word[::-1]  # Reverse if shuffle didn't change
            return shuffled

        # Find words (alphanumeric sequences) and shuffle their letters
        def replace_word(match: re.Match) -> str:
            word = match.group(0)
            return shuffle_word(word)

        # Shuffle words with 3+ characters
        result = re.sub(r'[A-Za-z]{3,}', replace_word, correct)

        # If nothing changed (all short words), append something
        if result == correct:
            result = correct + "_corrupted"

        return result

