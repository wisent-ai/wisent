from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger
import json

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["SimpleQAExtractor"]

log = setup_logger(__name__)


class SimpleQAExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for SimpleQA dataset - OpenAI's factuality benchmark.

    SimpleQA measures the ability of language models to answer short, fact-seeking questions.
    Responses are graded as "correct", "incorrect", or "not attempted".

    Schema (basicv8vc/SimpleQA):
        - problem: str (the factual question)
        - answer: str (the expected factual answer)
        - metadata: str (JSON with topic, answer_type, urls)
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from SimpleQA examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load dataset from HuggingFace
        docs = self.load_dataset(
            dataset_name="basicv8vc/SimpleQA",
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} SimpleQA examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid SimpleQA pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            question = doc.get("problem", "").strip()
            answer = doc.get("answer", "").strip()
            metadata_str = doc.get("metadata", "{}")

            if not question or not answer:
                log.debug("Skipping: missing question or answer")
                return None

            # Parse metadata if available
            try:
                metadata_parsed = json.loads(metadata_str) if metadata_str else {}
            except json.JSONDecodeError:
                metadata_parsed = {}

            topic = metadata_parsed.get("topic", "unknown")

            # Create incorrect answer by generating a plausible but wrong response
            incorrect_answer = self._create_incorrect_answer(answer, topic)

            # Format the question for the model
            formatted_question = f"Answer the following factual question concisely and accurately.\n\nQuestion: {question}\n\nAnswer:"

            metadata = {
                "label": "simpleqa",
                "source": "basicv8vc/SimpleQA",
                "topic": topic,
                "answer_type": metadata_parsed.get("answer_type", "factual"),
            }

            return self._build_pair(
                question=formatted_question,
                correct=answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _create_incorrect_answer(self, correct: str, topic: str) -> str:
        """Create a plausible but factually incorrect answer.

        Strategy: Generate answers that look plausible but are wrong.
        - For names: use similar-sounding or related names
        - For numbers: use different numbers
        - For dates: use different dates
        - For places: use related but wrong places
        """
        import random
        random.seed(hash(correct) % (2**32))

        # For numerical answers
        if correct.isdigit():
            num = int(correct)
            wrong_vals = [num * 2, num // 2 if num > 1 else num + 5, num + 10, num - 5]
            return str(random.choice([v for v in wrong_vals if v != num]))

        # For years (4 digit numbers)
        if len(correct) == 4 and correct.isdigit():
            year = int(correct)
            return str(random.choice([year - 10, year + 10, year - 5, year + 5]))

        # For short factual answers (names, places, etc.)
        # Scramble the characters to create a wrong but similar-looking answer
        if len(correct) < 100:
            words = correct.split()
            if len(words) >= 2:
                # Swap word order or modify
                scrambled = words.copy()
                random.shuffle(scrambled)
                if scrambled != words:
                    return ' '.join(scrambled)

            # Character-level scrambling for single words
            chars = list(correct)
            if len(chars) > 3:
                # Keep first and last, shuffle middle
                middle = chars[1:-1]
                random.shuffle(middle)
                return chars[0] + ''.join(middle) + chars[-1]

        # For longer answers, truncate and modify
        if len(correct) > 50:
            return correct[:len(correct)//2] + " [incomplete/incorrect]"

        # Fallback: return "Unknown" which is clearly wrong for factual questions
        return "Unknown"

