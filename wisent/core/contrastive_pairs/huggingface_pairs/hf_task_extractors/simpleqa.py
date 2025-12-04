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
        """Create a plausible but incorrect answer.

        For factuality benchmarks, we create answers that are superficially similar
        but factually wrong, or we use refusal/hedging patterns.
        """
        # Strategy 1: Add uncertainty/hedging (models sometimes do this)
        hedging_patterns = [
            f"I'm not entirely sure, but I believe it might be {correct} or something similar.",
            f"I don't have enough information to answer this definitively.",
            f"I cannot provide a reliable answer to this question.",
        ]

        # Strategy 2: Modify the answer slightly to make it wrong
        if correct.isdigit():
            # For numerical answers, change the number
            try:
                num = int(correct)
                wrong_num = num + 1 if num > 0 else num - 1
                return str(wrong_num)
            except ValueError:
                pass

        # Strategy 3: For short factual answers, create a contradicting one
        if len(correct) < 50:
            return f"I believe the answer is not {correct}, but rather something else entirely."

        # Default: Use hedging
        return hedging_patterns[0]

