from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["FRAMESExtractor"]

log = setup_logger(__name__)


class FRAMESExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for FRAMES benchmark - Factuality, Retrieval, And Multi-hop reasoning Evaluation.

    FRAMES is a benchmark for evaluating multi-hop reasoning and factual retrieval.
    It includes questions requiring:
    - Numerical reasoning
    - Tabular reasoning
    - Multiple constraints
    - Temporal reasoning
    - Post processing

    Schema (google/frames-benchmark):
        - Prompt: str (the multi-hop question)
        - Answer: str (gold answer text)
        - reasoning_types: str (category of reasoning required)
        - wiki_links: str (JSON array of Wikipedia reference links)
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from FRAMES examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load dataset from HuggingFace
        docs = self.load_dataset(
            dataset_name="google/frames-benchmark",
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} FRAMES examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid FRAMES pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            # FRAMES uses "Prompt" and "Answer" (capitalized)
            question = doc.get("Prompt", "").strip()
            answer = doc.get("Answer", "").strip()
            reasoning_types = doc.get("reasoning_types", "")

            if not question or not answer:
                log.debug("Skipping: missing question or answer")
                return None

            # Create incorrect answer based on reasoning type
            incorrect_answer = self._create_incorrect_answer(answer, reasoning_types)

            # Format the question for the model
            formatted_question = (
                f"Answer the following question that may require multi-step reasoning. "
                f"Provide a complete and accurate answer.\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )

            metadata = {
                "label": "frames",
                "source": "google/frames-benchmark",
                "reasoning_types": reasoning_types,
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

    def _create_incorrect_answer(self, correct: str, reasoning_types: str) -> str:
        """Create a plausible but incorrect answer based on reasoning type."""
        # For numerical reasoning, try to extract and modify numbers
        if "Numerical" in reasoning_types:
            import re
            numbers = re.findall(r'\d+\.?\d*', correct)
            if numbers:
                # Modify the first number found
                try:
                    num = float(numbers[0])
                    wrong_num = num * 1.5 if num > 0 else num - 10
                    return correct.replace(numbers[0], str(int(wrong_num)), 1)
                except ValueError:
                    pass

        # For temporal reasoning, create a temporally incorrect answer
        if "Temporal" in reasoning_types:
            return f"Based on the timeline, the answer would be different: {correct}... [temporally incorrect]"

        # For tabular reasoning
        if "Tabular" in reasoning_types:
            return f"According to the data, the result is not {correct} but rather a different value."

        # Default: Create a hedging/uncertain response
        return f"I believe the answer might be related to {correct}, but I'm not entirely certain."

