from __future__ import annotations

from typing import Any
from wisent.core.utils.cli.cli_logger import setup_logger

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.config_tools.constants import (
    FRAMES_ALT_ANSWER_LENGTH,
    FRAMES_NUMERIC_DELTA_LARGE,
    FRAMES_NUMERIC_DELTA_MEDIUM,
    FRAMES_NUMERIC_DELTA_SMALL,
    FRAMES_NUMERIC_DELTA_TINY,
    FRAMES_SHORT_ANSWER_THRESHOLD,
    YEAR_PERTURB_LARGE,
    YEAR_PERTURB_SMALL,
)

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
        """Create a plausible but factually incorrect answer based on reasoning type."""
        import re
        import random
        random.seed(hash(correct) % (2**32))

        # For numerical reasoning, modify numbers in a meaningful way
        if "Numerical" in reasoning_types:
            numbers = re.findall(r'\d+\.?\d*', correct)
            if numbers:
                num = float(numbers[0])
                wrong_vals = [num * 2, num / 2, num + FRAMES_NUMERIC_DELTA_LARGE, num - FRAMES_NUMERIC_DELTA_MEDIUM]
                wrong_num = random.choice([v for v in wrong_vals if v != num])
                return correct.replace(numbers[0], str(int(wrong_num)), 1)

        # For temporal reasoning, shift dates/years
        if "Temporal" in reasoning_types:
            years = re.findall(r'\b(19|20)\d{2}\b', correct)
            if years:
                year = int(years[0])
                wrong_year = random.choice([year - YEAR_PERTURB_LARGE, year + YEAR_PERTURB_LARGE, year - YEAR_PERTURB_SMALL, year + YEAR_PERTURB_SMALL])
                return correct.replace(str(year), str(wrong_year), 1)

        # For any answer with numbers, modify them
        numbers = re.findall(r'\d+', correct)
        if numbers:
            num = int(numbers[0])
            wrong_num = random.choice([num * 2, num + FRAMES_NUMERIC_DELTA_SMALL, num - FRAMES_NUMERIC_DELTA_TINY]) if num != 0 else FRAMES_NUMERIC_DELTA_TINY
            return correct.replace(numbers[0], str(wrong_num), 1)

        # For name-based answers, scramble or use different format
        if len(correct) < FRAMES_SHORT_ANSWER_THRESHOLD:
            words = correct.split()
            if len(words) >= 2:
                scrambled = words.copy()
                random.shuffle(scrambled)
                if scrambled != words:
                    return ' '.join(scrambled)

        # Fallback: clearly wrong answer
        return "Unable to determine" if len(correct) > FRAMES_ALT_ANSWER_LENGTH else correct[::-1]

