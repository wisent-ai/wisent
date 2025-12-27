from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["LongformWritingExtractor"]

log = setup_logger(__name__)

# Writing task categories
WRITING_CATEGORIES = [
    "creative",
    "technical",
    "analytical",
    "persuasive",
    "narrative",
    "expository",
]


class LongformWritingExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Longform Writing evaluation benchmarks.

    Longform Writing benchmarks evaluate LLMs' ability to generate high-quality
    extended text across various writing tasks. This includes creative writing,
    technical documentation, analytical essays, and more.

    Uses datasets like:
    - akoksal/LongForm (or similar longform writing datasets)
    - Writing prompts and their high-quality completions

    For longform writing evaluation:
    - Positive (correct) = Well-structured, coherent, high-quality writing
    - Negative (incorrect) = Poorly structured, incoherent, or low-quality writing
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "longform_writing_quality"

    def __init__(self, category: str | None = None, min_length: int = 500):
        """
        Initialize Longform Writing extractor.

        Args:
            category: Optional filter for writing category
            min_length: Minimum length of output text (default 500 chars)
        """
        super().__init__()
        self.category = category
        self.min_length = min_length

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from longform writing examples.

        For writing evaluation:
        - Positive (correct) = High-quality, well-structured writing
        - Negative (incorrect) = Low-quality, poorly structured writing

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Try loading from various longform writing datasets
        docs = []

        try:
            docs = self.load_dataset(
                dataset_name="akoksal/LongForm",
                split="train",
                limit=max_items * 2 if max_items else None,
            )
            log.info(f"Loaded {len(docs)} examples from akoksal/LongForm")
        except Exception as e:
            log.error(f"Failed to load akoksal/LongForm: {e}")
            return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid Longform Writing pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.
        """
        try:
            # Handle different schema formats
            prompt = doc.get("input", doc.get("prompt", doc.get("instruction", ""))).strip()
            output = doc.get("output", doc.get("text", doc.get("response", ""))).strip()
            category = doc.get("category", doc.get("task_type", "general"))

            if not prompt:
                log.debug("Skipping: missing prompt")
                return None

            if not output or len(output) < self.min_length:
                log.debug("Skipping: output too short or missing")
                return None

            # Filter by category if specified
            if self.category and self.category.lower() not in category.lower():
                return None

            # Build the writing task prompt
            task_prompt = self._build_task_prompt(prompt)

            # Positive = high-quality output
            correct_response = self._create_high_quality_response(output)
            # Negative = low-quality output
            incorrect_response = self._create_low_quality_response(prompt)

            metadata = {
                "label": "longform_writing",
                "source": "longform_writing",
                "category": category,
                "output_length": len(output),
                "is_longform_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _build_task_prompt(self, prompt: str) -> str:
        """Build the longform writing task prompt."""
        return f"""{prompt}

Please provide a detailed, well-structured response. Your writing should be:
- Coherent and logically organized
- Well-developed with supporting details
- Clear and engaging to read"""

    def _create_high_quality_response(self, output: str) -> str:
        """Create a high-quality writing response."""
        return output

    def _create_low_quality_response(self, prompt: str) -> str:
        """Create a low-quality writing response."""
        return (
            f"Here is my response about {prompt}...\n\n"
            "This is a topic that many people talk about. There are many things "
            "to say about it. Some people think one thing, other people think "
            "another thing. It's complicated.\n\n"
            "In conclusion, this is an important topic. There are pros and cons. "
            "The end."
        )

