from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["ChineseSimpleQAExtractor"]

log = setup_logger(__name__)

# Chinese SimpleQA primary categories
CHINESE_SIMPLEQA_CATEGORIES = [
    "中华文化",  # Chinese Culture
    "人文学科",  # Humanities
    "工程、技术与应用科学",  # Engineering, Technology, and Applied Sciences
    "生活、艺术与文化",  # Life, Art, and Culture
    "社会",  # Society
    "自然科学",  # Natural Science
]


class ChineseSimpleQAExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Chinese-SimpleQA - Chinese Factuality Evaluation Benchmark.

    Chinese SimpleQA is the first comprehensive Chinese benchmark to evaluate
    the factuality ability of language models to answer short questions.
    It contains 3,000 high-quality questions spanning 6 major topics with
    99 fine-grained subtopics.

    Categories:
    - 中华文化 (Chinese Culture)
    - 人文学科 (Humanities)
    - 工程、技术与应用科学 (Engineering, Technology, and Applied Sciences)
    - 生活、艺术与文化 (Life, Art, and Culture)
    - 社会 (Society)
    - 自然科学 (Natural Science)

    For factuality evaluation:
    - Positive (correct) = Factually accurate answer
    - Negative (incorrect) = Factually wrong answer

    Schema (OpenStellarTeam/Chinese-SimpleQA):
        - id: str (unique identifier)
        - primary_category: str (one of 6 major topics)
        - secondary_category: str (one of 99 subtopics)
        - question: str (question in Chinese)
        - answer: str (expected answer)
        - urls: list[str] (reference URLs)
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "factuality"

    def __init__(self, category: str | None = None):
        """
        Initialize Chinese SimpleQA extractor.

        Args:
            category: Optional filter for specific primary category
        """
        super().__init__()
        self.category = category

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Chinese SimpleQA examples.

        For factuality:
        - Positive (correct) = Factually accurate answer
        - Negative (incorrect) = Factually wrong answer

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        try:
            docs = self.load_dataset(
                dataset_name="OpenStellarTeam/Chinese-SimpleQA",
                split="train",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from Chinese-SimpleQA")
        except Exception as e:
            log.error(f"Failed to load Chinese-SimpleQA: {e}")
            return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            # Filter by category if specified
            if self.category:
                primary_cat = doc.get("primary_category", "")
                if self.category not in primary_cat:
                    continue

            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid Chinese-SimpleQA pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            doc_id = doc.get("id", "")
            question = doc.get("question", "").strip()
            answer = doc.get("answer", "").strip()
            primary_category = doc.get("primary_category", "")
            secondary_category = doc.get("secondary_category", "")
            urls = doc.get("urls", [])

            if not question or not answer:
                log.debug("Skipping: missing question or answer")
                return None

            # Build prompt
            prompt = self._build_prompt(question)

            # Correct response is the factual answer
            correct_response = self._create_correct_response(answer)

            # Incorrect response is a plausible but wrong answer
            incorrect_response = self._create_incorrect_response(question, answer, secondary_category)

            metadata = {
                "label": "chinese_simpleqa",
                "source": "OpenStellarTeam/Chinese-SimpleQA",
                "id": doc_id,
                "primary_category": primary_category,
                "secondary_category": secondary_category,
                "reference_urls": urls if isinstance(urls, list) else [urls],
                "is_factuality_benchmark": True,
                "is_chinese_benchmark": True,
                "language": "zh",
            }

            return self._build_pair(
                question=prompt,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _build_prompt(self, question: str) -> str:
        """Build the factuality prompt."""
        return f"""请回答以下问题，给出简短准确的答案。

问题：{question}

(Please answer the following question with a short and accurate answer.

Question: {question})"""

    def _create_correct_response(self, answer: str) -> str:
        """Create the correct response with the factual answer."""
        return f"""答案：{answer}

(Answer: {answer})"""

    def _create_incorrect_response(
        self,
        question: str,
        correct_answer: str,
        category: str,
    ) -> str:
        """Create a plausible but incorrect response."""
        # Generate a plausible-sounding but wrong answer
        # Based on question type and category
        category_lower = category.lower() if category else ""

        if "历史" in category_lower or "history" in category_lower:
            wrong = "不详/未知"  # Unknown
        elif "科学" in category_lower or "science" in category_lower:
            wrong = "无法确定"  # Cannot be determined
        elif "文化" in category_lower or "culture" in category_lower:
            wrong = "民间传说中无定论"  # No consensus in folklore
        else:
            # Generic wrong answer
            wrong = "未知"

        return f"""答案：{wrong}

(Answer: {wrong})

注：此答案不准确。(Note: This answer is inaccurate.)"""

