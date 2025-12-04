from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["CLUEWSCExtractor"]

log = setup_logger(__name__)


class CLUEWSCExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for CLUEWSC2020 - Chinese Winograd Schema Challenge.

    CLUEWSC is part of the CLUE (Chinese Language Understanding Evaluation) benchmark,
    focusing on coreference resolution in Chinese text. The task is to determine
    whether a pronoun (span2) refers to a specific entity (span1) in the given text.

    For coreference resolution:
    - Positive (correct) = Correct identification of coreference relationship
    - Negative (incorrect) = Wrong identification

    Schema (clue/clue - cluewsc2020):
        - idx: int (record identifier)
        - text: str (input sentence with target spans)
        - label: int (0=true, 1=false - whether spans are coreferent)
        - span1_text: str (first entity/reference text)
        - span2_text: str (second entity/reference text - usually pronoun)
        - span1_index: int (position of first span in text)
        - span2_index: int (position of second span in text)
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "coreference_resolution"

    def __init__(self, split: str = "validation"):
        """
        Initialize CLUEWSC extractor.

        Args:
            split: Dataset split to use ("train", "validation", "test")
        """
        super().__init__()
        self.split = split

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from CLUEWSC examples.

        For coreference resolution:
        - Positive (correct) = Correct coreference judgment
        - Negative (incorrect) = Wrong coreference judgment

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        try:
            docs = self.load_dataset(
                dataset_name="clue",
                config="cluewsc2020",
                split=self.split,
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from CLUEWSC ({self.split})")
        except Exception as e:
            log.error(f"Failed to load CLUEWSC: {e}")
            return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid CLUEWSC pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            idx = doc.get("idx", 0)
            text = doc.get("text", "").strip()
            label = doc.get("label", -1)  # 0 = true (coreferent), 1 = false
            span1_text = doc.get("span1_text", "")
            span2_text = doc.get("span2_text", "")
            span1_index = doc.get("span1_index", -1)
            span2_index = doc.get("span2_index", -1)

            if not text or not span1_text or not span2_text:
                log.debug("Skipping: missing text or spans")
                return None

            # Build the prompt
            prompt = self._build_prompt(text, span1_text, span2_text)

            # Determine correct answer based on label
            # label 0 = true (coreferent), label 1 = false (not coreferent)
            is_coreferent = (label == 0)

            # Build correct and incorrect responses
            correct_response = self._create_response(is_coreferent, span1_text, span2_text)
            incorrect_response = self._create_response(not is_coreferent, span1_text, span2_text)

            metadata = {
                "label": "cluewsc",
                "source": "clue/clue:cluewsc2020",
                "idx": idx,
                "split": self.split,
                "span1_text": span1_text,
                "span2_text": span2_text,
                "span1_index": span1_index,
                "span2_index": span2_index,
                "is_coreferent": is_coreferent,
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

    def _build_prompt(self, text: str, span1_text: str, span2_text: str) -> str:
        """Build the coreference resolution prompt."""
        return f"""请判断以下中文句子中，"{span2_text}"是否指代"{span1_text}"。

句子：{text}

问题：在上述句子中，"{span2_text}"是否指代"{span1_text}"？请回答"是"或"否"。

(Please determine whether "{span2_text}" refers to "{span1_text}" in the following Chinese sentence.

Sentence: {text}

Question: Does "{span2_text}" refer to "{span1_text}" in the above sentence? Please answer "Yes" or "No".)"""

    def _create_response(self, is_coreferent: bool, span1_text: str, span2_text: str) -> str:
        """Create a response for the coreference judgment."""
        if is_coreferent:
            return f"""是的，在这个句子中，"{span2_text}"指代"{span1_text}"。

(Yes, in this sentence, "{span2_text}" refers to "{span1_text}".)

答案：是 (Yes)"""
        else:
            return f"""不是，在这个句子中，"{span2_text}"不指代"{span1_text}"。

(No, in this sentence, "{span2_text}" does not refer to "{span1_text}".)

答案：否 (No)"""

