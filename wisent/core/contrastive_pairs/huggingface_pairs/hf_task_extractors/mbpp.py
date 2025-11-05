from __future__ import annotations

from typing import Any
import logging

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["MBPPExtractor"]

log = logging.getLogger(__name__)


class MBPPExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for MBPP (Mostly Basic Python Problems) dataset.

    Schema (mbpp or google-research/mbpp):
        - text: str (problem description)
        - code: str (correct solution code)
        - test_list: list[str] (test cases)
    """

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from MBPP examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load dataset
        docs = self.load_dataset(
            dataset_name="mbpp",
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} MBPP examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid MBPP pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            problem_text = doc.get("text", "").strip()
            correct_code = doc.get("code", "").strip()
            test_list = doc.get("test_list", [])
            test_imports = doc.get("test_imports", [])

            if not problem_text or not correct_code:
                log.debug("Skipping: missing problem text or code")
                return None

            # Create incorrect code (add syntax error or logical error)
            incorrect_code = self._create_incorrect_code(correct_code)

            # Format the prompt
            formatted_prompt = f"{problem_text}\n\nWrite a Python function to solve this problem."

            # Build test code from test_list (similar to MBPP dataset format)
            # The test_list contains assertion strings like "assert function_name(...) == expected"
            test_code = ""
            if test_list:
                # Add imports if provided
                if test_imports:
                    test_code += "\n".join(test_imports) + "\n\n"

                # Add test function
                test_code += "def check(candidate):\n"
                for test_case in test_list:
                    # Replace function name in assertion with 'candidate'
                    # This assumes the test_list contains assertions like "assert func(...) == result"
                    test_code += f"    {test_case}\n"

            metadata = {
                "label": "mbpp",
                "source": "mbpp",
                "test_code": test_code if test_code else None,
            }

            return self._build_pair(
                question=formatted_prompt,
                correct=correct_code,
                incorrect=incorrect_code,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _create_incorrect_code(self, correct: str) -> str:
        """Create an incorrect version of the code."""
        # Add a syntax error by removing closing parenthesis
        if "(" in correct and ")" in correct:
            # Find last closing paren and remove it
            idx = correct.rfind(")")
            if idx > 0:
                return correct[:idx] + correct[idx+1:]

        # Fallback: add comment that breaks the code
        return correct + "\n# Missing return statement"

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        """Build a ContrastivePair from question and responses."""
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None,
            metadata=metadata,
        )
