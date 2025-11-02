from __future__ import annotations

from typing import Any
import logging

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["LivecodebenchExtractor"]

log = logging.getLogger(__name__)


class LivecodebenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for livecodebench dataset.

    Schema (livecodebench/code_generation_lite):
        - question_content: str (question/prompt)
        - public_test_cases: list (test cases)
        - private_test_cases: list (test cases)
    """

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from livecodebench examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load dataset - using lite version since main version requires authentication
        docs = self.load_dataset(
            dataset_name="livecodebench/code_generation_lite",
            config="release_latest",
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} livecodebench examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid livecodebench pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            question = doc.get("question_content", "").strip()
            # Use test cases as the answer
            public_tests = doc.get("public_test_cases", [])
            private_tests = doc.get("private_test_cases", [])

            if not question:
                log.debug("Skipping: missing question")
                return None

            # Format test cases as answer
            test_summary = f"Public tests: {len(public_tests)}, Private tests: {len(private_tests)}"
            if public_tests:
                test_summary += f"\nExample test: {public_tests[0]}"

            correct_answer = test_summary

            # Create incorrect answer (modify test count)
            incorrect_answer = self._create_incorrect_answer(correct_answer)

            # Format the question
            formatted_question = f"Question: {question}\n\nWhat is the answer?"

            metadata = {
                "label": "livecodebench",
                "source": "livecodebench/code_generation_lite",
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
        """Create an incorrect answer by modifying the correct one."""
        # For code, corrupt it slightly
        if len(correct) > 10:
            return correct[:len(correct)//2] + "# CORRUPTED" + correct[len(correct)//2:]
        return f"{correct} # INCORRECT"

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
