from __future__ import annotations

from typing import Any
import logging

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["HumanEvalExtractor"]

log = logging.getLogger(__name__)


class HumanEvalExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for HumanEval and HumanEval+ coding benchmarks.

    HumanEval schema (openai_humaneval):
        - task_id: str (e.g., "HumanEval/0")
        - prompt: str (function signature + docstring)
        - canonical_solution: str (correct implementation)
        - test: str (unit tests)
        - entry_point: str (function name)
    """

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from HumanEval examples.

        For coding tasks, we create pairs where:
        - Positive: Correct implementation
        - Negative: Incorrect implementation (generated or placeholder)

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load HumanEval dataset
        docs = self.load_dataset(
            dataset_name="openai_humaneval",
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} HumanEval examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid HumanEval pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single HumanEval doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            task_id = doc.get("task_id", "")
            prompt = doc.get("prompt", "").strip()
            canonical_solution = doc.get("canonical_solution", "").strip()
            entry_point = doc.get("entry_point", "")
            test_code = doc.get("test", "").strip()

            if not prompt or not canonical_solution:
                log.debug(f"Skipping {task_id}: missing prompt or solution")
                return None

            # Construct the full correct implementation
            # In HumanEval, prompt contains signature+docstring, canonical_solution is body at column 0
            # We need to indent the body by 4 spaces to be inside the function
            lines = canonical_solution.split('\n')
            indented_lines = ['    ' + line if line and not line[0].isspace() else line for line in lines]
            indented_solution = '\n'.join(indented_lines)
            correct_code = prompt + "\n" + indented_solution

            # Create an incorrect implementation (return incorrect value/type)
            # For coding benchmarks, we create a simple buggy version
            incorrect_code = prompt + "\n    pass  # Incorrect: empty implementation"

            # Format the question to include the task
            question = f"Complete the following Python function:\n\n{prompt}"

            metadata = {
                "label": "humaneval",
                "task_id": task_id,
                "entry_point": entry_point,
                "test_code": test_code,  # Include test code for execution
            }

            return self._build_pair(
                question=question,
                correct=correct_code,
                incorrect=incorrect_code,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

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
