from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger

__all__ = ["HumanevalpackExtractor"]
log = setup_logger(__name__)

task_names = ("humanevalpack",)


class HumanevalpackExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for HumanEvalPack multilingual coding benchmark.

    HumanEvalPack schema (bigcode/humanevalpack):
        - task_id: str (e.g., "Python/0")
        - prompt: str (function signature + docstring)
        - canonical_solution: str (correct implementation)
        - test: str (unit tests)
        - entry_point: str (function name)
        - language: str (Python, JavaScript, etc.)
    """

    evaluator_name = "coding"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from HumanEvalPack examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        docs = self.load_dataset(
            dataset_name="bigcode/humanevalpack",
            dataset_config="python",
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []
        log.info(f"Extracting contrastive pairs from {len(docs)} HumanEvalPack examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid HumanEvalPack pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single HumanEvalPack doc into a ContrastivePair."""
        try:
            task_id = doc.get("task_id", "")
            prompt = doc.get("prompt", "")
            canonical_solution = doc.get("canonical_solution", "")  # Don't strip - preserves indentation
            entry_point = doc.get("entry_point", "")
            test_code = doc.get("test", "").strip()

            if not prompt or not canonical_solution:
                return None

            # Prompt ends with \n, canonical_solution already has proper indentation
            correct_code = prompt + canonical_solution
            incorrect_code = prompt + "    pass  # Incorrect: empty implementation\n"

            question = f"Complete the following function:\n\n{prompt}"

            metadata = {
                "label": "humanevalpack",
                "task_id": task_id,
                "entry_point": entry_point,
                "test_code": test_code,
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

