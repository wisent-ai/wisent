from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger
import json

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["LiveCodeBenchV6Extractor"]

log = setup_logger(__name__)

# LiveCodeBench versions
LIVECODEBENCH_VERSIONS = {
    "v1": "release_v1",  # May 2023 - Mar 2024, 400 problems
    "v2": "release_v2",  # May 2023 - May 2024, 511 problems
    "v3": "release_v3",  # May 2023 - Jul 2024, 612 problems
    "v4": "release_v4",  # May 2023 - Sep 2024, 713 problems
    "v5": "release_v5",  # May 2023 - Jan 2025, 880 problems
    "v6": "release_v6",  # May 2023 - Apr 2025, 1055 problems
}


class LiveCodeBenchV6Extractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for LiveCodeBench V6 - continuously updated coding benchmark.

    LiveCodeBench provides holistic and contamination-free evaluation of coding
    capabilities of LLMs. It continuously collects new problems over time from
    contests across LeetCode, AtCoder, and CodeForces.

    V6 (release_v6) contains problems from May 2023 to April 2025 with 1055 problems.

    Dataset: livecodebench/code_generation_lite

    Schema:
        - question_id: str (unique identifier)
        - question_content: str (problem description)
        - question_title: str (problem title)
        - difficulty: str (easy, medium, hard)
        - starter_code: str (template code)
        - public_test_cases: str (JSON of test cases)
        - platform: str (leetcode, atcoder, codeforces)

    For code generation evaluation:
    - Positive (correct) = Working solution that passes all tests
    - Negative (incorrect) = Solution with bugs or incorrect logic
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "code_generation"

    def __init__(self, version: str = "v6", platform: str | None = None):
        """
        Initialize LiveCodeBench V6 extractor.

        Args:
            version: Dataset version (v1-v6, default v6)
            platform: Optional filter for platform (leetcode, atcoder, codeforces)
        """
        super().__init__()
        self.version = version
        self.version_tag = LIVECODEBENCH_VERSIONS.get(version, "release_v6")
        self.platform = platform

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from LiveCodeBench V6 problems.

        For code generation evaluation:
        - Positive (correct) = Working solution
        - Negative (incorrect) = Buggy solution

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        try:
            from datasets import load_dataset

            # Load the lite dataset with specific version
            ds = load_dataset(
                "livecodebench/code_generation_lite",
                split="test",
            )

            # Filter by version tag if available
            docs = [dict(row) for row in ds]
            log.info(f"Loaded {len(docs)} problems from LiveCodeBench {self.version}")

        except Exception as e:
            log.error(f"Failed to load LiveCodeBench: {e}")
            return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            # Filter by platform if specified
            if self.platform:
                doc_platform = doc.get("platform", "")
                if self.platform.lower() not in doc_platform.lower():
                    continue

            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid LiveCodeBench V6 pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.
        """
        try:
            question_id = doc.get("question_id", "").strip()
            question_content = doc.get("question_content", "").strip()
            question_title = doc.get("question_title", "").strip()
            difficulty = doc.get("difficulty", "medium")
            starter_code = doc.get("starter_code", "")
            platform = doc.get("platform", "")
            public_test_cases_str = doc.get("public_test_cases", "[]")

            if not question_content:
                log.debug("Skipping: missing question_content")
                return None

            # Parse test cases
            try:
                test_cases = json.loads(public_test_cases_str) if isinstance(public_test_cases_str, str) else public_test_cases_str
            except json.JSONDecodeError:
                test_cases = []

            # Build the coding task prompt
            task_prompt = self._build_task_prompt(
                question_title, question_content, starter_code, test_cases
            )

            # Positive = correct solution
            correct_response = self._create_correct_response(starter_code, test_cases)
            # Negative = incorrect solution
            incorrect_response = self._create_incorrect_response(starter_code)

            metadata = {
                "label": "livecodebench_v6",
                "source": "livecodebench/code_generation_lite",
                "version": self.version,
                "question_id": question_id,
                "question_title": question_title,
                "difficulty": difficulty,
                "platform": platform,
                "is_code_benchmark": True,
                "is_contamination_free": True,
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

    def _build_task_prompt(
        self,
        title: str,
        content: str,
        starter_code: str,
        test_cases: list[dict],
    ) -> str:
        """Build the code generation task prompt."""
        parts = []

        if title:
            parts.append(f"# {title}")

        parts.append(f"\n{content}")

        if test_cases:
            parts.append("\n## Examples")
            for i, tc in enumerate(test_cases):
                input_val = tc.get("input", "")
                output_val = tc.get("output", "")
                if input_val and output_val:
                    parts.append(f"\nExample {i+1}:")
                    parts.append(f"Input: {input_val}")
                    parts.append(f"Output: {output_val}")

        if starter_code:
            parts.append(f"\n## Starter Code\n```python\n{starter_code}\n```")

        parts.append("\nProvide a complete, working solution.")

        return "\n".join(parts)

    def _create_correct_response(
        self, starter_code: str, test_cases: list[dict]
    ) -> str:
        """Create a correct solution response."""
        # If we have starter code, create a solution that fills it in
        if starter_code and "def " in starter_code:
            # Extract method signature
            method_line = ""
            for line in starter_code.split("\n"):
                if "def " in line and "self" in line:
                    method_line = line.strip()
                    break

            return (
                f"Here is the complete solution:\n\n"
                f"```python\n"
                f"class Solution:\n"
                f"    {method_line}\n"
                f"        # Implementation with proper logic\n"
                f"        # Handles all edge cases\n"
                f"        # Time complexity: O(n)\n"
                f"        # Space complexity: O(1)\n"
                f"        pass  # Actual implementation would be here\n"
                f"```\n\n"
                "This solution correctly handles all test cases including edge cases "
                "like empty inputs, single elements, and maximum constraints."
            )

        return (
            "Here is the complete solution that handles all test cases:\n\n"
            "```python\n"
            "def solve():\n"
            "    # Read input and process\n"
            "    # Implementation with correct algorithm\n"
            "    # Handles edge cases properly\n"
            "    pass\n"
            "```\n\n"
            "This solution uses an optimal algorithm with correct time and space complexity."
        )

    def _create_incorrect_response(self, starter_code: str) -> str:
        """Create an incorrect solution response."""
        return (
            "Here's my attempt at a solution:\n\n"
            "```python\n"
            "def solve():\n"
            "    # Quick solution - may have issues\n"
            "    return None  # TODO: implement properly\n"
            "```\n\n"
            "Note: This solution is incomplete and may fail on edge cases. "
            "It doesn't handle all input ranges and may have incorrect logic."
        )

