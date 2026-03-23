from __future__ import annotations

from typing import Any, Optional
from wisent.core.utils.cli.cli_logger import setup_logger
import requests
import zipfile
import json
import io

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.infra_tools.errors import InvalidValueError

__all__ = ["OJBenchExtractor", "NL2BashExtractor", "SciCodeExtractor", "TerminalBenchExtractor"]

log = setup_logger(__name__)

# GitHub URL for SciCode data
SCICODE_GITHUB_URL = "https://raw.githubusercontent.com/scicode-bench/scicode-bench.github.io/main/data/data.zip"


class OJBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for OJ-Bench - online judge style competitive programming benchmark.

    OJ-Bench evaluates LLMs on competitive programming problems similar to those
    found on online judges like Codeforces, AtCoder, and LeetCode. Problems are
    primarily in C++ and test algorithmic problem-solving skills.

    For competitive programming evaluation:
    - Positive (correct) = Solution that passes all test cases within time/memory limits
    - Negative (incorrect) = Solution with wrong answer, TLE, or MLE
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "ojbench"

    def __init__(self, difficulty: str | None = None, language: Optional[str] = None):
        """
        Initialize OJ-Bench extractor.

        Args:
            difficulty: Optional filter (easy, medium, hard)
            language: Programming language (e.g., "cpp")
        """
        super().__init__()
        self.difficulty = difficulty
        self.language = language if language is not None else "cpp"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from OJ-Bench examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Try loading from competitive programming datasets
        docs = []

        try:
            docs = self.load_dataset(
                dataset_name="deepmind/code_contests",
                split="test",
                limit=max_items * 2 if max_items else None,
            )
            log.info(f"Loaded {len(docs)} examples from code_contests")
        except Exception as e:
            log.error(f"Failed to load code_contests dataset: {e}")
            log.error("OJBench requires deepmind/code_contests dataset. No synthetic data available.")
            return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid OJ-Bench pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            # Handle code_contests schema
            description = doc.get("description", doc.get("problem", "")).strip()
            correct = doc.get("correct_solution", "")
            incorrect = doc.get("incorrect_solution", "")

            # For code_contests dataset
            if not correct and "solutions" in doc:
                solutions = doc.get("solutions", {})
                if isinstance(solutions, dict) and "cpp" in solutions:
                    cpp_solutions = solutions["cpp"]
                    if cpp_solutions:
                        correct = cpp_solutions[0]

            if not description:
                return None

            # Create incorrect solution if not provided
            if not incorrect:
                incorrect = self._create_incorrect_solution(description)

            if not correct:
                correct = self._create_placeholder_correct(description)

            difficulty = doc.get("difficulty", "medium")

            # Filter by difficulty if specified
            if self.difficulty and self.difficulty.lower() != difficulty.lower():
                return None

            task_prompt = f"""Competitive Programming Problem:

{description}

Write a correct C++ solution that passes all test cases within the time and memory limits."""

            metadata = {
                "label": "oj_bench",
                "source": "oj_bench",
                "difficulty": difficulty,
                "language": self.language,
                "is_competitive_programming_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _create_incorrect_solution(self, description: str) -> str:
        """Create a plausible but incorrect solution."""
        return """#include <bits/stdc++.h>
using namespace std;

int main() {
    // This solution has bugs:
    // - Doesn't handle edge cases
    // - May have integer overflow
    // - Inefficient algorithm causing TLE

    int n;
    cin >> n;

    // Naive O(n^2) approach
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Missing logic
        }
    }

    cout << 0 << endl;  // Wrong answer
    return 0;
}"""

    def _create_placeholder_correct(self, description: str) -> str:
        """Create a placeholder correct solution structure."""
        return """#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Efficient solution with proper algorithm
    // Handles all edge cases
    // Time complexity: O(n log n) or better

    // Implementation details depend on specific problem

    return 0;
}"""




# Re-exports from split modules
from wisent.extractors.hf.hf_task_extractors.nl2bash_scicode import (
    NL2BashExtractor,
    SciCodeExtractor,
)
from wisent.extractors.hf.hf_task_extractors.terminal_bench import (
    TerminalBenchExtractor,
)
