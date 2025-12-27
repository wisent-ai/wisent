from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["CodeforcesExtractor"]

log = setup_logger(__name__)


class CodeforcesExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Codeforces - Competitive Programming Benchmark.

    Based on open-r1/codeforces dataset containing 10k+ competitive programming
    problems from CodeForces with verified test cases and solutions.

    Dataset Configurations:
    - default: ~10k problems
    - verifiable: 8,760 executable problems with complete/generated tests
    - verifiable-prompts: Same with 2 generation prompts per problem

    For code generation:
    - Positive (correct) = Working solution that passes test cases
    - Negative (incorrect) = Solution with bugs or wrong algorithm

    Schema (open-r1/codeforces):
        - id: str (unique problem identifier)
        - title: str (problem title)
        - description: str (problem statement)
        - input_format: str (input description)
        - output_format: str (output description)
        - examples: list[dict] (example input/output pairs)
        - official_tests: list[dict] (test cases)
        - rating: int (problem difficulty rating)
        - tags: list[str] (algorithm tags)
        - time_limit: float (seconds)
        - memory_limit: float (MB)
        - editorial: str (solution explanation)
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "code_generation"

    def __init__(
        self,
        config: str = "verifiable",
        max_rating: int | None = None,
        min_rating: int | None = None,
        language: str = "python",
    ):
        """
        Initialize Codeforces extractor.

        Args:
            config: Dataset configuration ("default", "verifiable", "verifiable-prompts")
            max_rating: Filter problems by maximum difficulty rating
            min_rating: Filter problems by minimum difficulty rating
            language: Target programming language
        """
        super().__init__()
        self.config = config
        self.max_rating = max_rating
        self.min_rating = min_rating
        self.language = language

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Codeforces examples.

        For competitive programming:
        - Positive (correct) = Working solution approach
        - Negative (incorrect) = Wrong approach or buggy solution

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        try:
            docs = self.load_dataset(
                dataset_name="open-r1/codeforces",
                config=self.config,
                split="train",
                limit=max_items * 2 if max_items else None,  # Load extra for filtering
            )
            log.info(f"Loaded {len(docs)} problems from Codeforces ({self.config})")
        except Exception as e:
            log.error(f"Failed to load open-r1/codeforces: {e}")
            return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            # Filter by rating if specified
            rating = doc.get("rating", 0)
            if self.max_rating and rating > self.max_rating:
                continue
            if self.min_rating and rating < self.min_rating:
                continue

            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid Codeforces pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            problem_id = doc.get("id", "")
            title = doc.get("title", "")
            description = doc.get("description", "")
            input_format = doc.get("input_format", "")
            output_format = doc.get("output_format", "")
            examples = doc.get("examples", [])
            rating = doc.get("rating", 0)
            tags = doc.get("tags", [])
            time_limit = doc.get("time_limit", 1.0)
            memory_limit = doc.get("memory_limit", 256.0)
            editorial = doc.get("editorial", "")
            note = doc.get("note", "")

            if not description:
                log.debug("Skipping: missing description")
                return None

            # Build the problem prompt
            prompt = self._build_prompt(
                title=title,
                description=description,
                input_format=input_format,
                output_format=output_format,
                examples=examples,
                note=note,
                time_limit=time_limit,
                memory_limit=memory_limit,
            )

            # Build correct response (with proper approach)
            correct_response = self._create_correct_response(
                editorial=editorial,
                tags=tags,
                examples=examples,
            )

            # Build incorrect response (wrong approach)
            incorrect_response = self._create_incorrect_response(tags)

            metadata = {
                "label": "codeforces",
                "source": f"open-r1/codeforces:{self.config}",
                "problem_id": problem_id,
                "title": title,
                "rating": rating,
                "tags": tags if isinstance(tags, list) else [tags],
                "time_limit": time_limit,
                "memory_limit": memory_limit,
                "has_editorial": bool(editorial),
                "is_code_benchmark": True,
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

    def _build_prompt(
        self,
        title: str,
        description: str,
        input_format: str,
        output_format: str,
        examples: list,
        note: str,
        time_limit: float,
        memory_limit: float,
    ) -> str:
        """Build the problem prompt."""
        parts = []

        if title:
            parts.append(f"# {title}")
            parts.append("")

        parts.append("## Problem Statement")
        parts.append(description)
        parts.append("")

        if input_format:
            parts.append("## Input Format")
            parts.append(input_format)
            parts.append("")

        if output_format:
            parts.append("## Output Format")
            parts.append(output_format)
            parts.append("")

        if examples:
            parts.append("## Examples")
            for i, ex in enumerate(examples, 1):
                inp = ex.get("input", "")
                out = ex.get("output", "")
                parts.append(f"### Example {i}")
                parts.append(f"Input:\n```\n{inp}\n```")
                parts.append(f"Output:\n```\n{out}\n```")
                parts.append("")

        if note:
            parts.append("## Note")
            parts.append(note)
            parts.append("")

        parts.append(f"Time Limit: {time_limit}s | Memory Limit: {memory_limit}MB")
        parts.append("")
        parts.append(f"Write a solution in {self.language}.")

        return "\n".join(parts)

    def _create_correct_response(
        self,
        editorial: str,
        tags: list,
        examples: list,
    ) -> str:
        """Create a correct response with proper approach."""
        parts = []

        # Add approach based on tags
        if tags:
            tag_list = ", ".join(tags) if isinstance(tags, list) else str(tags)
            parts.append(f"## Approach")
            parts.append(f"This problem involves: {tag_list}")
            parts.append("")

        # Add editorial if available
        if editorial:
            parts.append("## Solution Explanation")
            parts.append(editorial)
            parts.append("")

        # Add solution structure
        parts.append("## Solution")
        parts.append(f"```{self.language}")

        if self.language == "python":
            parts.append(self._generate_python_template(tags))
        else:
            parts.append(self._generate_cpp_template(tags))

        parts.append("```")

        return "\n".join(parts)

    def _generate_python_template(self, tags: list) -> str:
        """Generate a Python solution template based on tags."""
        tag_str = " ".join(tags) if isinstance(tags, list) else ""

        if "dp" in tag_str or "dynamic programming" in tag_str:
            return """def solve():
    n = int(input())
    # Initialize DP array
    dp = [0] * (n + 1)
    # Base case
    dp[0] = 1
    # Fill DP table
    for i in range(1, n + 1):
        # State transition
        dp[i] = ...  # Fill based on problem logic
    print(dp[n])

solve()"""
        elif "graph" in tag_str or "bfs" in tag_str or "dfs" in tag_str:
            return """from collections import deque

def solve():
    n, m = map(int, input().split())
    graph = [[] for _ in range(n + 1)]
    for _ in range(m):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)

    # BFS/DFS traversal
    visited = [False] * (n + 1)
    # ... solution logic

solve()"""
        else:
            return """def solve():
    # Read input
    n = int(input())
    arr = list(map(int, input().split()))

    # Process and compute answer
    result = 0
    # ... solution logic

    print(result)

solve()"""

    def _generate_cpp_template(self, tags: list) -> str:
        """Generate a C++ solution template."""
        return """#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    cin >> n;

    // Solution logic here

    return 0;
}"""

    def _create_incorrect_response(self, tags: list) -> str:
        """Create an incorrect response with wrong approach."""
        return f"""## Approach
Let me try a brute force approach without considering the constraints.

## Solution
```{self.language}
# WARNING: This solution is likely incorrect or will TLE

def solve():
    n = int(input())
    # Naive O(n^2) or worse approach
    result = 0
    for i in range(n):
        for j in range(n):
            # This approach doesn't use the optimal algorithm
            pass
    print(result)

solve()
```

Note: This solution does not use the optimal approach and may fail on large inputs or edge cases."""

