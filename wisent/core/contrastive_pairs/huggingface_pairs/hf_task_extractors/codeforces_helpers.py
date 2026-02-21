"""Codeforces extractor helper methods."""
from __future__ import annotations

from typing import Any


class CodeforcesHelperMixin:
    """Mixin providing prompt building and response generation for Codeforces."""

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

