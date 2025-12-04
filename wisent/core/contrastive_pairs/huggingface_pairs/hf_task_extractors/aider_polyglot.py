from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["AiderPolyglotExtractor"]

log = setup_logger(__name__)

# Languages supported by Aider Polyglot benchmark
AIDER_POLYGLOT_LANGUAGES = [
    "python",
    "javascript",
    "java",
    "cpp",
    "go",
    "rust",
]


class AiderPolyglotExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Aider Polyglot-style code editing benchmarks.

    Aider's polyglot benchmark tests LLMs on 225 challenging Exercism coding
    exercises across C++, Go, Java, JavaScript, Python, and Rust. This extractor
    uses the jinaai/code_exercises dataset which provides similar code exercise
    problems in Python.

    The benchmark evaluates:
    - Code generation from docstrings
    - Code editing and completion
    - Multi-turn correction (fixing failed attempts)

    For code editing:
    - Positive (correct) = Working solution that passes tests
    - Negative (incorrect) = Buggy or incomplete solution

    Schema (jinaai/code_exercises):
        - problem: str (function signature with docstring)
        - solution: str (complete solution implementation)

    Note: The original Aider Polyglot benchmark is hosted on GitHub at
    github.com/Aider-AI/polyglot-benchmark. This extractor uses HuggingFace
    alternatives with similar structure.
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "code_editing"

    def __init__(self, language: str = "python"):
        """
        Initialize Aider Polyglot extractor.

        Args:
            language: Target programming language (currently python supported)
        """
        super().__init__()
        self.language = language

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from code exercise examples.

        For code editing:
        - Positive (correct) = Working solution
        - Negative (incorrect) = Buggy or incomplete solution

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Try primary dataset
        try:
            docs = self.load_dataset(
                dataset_name="jinaai/code_exercises",
                split="train",
                limit=max_items,
            )
            dataset_source = "jinaai/code_exercises"
            log.info(f"Loaded {len(docs)} examples from {dataset_source}")
        except Exception as e:
            log.warning(f"Failed to load jinaai/code_exercises: {e}")
            # Try alternative dataset
            try:
                docs = self.load_dataset(
                    dataset_name="synapse-alpha/coding_exercises",
                    split="train",
                    limit=max_items,
                )
                dataset_source = "synapse-alpha/coding_exercises"
                log.info(f"Loaded {len(docs)} examples from {dataset_source}")
            except Exception as e2:
                log.error(f"Failed to load any code exercises dataset: {e2}")
                return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc, dataset_source)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid Aider Polyglot pairs extracted")

        return pairs

    def _extract_pair_from_doc(
        self,
        doc: dict[str, Any],
        source: str,
    ) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            problem = doc.get("problem", "").strip()
            solution = doc.get("solution", "").strip()

            if not problem or not solution:
                log.debug("Skipping: missing problem or solution")
                return None

            # Build the prompt
            prompt = self._build_prompt(problem)

            # Correct response is the working solution
            correct_response = self._create_correct_response(solution)

            # Incorrect response is a buggy version
            incorrect_response = self._create_incorrect_response(problem, solution)

            metadata = {
                "label": "aider_polyglot",
                "source": source,
                "language": self.language,
                "is_code_benchmark": True,
                "is_code_editing_benchmark": True,
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

    def _build_prompt(self, problem: str) -> str:
        """Build the code editing prompt."""
        return f"""Complete the following Python function based on its docstring.

{problem}

Please provide the complete implementation."""

    def _create_correct_response(self, solution: str) -> str:
        """Create the correct response with working solution."""
        return f"""Here is the complete implementation:

```python
{solution}
```

This solution correctly implements the function according to the docstring specification."""

    def _create_incorrect_response(self, problem: str, solution: str) -> str:
        """Create an incorrect response with common bugs."""
        # Extract function name from problem if possible
        func_name = "the function"
        if "def " in problem:
            try:
                func_part = problem.split("def ")[1]
                func_name = func_part.split("(")[0]
            except (IndexError, AttributeError):
                pass

        # Create a buggy version by introducing common errors
        buggy_solution = self._introduce_bugs(solution)

        return f"""Here is my implementation:

```python
{buggy_solution}
```

Note: This implementation may have issues:
- Missing edge case handling
- Potential off-by-one errors
- Incomplete logic"""

    def _introduce_bugs(self, solution: str) -> str:
        """Introduce common bugs into a solution."""
        lines = solution.split("\n")

        if len(lines) > 3:
            # Remove a line to create incomplete logic
            middle_idx = len(lines) // 2
            buggy_lines = lines[:middle_idx] + ["    pass  # TODO: complete implementation"] + lines[middle_idx+2:]
            return "\n".join(buggy_lines)
        elif lines:
            # For short solutions, replace with pass
            first_line = lines[0] if lines else "def func():"
            return f"{first_line}\n    pass  # Implementation incomplete"
        else:
            return "pass  # No implementation"

