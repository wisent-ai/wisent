from __future__ import annotations

from typing import Any
from wisent.core.utils.cli.cli_logger import setup_logger

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["SWEBenchVerifiedExtractor", "MultiSWEBenchExtractor"]

log = setup_logger(__name__)

# Repositories in SWE-bench
SWE_BENCH_REPOS = [
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "psf/requests",
    "pytest-dev/pytest",
    "scikit-learn/scikit-learn",
    "sphinx-doc/sphinx",
    "sympy/sympy",
]

# Languages in Multi-SWE-bench
MULTI_SWE_BENCH_LANGUAGES = [
    "java",
    "typescript",
    "javascript",
    "go",
    "rust",
    "c",
    "cpp",
]


class SWEBenchVerifiedExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for SWE-bench Verified - human-validated software engineering tasks.

    SWE-bench Verified is a curated subset of 500 samples from SWE-bench that have
    been verified by human annotators to be solvable. The benchmark evaluates LLMs'
    ability to automatically solve real GitHub issues.

    Dataset: princeton-nlp/SWE-bench_Verified (or SWE-bench/SWE-bench_Verified)

    Schema:
        - repo: str (repository name)
        - instance_id: str (unique identifier)
        - base_commit: str (baseline commit hash)
        - patch: str (solution code changes)
        - test_patch: str (test modifications)
        - problem_statement: str (issue description)
        - hints_text: str (optional guidance)
        - difficulty: str (problem difficulty rating)
        - FAIL_TO_PASS: str (tests that should pass after fix)
        - PASS_TO_PASS: str (tests that should continue passing)

    For software engineering evaluation:
    - Positive (correct) = Patch that resolves the issue
    - Negative (incorrect) = Incomplete or incorrect patch
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "swe_bench"

    def __init__(self, difficulty: str | None = None):
        """
        Initialize SWE-bench Verified extractor.

        Args:
            difficulty: Optional filter for difficulty (e.g., "easy", "medium", "hard")
        """
        super().__init__()
        self.difficulty = difficulty

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from SWE-bench Verified examples.

        For software engineering evaluation:
        - Positive (correct) = Working patch
        - Negative (incorrect) = Incorrect patch

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        try:
            docs = self.load_dataset(
                dataset_name="princeton-nlp/SWE-bench_Verified",
                split="test",
                limit=max_items * 2 if max_items else None,
            )
            log.info(f"Loaded {len(docs)} examples from SWE-bench Verified")
        except Exception as e:
            log.warning(f"Failed to load princeton-nlp/SWE-bench_Verified: {e}")
            try:
                docs = self.load_dataset(
                    dataset_name="SWE-bench/SWE-bench_Verified",
                    split="test",
                    limit=max_items * 2 if max_items else None,
                )
                log.info(f"Loaded {len(docs)} examples from SWE-bench/SWE-bench_Verified")
            except Exception as e2:
                log.error(f"Failed to load SWE-bench Verified: {e2}")
                return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            # Filter by difficulty if specified
            if self.difficulty:
                doc_difficulty = doc.get("difficulty", "")
                if self.difficulty.lower() not in doc_difficulty.lower():
                    continue

            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid SWE-bench Verified pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.
        """
        try:
            repo = doc.get("repo", "").strip()
            instance_id = doc.get("instance_id", "").strip()
            problem_statement = doc.get("problem_statement", "").strip()
            patch = doc.get("patch", "").strip()
            hints_text = doc.get("hints_text", "")
            difficulty = doc.get("difficulty", "unknown")
            fail_to_pass = doc.get("FAIL_TO_PASS", "")
            pass_to_pass = doc.get("PASS_TO_PASS", "")

            if not problem_statement or not patch:
                log.debug("Skipping: missing problem_statement or patch")
                return None

            # Build the software engineering task prompt
            task_prompt = self._build_task_prompt(
                repo, problem_statement, hints_text, fail_to_pass
            )

            # Positive = correct patch
            correct_response = self._create_correct_response(patch, fail_to_pass)
            # Negative = incorrect/incomplete patch
            incorrect_response = self._create_incorrect_response(problem_statement)

            metadata = {
                "label": "swe_bench_verified",
                "source": "princeton-nlp/SWE-bench_Verified",
                "repo": repo,
                "instance_id": instance_id,
                "difficulty": difficulty,
                "is_software_engineering_benchmark": True,
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
        repo: str,
        problem_statement: str,
        hints_text: str,
        fail_to_pass: str,
    ) -> str:
        """Build the software engineering task prompt."""
        parts = [
            f"Repository: {repo}",
            f"\n## Issue Description\n{problem_statement}",
        ]

        if hints_text:
            parts.append(f"\n## Hints\n{hints_text}")

        if fail_to_pass:
            parts.append(f"\n## Tests to Pass\n{fail_to_pass}")

        parts.append(
            "\n## Task\nProvide a patch that resolves the issue described above. "
            "The patch should make the failing tests pass while not breaking existing tests."
        )

        return "\n".join(parts)

    def _create_correct_response(self, patch: str, fail_to_pass: str) -> str:
        """Create a response with the correct patch."""
        return (
            f"Here is the patch to resolve this issue:\n\n"
            f"```diff\n{patch}\n```\n\n"
            "This patch addresses the root cause of the issue and ensures that "
            "all specified tests pass while maintaining backward compatibility."
        )

    def _create_incorrect_response(self, problem_statement: str) -> str:
        """Create an incorrect/incomplete response."""
        return (
            "I attempted to analyze the issue but here's a partial solution:\n\n"
            "```diff\n"
            "- # Original code\n"
            "+ # TODO: Fix the issue\n"
            "+ pass  # Placeholder\n"
            "```\n\n"
            "Note: This patch is incomplete and may not fully resolve the issue. "
            "Additional investigation is needed to understand the root cause."
        )




# Re-export from split module
from wisent.extractors.hf.hf_task_extractors.swe_bench_multi import (
    MultiSWEBenchExtractor,
)
