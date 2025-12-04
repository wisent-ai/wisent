from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

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
    evaluator_name = "software_engineering"

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



class MultiSWEBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Multi-SWE-bench - multilingual software engineering benchmark.

    Multi-SWE-bench extends SWE-bench to cover 7 programming languages:
    Java, TypeScript, JavaScript, Go, Rust, C, and C++. Contains 1,632
    high-quality instances curated by 68 expert annotators.

    Dataset: ByteDance-Seed/Multi-SWE-bench

    Schema:
        - org: str (GitHub organization)
        - repo: str (repository name)
        - number: int (PR number)
        - title: str (PR title)
        - body: str (PR description)
        - fix_patch: str (code fix)
        - test_patch: str (test modifications)
        - instance_id: str (unique identifier)
        - run_result: dict (test execution results)

    For multilingual software engineering evaluation:
    - Positive (correct) = Patch that resolves the issue across languages
    - Negative (incorrect) = Incomplete or language-inappropriate patch
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "software_engineering_multilingual"

    def __init__(self, language: str | None = None):
        """
        Initialize Multi-SWE-bench extractor.

        Args:
            language: Optional filter for language (java, typescript, javascript, go, rust, c, cpp)
        """
        super().__init__()
        self.language = language

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Multi-SWE-bench examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        try:
            docs = self.load_dataset(
                dataset_name="ByteDance-Seed/Multi-SWE-bench",
                split="train",
                limit=max_items * 2 if max_items else None,
            )
            log.info(f"Loaded {len(docs)} examples from Multi-SWE-bench")
        except Exception as e:
            log.error(f"Failed to load Multi-SWE-bench: {e}")
            return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            # Filter by language if specified
            if self.language:
                instance_id = doc.get("instance_id", "")
                # Language is typically part of repo name or instance_id
                if self.language.lower() not in instance_id.lower():
                    continue

            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid Multi-SWE-bench pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.
        """
        try:
            org = doc.get("org", "").strip()
            repo = doc.get("repo", "").strip()
            title = doc.get("title", "").strip()
            body = doc.get("body", "").strip()
            fix_patch = doc.get("fix_patch", "").strip()
            test_patch = doc.get("test_patch", "")
            instance_id = doc.get("instance_id", "").strip()
            number = doc.get("number", 0)

            if not body and not title:
                log.debug("Skipping: missing title and body")
                return None

            if not fix_patch:
                log.debug("Skipping: missing fix_patch")
                return None

            # Build the task prompt
            task_prompt = self._build_task_prompt(org, repo, title, body, number)

            # Positive = correct patch
            correct_response = self._create_correct_response(fix_patch)
            # Negative = incorrect patch
            incorrect_response = self._create_incorrect_response(title)

            # Detect language from repo or instance_id
            language = self._detect_language(repo, instance_id)

            metadata = {
                "label": "multi_swe_bench",
                "source": "ByteDance-Seed/Multi-SWE-bench",
                "org": org,
                "repo": repo,
                "instance_id": instance_id,
                "language": language,
                "pr_number": number,
                "is_software_engineering_benchmark": True,
                "is_multilingual_benchmark": True,
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

    def _detect_language(self, repo: str, instance_id: str) -> str:
        """Detect programming language from repo name or instance_id."""
        combined = f"{repo} {instance_id}".lower()
        for lang in MULTI_SWE_BENCH_LANGUAGES:
            if lang in combined:
                return lang
        return "unknown"

    def _build_task_prompt(
        self,
        org: str,
        repo: str,
        title: str,
        body: str,
        number: int,
    ) -> str:
        """Build the software engineering task prompt."""
        parts = [
            f"Repository: {org}/{repo}",
            f"PR #{number}: {title}",
        ]

        if body:
            parts.append(f"\n## Description\n{body}")

        parts.append(
            "\n## Task\nProvide a patch that addresses this pull request. "
            "The patch should correctly implement the required changes."
        )

        return "\n".join(parts)

    def _create_correct_response(self, patch: str) -> str:
        """Create a response with the correct patch."""
        return (
            f"Here is the patch to address this PR:\n\n"
            f"```diff\n{patch}\n```\n\n"
            "This patch implements the required changes while maintaining "
            "code quality and test coverage."
        )

    def _create_incorrect_response(self, title: str) -> str:
        """Create an incorrect response."""
        return (
            "I looked at the PR but here's an incomplete attempt:\n\n"
            "```diff\n"
            "- // Original implementation\n"
            "+ // Attempted fix - needs more work\n"
            "+ throw new Error('Not implemented');\n"
            "```\n\n"
            "This solution is incomplete and doesn't properly address the requirements."
        )

